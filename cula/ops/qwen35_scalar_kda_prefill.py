# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.5 scalar-gated KDA prefill wrapper."""

from __future__ import annotations

import torch

try:
    import cula.cudac as cula_cuda
except ImportError:
    cula_cuda = None


def qwen35_scalar_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked scalar-gated delta-rule prefill for Qwen3.5."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q/k/v must be 4D [B,T,HV,D], got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v must have the same shape, got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    B, T, HV, K = q.shape
    if K != 128 or v.shape[-1] != 128:
        raise ValueError(f"Qwen3.5 prefill expects K=V=128, got q={tuple(q.shape)} v={tuple(v.shape)}")
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0)
    if a.shape != (B, T, HV) or b.shape != (B, T, HV):
        raise ValueError(f"a/b must be [B,T,HV], got a={tuple(a.shape)} b={tuple(b.shape)} expected={(B, T, HV)}")
    if A_log.shape != (HV,) or dt_bias.shape != (HV,):
        raise ValueError(f"A_log/dt_bias must be [HV], got A_log={tuple(A_log.shape)} dt_bias={tuple(dt_bias.shape)}")
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("cu_seqlens mode expects flattened q/k/v with batch size 1")
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
            raise ValueError(f"cu_seqlens must be 1D int32, got {tuple(cu_seqlens.shape)} {cu_seqlens.dtype}")
    if initial_state is not None and initial_state.shape[1:] != (HV, K, K):
        raise ValueError(f"initial_state must be [N,HV,128,128], got {tuple(initial_state.shape)}")

    use_cudac = (
        backend in ("auto", "cudac")
        and cula_cuda is not None
        and hasattr(cula_cuda, "qwen35_scalar_kda_prefill")
        and q.is_cuda
    )
    if backend == "cudac" and not use_cudac:
        raise RuntimeError("Requested backend='cudac' but qwen35_scalar_kda_prefill is not available.")

    if use_cudac:
        if HV != 48:
            raise ValueError(f"backend='cudac' currently expects Qwen3.5 HV=48, got {HV}")
        state_count = B if cu_seqlens is None else cu_seqlens.numel() - 1
        out = torch.empty_like(v)
        final_state = torch.empty(state_count, HV, K, K, device=q.device, dtype=torch.float32)
        initial_state_arg = (
            torch.empty(0, device=q.device, dtype=torch.float32)
            if initial_state is None
            else initial_state.contiguous()
        )
        cu_seqlens_arg = (
            torch.empty(0, device=q.device, dtype=torch.int32)
            if cu_seqlens is None
            else cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        )
        cula_cuda.qwen35_scalar_kda_prefill(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            a.contiguous(),
            b.contiguous(),
            A_log.contiguous(),
            dt_bias.contiguous(),
            initial_state_arg,
            cu_seqlens_arg,
            out,
            final_state,
        )
        return out, final_state

    if backend not in ("auto", "reference"):
        raise ValueError(f"Unsupported backend={backend}")

    state_count = B if cu_seqlens is None else cu_seqlens.numel() - 1
    state = (
        torch.zeros(state_count, HV, K, K, device=q.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.empty_like(v)
    q_f = torch.nn.functional.normalize(q.float(), dim=-1) * (K**-0.5)
    k_f = torch.nn.functional.normalize(k.float(), dim=-1)
    v_f = v.float()
    a_f = a.float()
    b_f = b.float()
    A_log_f = A_log.float()
    dt_bias_f = dt_bias.float()

    def _run_sequence(batch_idx: int, state_idx: int, start: int, end: int) -> None:
        for t in range(start, end):
            for hv in range(HV):
                state_kv = state[state_idx, hv]
                decay = torch.exp(-torch.exp(A_log_f[hv]) * torch.nn.functional.softplus(a_f[batch_idx, t, hv] + dt_bias_f[hv]))
                beta = torch.sigmoid(b_f[batch_idx, t, hv])
                k_vec = k_f[batch_idx, t, hv]
                q_vec = q_f[batch_idx, t, hv]
                proj = decay * (state_kv.transpose(0, 1) @ k_vec)
                v_new = beta * (v_f[batch_idx, t, hv] - proj)
                state_kv_new = decay * state_kv + k_vec.unsqueeze(1) * v_new.unsqueeze(0)
                out[batch_idx, t, hv] = (state_kv_new.transpose(0, 1) @ q_vec).to(out.dtype)
                state[state_idx, hv] = state_kv_new

    if cu_seqlens is None:
        for bidx in range(B):
            _run_sequence(bidx, bidx, 0, T)
    else:
        for sidx in range(state_count):
            _run_sequence(0, sidx, int(cu_seqlens[sidx].item()), int(cu_seqlens[sidx + 1].item()))
    return out, state
