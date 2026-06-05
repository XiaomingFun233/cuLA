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

"""CuTe DSL placeholder for Qwen3.5 scalar-gated KDA decode."""

from __future__ import annotations

import torch

from cula.ops.kda_decode import kda_decode


def qwen35_scalar_kda_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    *,
    state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token scalar-gated delta-rule decode for Qwen3.5."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q/k/v must be 4D, got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got q={tuple(q.shape)} vs k={tuple(k.shape)}")
    if q.shape[1] != 1 or v.shape[1] != 1:
        raise ValueError(f"Decode expects single-token sequence dim, got q={tuple(q.shape)} v={tuple(v.shape)}")

    N, _, HV, K = q.shape
    if a.ndim == 2:
        a = a.unsqueeze(1)
    if b.ndim == 2:
        b = b.unsqueeze(1)
    if a.shape != (N, 1, HV) or b.shape != (N, 1, HV):
        raise ValueError(f"a/b must be [N,1,HV], got a={tuple(a.shape)} b={tuple(b.shape)}")
    if A_log.shape != (HV,) or dt_bias.shape != (HV,):
        raise ValueError(f"A_log/dt_bias must be [HV], got A_log={tuple(A_log.shape)} dt_bias={tuple(dt_bias.shape)}")

    a_expanded = a.unsqueeze(-1).expand(N, 1, HV, K)
    dt_bias_expanded = dt_bias[:, None].expand(HV, K).contiguous()
    state_indices = (
        torch.arange(N, device=q.device, dtype=torch.int32)
        if state_indices is None
        else state_indices.to(device=q.device, dtype=torch.int32)
    )
    o = kda_decode(
        A_log=A_log.contiguous(),
        dt_bias=dt_bias_expanded,
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        a=a_expanded.contiguous(),
        b=b.contiguous(),
        initial_state_source=recurrent_state,
        initial_state_indices=state_indices,
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
        state_layout="kv",
    )
    return o, recurrent_state
