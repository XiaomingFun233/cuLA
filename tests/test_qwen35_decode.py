#!/usr/bin/env python3
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

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.ops.qwen35_conv1d_decode import qwen35_conv1d_decode_reference, qwen35_conv1d_decode_update
from cula.ops.qwen35_layout_decode import qwen35_layout_decode, qwen35_layout_decode_reference
from cula.ops.qwen35_scalar_kda_decode import qwen35_layout_scalar_kda_decode, qwen35_scalar_kda_decode
from cula.qwen35.common import DEFAULT_QWEN35_LINEAR_ATTN_CONFIG
from cula.qwen35.runtime import qwen35_linear_attention_decode

try:
    import cula.cudac as cula_cuda
except ImportError:
    cula_cuda = None


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _has_qwen35_cudac():
    return (
        torch.cuda.is_available()
        and cula_cuda is not None
        and hasattr(cula_cuda, "qwen35_conv1d_decode")
        and hasattr(cula_cuda, "qwen35_layout_decode")
        and hasattr(cula_cuda, "qwen35_scalar_kda_decode")
    )


def _has_qwen35_fused_layout_kda_cudac():
    return _has_qwen35_cudac() and hasattr(cula_cuda, "qwen35_layout_scalar_kda_decode")


def make_inputs(tokens: int = 2, pool_size: int = 3, device: torch.device | None = None):
    device = _device() if device is None else device
    config = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG
    torch.manual_seed(0)
    mixed_qkv = torch.randn(tokens, config.conv_dim, device=device, dtype=config.qkv_dtype)
    a = torch.randn(tokens, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    b = torch.randn(tokens, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    conv_weight = torch.randn(config.conv_dim, config.conv_kernel_size, device=device, dtype=config.qkv_dtype)
    conv_state = torch.randn(tokens, config.conv_dim, config.conv_kernel_size, device=device, dtype=config.qkv_dtype)
    recurrent_state = torch.randn(
        pool_size,
        config.num_v_heads,
        config.head_k_dim,
        config.head_v_dim,
        device=device,
        dtype=config.state_dtype,
    ) * 0.01
    A_log = -torch.rand(config.num_v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(config.num_v_heads, device=device, dtype=torch.float32) * 0.1
    state_indices = torch.arange(tokens, device=device, dtype=torch.int32) % pool_size
    return mixed_qkv, a, b, conv_weight, conv_state, recurrent_state, A_log, dt_bias, state_indices


def manual_conv_decode(x_t: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor):
    state_tail = conv_state[..., 1:].float()
    window = torch.cat([state_tail, x_t.unsqueeze(-1).float()], dim=-1)
    conv = (window * weight.float().unsqueeze(0)).sum(dim=-1)
    y = torch.nn.functional.silu(conv).to(dtype=x_t.dtype)
    state_new = conv_state.clone()
    state_new[..., 0] = conv_state[..., 1]
    state_new[..., 1] = conv_state[..., 2]
    state_new[..., 2] = conv_state[..., 3]
    state_new[..., 3] = x_t
    return y, state_new


def manual_qwen35_decode_reference(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
):
    config = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG
    conv_out, conv_state_out = manual_conv_decode(mixed_qkv, conv_state, conv_weight)
    q_end = config.key_dim
    k_end = q_end + config.key_dim
    q = conv_out[:, :q_end].view(mixed_qkv.shape[0], config.num_k_heads, config.head_k_dim)
    k = conv_out[:, q_end:k_end].view(mixed_qkv.shape[0], config.num_k_heads, config.head_k_dim)
    v = conv_out[:, k_end:].view(mixed_qkv.shape[0], config.num_v_heads, config.head_v_dim)
    q_rep = q.repeat_interleave(config.qk_repeat_factor, dim=1)
    k_rep = k.repeat_interleave(config.qk_repeat_factor, dim=1)

    scale = config.head_k_dim**-0.5
    q_f = torch.nn.functional.normalize(q_rep.float(), dim=-1) * scale
    k_f = torch.nn.functional.normalize(k_rep.float(), dim=-1)
    v_f = v.float()
    state_out = recurrent_state.clone()
    out = torch.empty(mixed_qkv.shape[0], config.value_dim, device=mixed_qkv.device, dtype=mixed_qkv.dtype)

    for token_idx in range(mixed_qkv.shape[0]):
        per_token = []
        pool_idx = int(state_indices[token_idx].item())
        for hv in range(config.num_v_heads):
            state_kv = state_out[pool_idx, hv]
            decay = torch.exp(-torch.exp(A_log[hv]) * torch.nn.functional.softplus(a[token_idx, hv].float() + dt_bias[hv]))
            beta = torch.sigmoid(b[token_idx, hv].float())
            k_vec = k_f[token_idx, hv]
            q_vec = q_f[token_idx, hv]
            proj = decay * (state_kv.transpose(0, 1) @ k_vec)
            v_new = beta * (v_f[token_idx, hv] - proj)
            state_new_kv = decay * state_kv + k_vec.unsqueeze(1) * v_new.unsqueeze(0)
            per_token.append((state_new_kv.transpose(0, 1) @ q_vec).to(mixed_qkv.dtype))
            state_out[pool_idx, hv] = state_new_kv
        out[token_idx] = torch.cat(per_token, dim=0)
    return out, conv_state_out, state_out


@pytest.mark.parametrize("tokens", [1, 2])
def test_qwen35_conv_decode_reference(tokens: int):
    mixed_qkv, _, _, conv_weight, conv_state, _, _, _, _ = make_inputs(tokens=tokens)
    y_ref, state_ref = manual_conv_decode(mixed_qkv, conv_state, conv_weight)
    y_op, state_op = qwen35_conv1d_decode_update(mixed_qkv, conv_state, conv_weight, backend="reference")
    assert torch.equal(y_ref, y_op)
    assert torch.equal(state_ref, state_op)
    y_ref2, state_ref2 = qwen35_conv1d_decode_reference(mixed_qkv, conv_state, conv_weight)
    assert torch.equal(y_ref, y_ref2)
    assert torch.equal(state_ref, state_ref2)


def test_qwen35_layout_decode_reference():
    mixed_qkv, a, b, _, _, _, _, _, _ = make_inputs(tokens=2)
    q_rep_ref, k_rep_ref, v_ref, a_ref, b_ref = qwen35_layout_decode_reference(mixed_qkv, a, b)
    q_rep, k_rep, v, a_kernel, b_kernel = qwen35_layout_decode(mixed_qkv, a, b, backend="reference")
    assert torch.equal(q_rep_ref, q_rep)
    assert torch.equal(k_rep_ref, k_rep)
    assert torch.equal(v_ref, v)
    assert torch.equal(a_ref, a_kernel)
    assert torch.equal(b_ref, b_kernel)


@pytest.mark.parametrize("tokens", [1, 2])
def test_qwen35_decode_reference_chain(tokens: int):
    mixed_qkv, a, b, conv_weight, conv_state, recurrent_state, A_log, dt_bias, state_indices = make_inputs(tokens=tokens)
    out_ref, conv_state_ref, recurrent_state_ref = manual_qwen35_decode_reference(
        mixed_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        conv_state,
        recurrent_state,
        state_indices,
    )
    out, conv_state_out, recurrent_state_out = qwen35_linear_attention_decode(
        mixed_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        backend="reference",
    )

    assert torch.allclose(out_ref.float(), out.float(), atol=1e-5, rtol=1e-5)
    assert torch.equal(conv_state_ref, conv_state_out)
    assert torch.allclose(recurrent_state_ref, recurrent_state_out, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not _has_qwen35_cudac(), reason="Qwen3.5 CUDA decode backend is not available")
@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_qwen35_decode_cudac_matches_reference(tokens: int):
    # Decode batches represent distinct active sequences, so keep state rows unique
    # to avoid intentionally racing multiple token updates against one cache row.
    mixed_qkv, a, b, conv_weight, conv_state, recurrent_state, A_log, dt_bias, state_indices = make_inputs(
        tokens=tokens,
        pool_size=max(tokens, 3),
        device=torch.device("cuda"),
    )
    out_ref, conv_state_ref, recurrent_state_ref = qwen35_linear_attention_decode(
        mixed_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        backend="reference",
    )
    out, conv_state_out, recurrent_state_out = qwen35_linear_attention_decode(
        mixed_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        backend="cudac",
    )

    torch.cuda.synchronize()
    assert torch.allclose(out_ref.float(), out.float(), atol=3e-2, rtol=3e-2)
    assert torch.equal(conv_state_ref, conv_state_out)
    assert torch.allclose(recurrent_state_ref, recurrent_state_out, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not _has_qwen35_fused_layout_kda_cudac(), reason="Qwen3.5 fused layout+KDA CUDA backend is not available")
@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_qwen35_fused_layout_kda_cudac_matches_unfused(tokens: int):
    mixed_qkv, a, b, conv_weight, conv_state, recurrent_state, A_log, dt_bias, state_indices = make_inputs(
        tokens=tokens,
        pool_size=max(tokens, 3),
        device=torch.device("cuda"),
    )
    conv_out, _ = qwen35_conv1d_decode_update(
        mixed_qkv,
        conv_state,
        conv_weight,
        activation="silu",
        backend="cudac",
    )
    q_rep, k_rep, v, a_kernel, b_kernel = qwen35_layout_decode(conv_out, a, b, backend="cudac")
    out_unfused, state_unfused = qwen35_scalar_kda_decode(
        q=q_rep.unsqueeze(1),
        k=k_rep.unsqueeze(1),
        v=v.unsqueeze(1),
        a=a_kernel,
        b=b_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        backend="cudac",
    )
    out_fused, state_fused = qwen35_layout_scalar_kda_decode(
        mixed_qkv_conv=conv_out,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
        backend="cudac",
    )

    torch.cuda.synchronize()
    assert torch.equal(out_unfused, out_fused)
    assert torch.equal(state_unfused, state_fused)


@pytest.mark.skipif(not _has_qwen35_cudac(), reason="Qwen3.5 CUDA decode backend is not available")
def test_qwen35_decode_cudac_rejects_duplicate_state_indices():
    mixed_qkv, a, b, conv_weight, conv_state, recurrent_state, A_log, dt_bias, _ = make_inputs(
        tokens=2,
        pool_size=3,
        device=torch.device("cuda"),
    )
    state_indices = torch.zeros(2, device=mixed_qkv.device, dtype=torch.int32)

    with pytest.raises(ValueError, match="requires unique state_indices"):
        qwen35_linear_attention_decode(
            mixed_qkv,
            a,
            b,
            conv_weight,
            A_log,
            dt_bias,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            state_indices=state_indices,
            backend="cudac",
        )
