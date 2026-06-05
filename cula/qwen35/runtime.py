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

"""Runtime dispatch for Qwen3.5 linear-attention kernels."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import torch

from cula.qwen35.common import (
    DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
    Qwen35LinearAttentionConfig,
    infer_local_config,
    validate_mixed_qkv,
    validate_scalar_gate_inputs,
    validate_state_tensors,
)
from cula.ops.qwen35_conv1d_decode import qwen35_conv1d_decode_update
from cula.ops.qwen35_scalar_kda_decode import qwen35_scalar_kda_decode

_stream_cache: dict[tuple[str, int], cuda.CUstream] = {}


def _get_cached_stream(device: torch.device) -> cuda.CUstream:
    stream_id = int(torch.cuda.current_stream(device=device).cuda_stream)
    cache_key = (str(device), stream_id)
    if cache_key not in _stream_cache:
        _stream_cache[cache_key] = cuda.CUstream(stream_id)
    return _stream_cache[cache_key]


def qwen35_linear_attention_prefill(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
    cu_seqlens: torch.Tensor | None = None,
    recurrent_state: torch.Tensor | None = None,
    conv_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Qwen3.5 prefill wrapper.

    This is a thin runtime boundary. The underlying CuTe kernels are added in
    dedicated `cula.ops.qwen35_*` modules.
    """

    del conv_weight, A_log, dt_bias, cu_seqlens
    validate_mixed_qkv(mixed_qkv, config)
    validate_scalar_gate_inputs(a, b, config)
    validate_state_tensors(conv_state, recurrent_state, config)
    _get_cached_stream(mixed_qkv.device)
    raise NotImplementedError("Qwen3.5 prefill kernel path is not implemented yet.")


def qwen35_linear_attention_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    config: Qwen35LinearAttentionConfig = DEFAULT_QWEN35_LINEAR_ATTN_CONFIG,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Qwen3.5 decode wrapper.

    Args:
        mixed_qkv: [tokens, local_conv_dim]
        a, b: [tokens, local_num_v_heads]
        conv_weight: [local_conv_dim, 1, 4] or [local_conv_dim, 4]
        A_log, dt_bias: [local_num_v_heads]
        conv_state: [tokens, local_conv_dim, 4]
        recurrent_state: [pool, local_num_v_heads, 128, 128]

    Returns:
        - core_attn_out_flat: [tokens, local_value_dim]
        - updated_conv_state
        - updated_recurrent_state
    """

    validate_mixed_qkv(mixed_qkv, config)
    validate_scalar_gate_inputs(a, b, config)
    validate_state_tensors(conv_state, recurrent_state, config)
    _get_cached_stream(mixed_qkv.device)

    if mixed_qkv.shape[0] != a.shape[0]:
        raise ValueError(f"Token dimension mismatch, got mixed_qkv={tuple(mixed_qkv.shape)} a={tuple(a.shape)}")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError(f"A_log and dt_bias must be 1D, got {tuple(A_log.shape)} and {tuple(dt_bias.shape)}")
    if A_log.shape != dt_bias.shape:
        raise ValueError(f"A_log and dt_bias must have the same shape, got {tuple(A_log.shape)} vs {tuple(dt_bias.shape)}")

    tokens = mixed_qkv.shape[0]
    local_num_v_heads = a.shape[1]
    local_key_dim, local_value_dim, local_num_k_heads = infer_local_config(
        mixed_qkv.shape[1],
        local_num_v_heads,
        config=config,
    )
    if conv_state.shape != (tokens, mixed_qkv.shape[1], config.conv_kernel_size):
        raise ValueError(
            f"conv_state must be [tokens, local_conv_dim, {config.conv_kernel_size}], got {tuple(conv_state.shape)}"
        )
    if recurrent_state.shape[1:] != (local_num_v_heads, config.head_k_dim, config.head_v_dim):
        raise ValueError(
            "recurrent_state must be [pool, local_num_v_heads, head_k_dim, head_v_dim], "
            f"got {tuple(recurrent_state.shape)}"
        )
    if A_log.numel() != local_num_v_heads:
        raise ValueError(f"A_log must match local_num_v_heads={local_num_v_heads}, got {A_log.numel()}")

    conv_out, conv_state_out = qwen35_conv1d_decode_update(
        mixed_qkv,
        conv_state,
        conv_weight,
        activation="silu",
    )

    q_end = local_key_dim
    k_end = q_end + local_key_dim
    q_flat = conv_out[:, :q_end]
    k_flat = conv_out[:, q_end:k_end]
    v_flat = conv_out[:, k_end:]

    q = q_flat.view(tokens, local_num_k_heads, config.head_k_dim)
    k = k_flat.view(tokens, local_num_k_heads, config.head_k_dim)
    v = v_flat.view(tokens, local_num_v_heads, config.head_v_dim)

    repeat_factor = local_num_v_heads // local_num_k_heads
    q = q.repeat_interleave(repeat_factor, dim=1).unsqueeze(1).contiguous()
    k = k.repeat_interleave(repeat_factor, dim=1).unsqueeze(1).contiguous()
    v = v.unsqueeze(1).contiguous()

    core_attn_out, recurrent_state_out = qwen35_scalar_kda_decode(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        state_indices=state_indices,
    )
    core_attn_out = core_attn_out.reshape(tokens, local_value_dim)
    return core_attn_out, conv_state_out, recurrent_state_out
