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

"""CuTe DSL kernel for Qwen3.5 single-token conv-state update."""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

THREADS = 256
KERNEL_SIZE = 4


@cute.kernel
def _qwen35_conv1d_decode_kernel(
    x_t: cute.Tensor,
    conv_state: cute.Tensor,
    weight: cute.Tensor,
    y: cute.Tensor,
    B: cutlass.Constexpr[int],
    C: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    linear_idx = bidx * THREADS + tidx
    if linear_idx < B * C:
        b = linear_idx // C
        c = linear_idx % C

        s0 = cutlass.Float32(conv_state[(b, c, 1)])
        s1 = cutlass.Float32(conv_state[(b, c, 2)])
        s2 = cutlass.Float32(conv_state[(b, c, 3)])
        s3 = cutlass.Float32(x_t[(b, c)])

        w0 = cutlass.Float32(weight[(c, 0)])
        w1 = cutlass.Float32(weight[(c, 1)])
        w2 = cutlass.Float32(weight[(c, 2)])
        w3 = cutlass.Float32(weight[(c, 3)])

        out = s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3
        sig = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-out))
        out = out * sig

        conv_state[(b, c, 0)] = cutlass.BFloat16(s0)
        conv_state[(b, c, 1)] = cutlass.BFloat16(s1)
        conv_state[(b, c, 2)] = cutlass.BFloat16(s2)
        conv_state[(b, c, 3)] = cutlass.BFloat16(s3)
        y[(b, c)] = cutlass.BFloat16(out)


@cute.jit
def _run_qwen35_conv1d_decode(
    x_t: cute.Tensor,
    conv_state: cute.Tensor,
    weight: cute.Tensor,
    y: cute.Tensor,
    B: cutlass.Constexpr[int],
    C: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    _qwen35_conv1d_decode_kernel(
        x_t,
        conv_state,
        weight,
        y,
        B,
        C,
        K,
    ).launch(
        grid=(cute.ceil_div(B * C, THREADS), 1, 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


@functools.cache
def _get_compiled_kernel(
    B: int,
    C: int,
    K: int,
):
    return {}


def qwen35_conv1d_decode_update(
    x_t: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token depthwise causal conv1d update.

    Expected shapes:
    - x_t: [B, C]
    - conv_state: [B, C, 4]
    - weight: [C, 1, 4] or [C, 4]
    """

    if activation != "silu":
        raise ValueError(f"Only silu activation is currently supported, got {activation}")
    if x_t.ndim != 2:
        raise ValueError(f"x_t must be 2D [batch, channels], got {tuple(x_t.shape)}")
    if conv_state.ndim != 3:
        raise ValueError(f"conv_state must be 3D [batch, channels, kernel], got {tuple(conv_state.shape)}")
    if conv_state.shape[:2] != x_t.shape:
        raise ValueError(f"conv_state batch/channel dims must match x_t, got x_t={tuple(x_t.shape)} conv_state={tuple(conv_state.shape)}")
    kernel_size = conv_state.shape[-1]
    if kernel_size != 4:
        raise ValueError(f"Expected kernel_size=4 for Qwen3.5, got {kernel_size}")

    if weight.ndim == 3:
        if weight.shape[1] != 1 or weight.shape[2] != kernel_size:
            raise ValueError(f"weight must be [channels,1,{kernel_size}], got {tuple(weight.shape)}")
        weight_2d = weight.squeeze(1)
    elif weight.ndim == 2:
        if weight.shape[1] != kernel_size:
            raise ValueError(f"weight must be [channels,{kernel_size}], got {tuple(weight.shape)}")
        weight_2d = weight
    else:
        raise ValueError(f"weight must be 2D or 3D, got {tuple(weight.shape)}")

    if weight_2d.shape[0] != x_t.shape[1]:
        raise ValueError(f"weight channels must match x_t channels, got weight={tuple(weight_2d.shape)} x_t={tuple(x_t.shape)}")

    x_t = x_t.contiguous()
    conv_state = conv_state.contiguous()
    weight_2d = weight_2d.contiguous()
    y = torch.empty_like(x_t)

    B, C = x_t.shape
    cache = _get_compiled_kernel(B, C, kernel_size)
    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(
            _run_qwen35_conv1d_decode,
            from_dlpack(x_t, assumed_align=16),
            from_dlpack(conv_state, assumed_align=16),
            from_dlpack(weight_2d, assumed_align=16),
            from_dlpack(y, assumed_align=16),
            B=B,
            C=C,
            K=kernel_size,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cache["compiled"](x_t, conv_state, weight_2d, y, stream)
    return y, conv_state
