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

"""Benchmark Qwen3.5 prefill kernels.

Reports:
  - layout: cuLA Qwen3.5 prefill layout split/repeat kernel
  - scalar_kda: cuLA Qwen3.5 scalar-gated KDA prefill kernel
  - fla_gdr: optional FLA chunk_gated_delta_rule baseline
  - sgl_gdr: optional SGLang vendored Triton chunk_gated_delta_rule baseline

Baselines are optional. SGLang Qwen3.5 prefill uses the same chunked gated
delta rule family in its Triton GDN kernel; decode uses a recurrent packed
kernel instead.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pathlib
import statistics
import sys
from collections.abc import Callable

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cula.ops.qwen35_layout_prefill import qwen35_layout_prefill
from cula.ops.qwen35_scalar_kda_prefill import qwen35_scalar_kda_prefill
from cula.qwen35.common import DEFAULT_QWEN35_LINEAR_ATTN_CONFIG as CONFIG


def benchmark_cuda_fn(fn: Callable[[], object], *, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for idx in range(rep):
        starts[idx].record()
        fn()
        ends[idx].record()
    torch.cuda.synchronize()

    times = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    if len(times) <= 2:
        return statistics.mean(times)
    times = sorted(times)
    return statistics.mean(times[len(times) // 4 : 3 * len(times) // 4])


def error_stats(ref: torch.Tensor, out: torch.Tensor) -> tuple[float, float, float]:
    ref_f = ref.float()
    out_f = out.float()
    diff = (ref_f - out_f).abs()
    rmse = diff.square().mean().sqrt().item()
    ref_rms = ref_f.square().mean().sqrt().item()
    rel_rms = rmse / (ref_rms + 1.0e-8)
    rel_max = diff.max().item() / (ref_f.abs().max().item() + 1.0e-8)
    mean_abs = diff.mean().item()
    return rel_rms, rel_max, mean_abs


def resolve_fla_chunk_gdr():
    try:
        module = importlib.import_module("fla.ops.gated_delta_rule")
    except ImportError as exc:
        return None, f"cannot import fla.ops.gated_delta_rule: {exc}"
    if not hasattr(module, "chunk_gated_delta_rule"):
        return None, "fla.ops.gated_delta_rule has no chunk_gated_delta_rule"
    return module.chunk_gated_delta_rule, "fla.ops.gated_delta_rule.chunk_gated_delta_rule"


def resolve_sgl_chunk_gdr(sglang_path: pathlib.Path | None):
    if sglang_path is not None:
        for root in (sglang_path, sglang_path / "python"):
            if root.exists():
                sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("sglang.srt.layers.attention.fla.chunk")
    except ImportError as exc:
        return None, f"cannot import sglang.srt.layers.attention.fla.chunk: {exc}"
    if not hasattr(module, "chunk_gated_delta_rule"):
        return None, "sglang.srt.layers.attention.fla.chunk has no chunk_gated_delta_rule"
    return module.chunk_gated_delta_rule, "sglang.srt.layers.attention.fla.chunk.chunk_gated_delta_rule"


def make_inputs(batch: int, seq_len: int, *, device: torch.device, seed: int):
    torch.manual_seed(seed)
    q = torch.randn(batch, seq_len, CONFIG.num_v_heads, CONFIG.head_k_dim, device=device, dtype=CONFIG.qkv_dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(batch, seq_len, CONFIG.num_v_heads, device=device, dtype=CONFIG.qkv_dtype)
    b = torch.randn(batch, seq_len, CONFIG.num_v_heads, device=device, dtype=CONFIG.qkv_dtype)
    beta = torch.sigmoid(b.float()).to(dtype=CONFIG.qkv_dtype)
    A_log = -torch.rand(CONFIG.num_v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(CONFIG.num_v_heads, device=device, dtype=torch.float32) * 0.1
    log_gate = (-torch.exp(A_log).view(1, 1, -1) * torch.nn.functional.softplus(a.float() + dt_bias.view(1, 1, -1))).to(
        dtype=CONFIG.qkv_dtype
    )
    initial_state = torch.randn(
        batch,
        CONFIG.num_v_heads,
        CONFIG.head_k_dim,
        CONFIG.head_v_dim,
        device=device,
        dtype=torch.float32,
    ) * 0.01
    mixed_qkv_conv = torch.randn(batch * seq_len, CONFIG.conv_dim, device=device, dtype=CONFIG.qkv_dtype)
    a_flat = a.reshape(batch * seq_len, CONFIG.num_v_heads).contiguous()
    b_flat = b.reshape(batch * seq_len, CONFIG.num_v_heads).contiguous()
    return q, k, v, a, b, beta, log_gate, A_log, dt_bias, initial_state, mixed_qkv_conv, a_flat, b_flat


def run_cula_scalar(q, k, v, a, b, A_log, dt_bias, initial_state):
    return qwen35_scalar_kda_prefill(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state=initial_state,
        backend="cudac",
    )


def run_chunk_gdr(chunk_gdr, q, k, v, log_gate, beta, initial_state, initial_state_indices):
    # SGLang/FLA GDR chunk kernels use [N, H, V, K] state layout. cuLA's
    # Qwen3.5 wrapper uses [N, H, K, V], so pass the transposed view here.
    initial_state_vk = initial_state.transpose(-1, -2).contiguous()
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=log_gate,
        beta=beta,
        initial_state=initial_state_vk,
        initial_state_indices=initial_state_indices,
        output_final_state=True,
        scale=CONFIG.head_k_dim**-0.5,
        use_qk_l2norm_in_kernel=True,
        head_first=False,
    )
    try:
        sig = inspect.signature(chunk_gdr)
        kwargs = {key: value for key, value in kwargs.items() if key in sig.parameters}
    except (TypeError, ValueError):
        pass
    return chunk_gdr(**kwargs)


def _normalize_chunk_result(result):
    if isinstance(result, tuple):
        out = result[0]
        state = result[-1] if len(result) >= 2 else None
        return out, state
    return result, None


def _state_to_cula_layout(state: torch.Tensor | None) -> torch.Tensor | None:
    if state is None:
        return None
    return state.transpose(-1, -2).contiguous()


def print_header(device: torch.device, args: argparse.Namespace, baseline_sources: dict[str, str]) -> None:
    print("Qwen3.5 prefill benchmark")
    print(f"  device: {torch.cuda.get_device_name(device)}")
    print(f"  dtype: {CONFIG.qkv_dtype}")
    print(f"  batch: {args.batch}")
    print(f"  seq lens: {args.seq_lens}")
    print(f"  warmup/rep: {args.warmup}/{args.rep}")
    print(f"  baselines: {baseline_sources or 'disabled/unavailable'}")
    print()
    print(
        f"{'baseline':>8} {'B':>3} {'T':>7} {'layout_ms':>11} {'cula_kda_ms':>12} {'cula_total':>11} "
        f"{'base_ms':>11} {'speedup':>9} {'rel_rms':>10} {'rel_max':>10}"
    )
    print("-" * 108)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", choices=["none", "fla", "sgl", "all"], default="sgl")
    parser.add_argument("--sglang-path", type=pathlib.Path, default=None)
    parser.add_argument("--skip-accuracy", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    device = torch.device("cuda")

    baselines: dict[str, Callable] = {}
    baseline_sources: dict[str, str] = {}
    if args.baseline in ("fla", "all"):
        fla_chunk_gdr, fla_source_or_error = resolve_fla_chunk_gdr()
        if fla_chunk_gdr is None:
            print(f"Skipping FLA baseline: {fla_source_or_error}")
        else:
            baselines["fla"] = fla_chunk_gdr
            baseline_sources["fla"] = fla_source_or_error
    if args.baseline in ("sgl", "all"):
        sgl_chunk_gdr, sgl_source_or_error = resolve_sgl_chunk_gdr(args.sglang_path)
        if sgl_chunk_gdr is None:
            print(f"Skipping SGLang baseline: {sgl_source_or_error}")
        else:
            baselines["sgl"] = sgl_chunk_gdr
            baseline_sources["sgl"] = sgl_source_or_error

    print_header(device, args, baseline_sources)

    for seq_len in args.seq_lens:
        q, k, v, a, b, beta, log_gate, A_log, dt_bias, initial_state, mixed_qkv_conv, a_flat, b_flat = make_inputs(
            args.batch,
            seq_len,
            device=device,
            seed=args.seed,
        )
        initial_state_indices = torch.arange(args.batch, device=device, dtype=torch.int32)

        def layout_fn():
            return qwen35_layout_prefill(mixed_qkv_conv, a_flat, b_flat, backend="cudac")

        def cula_kda_fn():
            return run_cula_scalar(q, k, v, a, b, A_log, dt_bias, initial_state)

        layout_ms = benchmark_cuda_fn(layout_fn, warmup=args.warmup, rep=args.rep)
        cula_kda_ms = benchmark_cuda_fn(cula_kda_fn, warmup=args.warmup, rep=args.rep)
        cula_total_ms = layout_ms + cula_kda_ms

        if not baselines:
            print(
                f"{'none':>8} {args.batch:3d} {seq_len:7d} {layout_ms:11.4f} {cula_kda_ms:12.4f} {cula_total_ms:11.4f} "
                f"{float('nan'):11.4f} {float('nan'):9.3f} {float('nan'):10.3e} {float('nan'):10.3e}"
            )

        for baseline_name, chunk_gdr in baselines.items():
            def baseline_fn():
                return run_chunk_gdr(chunk_gdr, q, k, v, log_gate, beta, initial_state, initial_state_indices)

            rel_rms = float("nan")
            rel_max = float("nan")
            if not args.skip_accuracy:
                out_cula, state_cula = cula_kda_fn()
                out_base, state_base = _normalize_chunk_result(baseline_fn())
                state_base = _state_to_cula_layout(state_base)
                torch.cuda.synchronize()
                rel_rms, rel_max, _ = error_stats(out_base, out_cula)
                if state_base is not None and tuple(state_base.shape) == tuple(state_cula.shape):
                    rel_rms_s, rel_max_s, _ = error_stats(state_base, state_cula)
                    rel_rms = max(rel_rms, rel_rms_s)
                    rel_max = max(rel_max, rel_max_s)

            base_ms = benchmark_cuda_fn(baseline_fn, warmup=args.warmup, rep=args.rep)
            speedup = base_ms / cula_kda_ms if cula_kda_ms > 0 else float("inf")
            print(
                f"{baseline_name:>8} {args.batch:3d} {seq_len:7d} {layout_ms:11.4f} {cula_kda_ms:12.4f} {cula_total_ms:11.4f} "
                f"{base_ms:11.4f} {speedup:9.3f} {rel_rms:10.3e} {rel_max:10.3e}"
            )

        del q, k, v, a, b, beta, log_gate, A_log, dt_bias, initial_state, initial_state_indices, mixed_qkv_conv, a_flat, b_flat
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
