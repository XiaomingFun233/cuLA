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

"""Benchmark Qwen3.5 decode on the active CUDA device.

Two timing scopes are reported:
  - native_core: direct native scalar GDN decode op.
  - triton_core: FLA/SGLang-style fused_sigmoid_gating_delta_rule_update
    Triton decode op vendored in cuLA.
  - sglang_core: fused_sigmoid_gating_delta_rule_update from SGLang, when
    available from the installed package or --sglang-path.
  - fused_layout_kda: direct cuLA fused Qwen3.5 layout + scalar KDA decode op.
  - sglang_packed: SGLang packed Qwen3.5 layout + recurrent update op, when
    available from the installed package or --sglang-path.
  - full: cuLA Python Qwen3.5 decode chain, including conv + layout + core.

State buffers are reset before each timed iteration and the reset copy is not
included in the event timing window.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import inspect
import pathlib
import statistics
import sys
import time
from collections.abc import Callable

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cula.cudac as cula_cuda
from cula.ops.kda_decode_fla import fused_sigmoid_gating_delta_rule_update as triton_fused_sigmoid_update
from cula.qwen35.common import DEFAULT_QWEN35_LINEAR_ATTN_CONFIG as CONFIG
from cula.qwen35.common import Qwen35LinearAttentionConfig
from cula.qwen35.runtime import qwen35_linear_attention_decode

SGLANG_CORE_MODULES = [
    "sglang.srt.layers.attention.linear.kernels.gdn_triton",
    "sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent",
]
SGLANG_CORE_FILES = [
    pathlib.Path("sglang/srt/layers/attention/linear/kernels/gdn_triton.py"),
    pathlib.Path("sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py"),
]
SGLANG_PACKED_MODULES = [
    "sglang.srt.layers.attention.fla.fused_recurrent",
    "sglang.srt.layers.attention.linear.kernels.gdn_triton",
]
SGLANG_PACKED_FILES = [
    pathlib.Path("sglang/srt/layers/attention/fla/fused_recurrent.py"),
    pathlib.Path("sglang/srt/layers/attention/linear/kernels/gdn_triton.py"),
]


def accelerator_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("No CUDA accelerator is available.")


def accelerator_name(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(f"Unsupported device={device}")
    return torch.cuda.get_device_name(device.index or 0)


def synchronize(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError(f"Unsupported device={device}")
    torch.cuda.synchronize()


def benchmark_accel_fn(
    fn: Callable[[], object],
    *,
    device: torch.device,
    setup_fn: Callable[[], None] | None,
    warmup: int,
    rep: int,
) -> float:
    for _ in range(warmup):
        if setup_fn is not None:
            setup_fn()
        fn()
    synchronize(device)

    times: list[float] = []
    try:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for i in range(rep):
            if setup_fn is not None:
                setup_fn()
            starts[i].record()
            fn()
            ends[i].record()
        synchronize(device)
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    except Exception:
        for _ in range(rep):
            if setup_fn is not None:
                setup_fn()
            synchronize(device)
            t0 = time.perf_counter()
            fn()
            synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)

    if not times:
        return 0.0
    if len(times) < 4:
        return statistics.mean(times)
    times = sorted(times)
    iqr = times[len(times) // 4 : 3 * len(times) // 4]
    return statistics.mean(iqr)


def local_config_from_tp_size(tp_size: int) -> Qwen35LinearAttentionConfig:
    if tp_size not in (1, 2, 4, 8):
        raise ValueError(f"tp_size must be one of 1, 2, 4, 8, got {tp_size}")
    return Qwen35LinearAttentionConfig(
        hidden_size=CONFIG.hidden_size // tp_size,
        conv_kernel_size=CONFIG.conv_kernel_size,
        num_k_heads=CONFIG.num_k_heads // tp_size,
        num_v_heads=CONFIG.num_v_heads // tp_size,
        head_k_dim=CONFIG.head_k_dim,
        head_v_dim=CONFIG.head_v_dim,
        qkv_dtype=CONFIG.qkv_dtype,
        state_dtype=CONFIG.state_dtype,
    )


def make_full_inputs(tokens: int, device: torch.device, seed: int, config: Qwen35LinearAttentionConfig):
    torch.manual_seed(seed)
    pool_size = max(tokens, 1)
    mixed_qkv = torch.randn(tokens, config.conv_dim, device=device, dtype=config.qkv_dtype)
    a = torch.randn(tokens, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    b = torch.randn(tokens, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    conv_weight = torch.randn(config.conv_dim, config.conv_kernel_size, device=device, dtype=config.qkv_dtype)
    conv_state = torch.randn(
        tokens,
        config.conv_dim,
        config.conv_kernel_size,
        device=device,
        dtype=config.qkv_dtype,
    )
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
    state_indices = torch.arange(tokens, device=device, dtype=torch.int32)
    return mixed_qkv, a, b, conv_weight, conv_state, recurrent_state, A_log, dt_bias, state_indices


def make_fused_layout_kda_inputs(tokens: int, device: torch.device, seed: int, config: Qwen35LinearAttentionConfig):
    torch.manual_seed(seed)
    mixed_qkv_conv = torch.randn(tokens, config.conv_dim, device=device, dtype=config.qkv_dtype)
    a = torch.randn(tokens, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    b = torch.randn(tokens, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    A_log = -torch.rand(config.num_v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(config.num_v_heads, device=device, dtype=torch.float32) * 0.1
    state = torch.randn(
        tokens,
        config.num_v_heads,
        config.head_k_dim,
        config.head_v_dim,
        device=device,
        dtype=config.state_dtype,
    ) * 0.01
    state_work = torch.empty_like(state)
    state_indices = torch.arange(tokens, device=device, dtype=torch.int32)
    out = torch.empty(tokens, config.num_v_heads, config.head_v_dim, device=device, dtype=config.qkv_dtype)
    return mixed_qkv_conv, a, b, A_log, dt_bias, state, state_work, state_indices, out


def make_core_inputs(tokens: int, device: torch.device, seed: int, config: Qwen35LinearAttentionConfig):
    torch.manual_seed(seed)
    q = torch.randn(tokens, config.num_v_heads, config.head_k_dim, device=device, dtype=config.qkv_dtype)
    k = torch.randn(tokens, config.num_v_heads, config.head_k_dim, device=device, dtype=config.qkv_dtype)
    v = torch.randn(tokens, config.num_v_heads, config.head_v_dim, device=device, dtype=config.qkv_dtype)
    a = torch.randn(tokens, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    b = torch.randn(tokens, config.num_v_heads, device=device, dtype=config.qkv_dtype)
    A_log = -torch.rand(config.num_v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(config.num_v_heads, device=device, dtype=torch.float32) * 0.1
    state = torch.randn(
        tokens,
        config.num_v_heads,
        config.head_k_dim,
        config.head_v_dim,
        device=device,
        dtype=config.state_dtype,
    ) * 0.01
    state_indices = torch.arange(tokens, device=device, dtype=torch.int32)
    out = torch.empty_like(v)
    state_work = torch.empty_like(state)
    return q, k, v, a, b, A_log, dt_bias, state, state_work, state_indices, out


def _add_sglang_import_roots(sglang_path: pathlib.Path | None) -> None:
    if sglang_path is not None:
        for import_root in (sglang_path, sglang_path / "python"):
            if import_root.exists():
                sys.path.insert(0, str(import_root))


def _resolve_sglang_symbol(
    *,
    sglang_path: pathlib.Path | None,
    module_names: list[str],
    file_paths: list[pathlib.Path],
    symbol_names: list[str],
):
    _add_sglang_import_roots(sglang_path)

    import_errors = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, PermissionError, ModuleNotFoundError) as exc:
            import_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        for symbol_name in symbol_names:
            if hasattr(module, symbol_name):
                return getattr(module, symbol_name), f"{module_name}.{symbol_name}"

    if sglang_path is not None:
        candidates: list[pathlib.Path] = []
        for rel_path in file_paths:
            candidates.extend([sglang_path / rel_path, sglang_path / "python" / rel_path])
        for idx, path in enumerate(candidates):
            if path.exists():
                spec = importlib.util.spec_from_file_location(f"_sglang_qwen35_decode_provider_{idx}", path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for symbol_name in symbol_names:
                    if hasattr(module, symbol_name):
                        return getattr(module, symbol_name), f"{path}:{symbol_name}"
        raise RuntimeError(
            f"Could not find any of {symbol_names} under --sglang-path={sglang_path}. "
            "Pass the SGLang repo root or its python/ directory. "
            f"Import errors: {'; '.join(import_errors) or 'none'}"
        )

    return None, None


def resolve_sglang_core_update(sglang_path: pathlib.Path | None):
    """Return SGLang's scalar-gated recurrent update function if available."""
    return _resolve_sglang_symbol(
        sglang_path=sglang_path,
        module_names=SGLANG_CORE_MODULES,
        file_paths=SGLANG_CORE_FILES,
        symbol_names=["fused_sigmoid_gating_delta_rule_update"],
    )


def resolve_sglang_packed_decode(sglang_path: pathlib.Path | None):
    """Return SGLang's packed layout + recurrent decode function if available."""
    return _resolve_sglang_symbol(
        sglang_path=sglang_path,
        module_names=SGLANG_PACKED_MODULES,
        file_paths=SGLANG_PACKED_FILES,
        symbol_names=[
            "fused_recurrent_gated_delta_rule_packed_decode",
            "fused_recurrent_gated_delta_rule_packed_decode_cpu",
        ],
    )


def call_with_supported_kwargs(fn: Callable, **kwargs):
    """Call a provider while tolerating minor SGLang signature drift."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return fn(**kwargs)
    filtered = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return fn(**filtered)


def bench_native_core(tokens: int, device: torch.device, warmup: int, rep: int, seed: int, config: Qwen35LinearAttentionConfig) -> float:
    q, k, v, a, b, A_log, dt_bias, state, state_work, state_indices, out = make_core_inputs(tokens, device, seed, config)

    def setup() -> None:
        state_work.copy_(state)

    def run() -> None:
        cula_cuda.qwen35_scalar_kda_decode(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state_work,
            state_indices,
            out,
        )

    return benchmark_accel_fn(run, device=device, setup_fn=setup, warmup=warmup, rep=rep)


def bench_triton_core(tokens: int, device: torch.device, warmup: int, rep: int, seed: int, config: Qwen35LinearAttentionConfig) -> float:
    q, k, v, a, b, A_log, dt_bias, state, state_work, state_indices, _ = make_core_inputs(tokens, device, seed, config)
    q_4d = q.unsqueeze(1).contiguous()
    k_4d = k.unsqueeze(1).contiguous()
    v_4d = v.unsqueeze(1).contiguous()
    a_3d = a.unsqueeze(1).contiguous()
    b_3d = b.unsqueeze(1).contiguous()

    def setup() -> None:
        state_work.copy_(state)

    def run() -> None:
        triton_fused_sigmoid_update(
            A_log=A_log,
            a=a_3d,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q_4d,
            k=k_4d,
            v=v_4d,
            b=b_3d,
            initial_state_source=state_work,
            initial_state_indices=state_indices,
            scale=config.head_k_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
            is_kda=False,
        )

    return benchmark_accel_fn(run, device=device, setup_fn=setup, warmup=warmup, rep=rep)


def bench_fused_layout_kda(tokens: int, device: torch.device, warmup: int, rep: int, seed: int, config: Qwen35LinearAttentionConfig) -> float:
    mixed_qkv_conv, a, b, A_log, dt_bias, state, state_work, state_indices, out = make_fused_layout_kda_inputs(
        tokens, device, seed, config
    )

    def setup() -> None:
        state_work.copy_(state)

    def run() -> None:
        cula_cuda.qwen35_layout_scalar_kda_decode(
            mixed_qkv_conv,
            a,
            b,
            A_log,
            dt_bias,
            state_work,
            state_indices,
            out,
        )

    return benchmark_accel_fn(run, device=device, setup_fn=setup, warmup=warmup, rep=rep)


def bench_sglang_core(
    tokens: int,
    device: torch.device,
    warmup: int,
    rep: int,
    seed: int,
    sglang_fused_update: Callable,
    config: Qwen35LinearAttentionConfig,
) -> float:
    q, k, v, a, b, A_log, dt_bias, state, _, state_indices, _ = make_core_inputs(tokens, device, seed, config)
    q_4d = q.unsqueeze(1).contiguous()
    k_4d = k.unsqueeze(1).contiguous()
    v_4d = v.unsqueeze(1).contiguous()
    a_3d = a.unsqueeze(1).contiguous()
    b_3d = b.unsqueeze(1).contiguous()
    state_vk = state.transpose(-1, -2).contiguous()
    state_vk_work = torch.empty_like(state_vk)

    def setup() -> None:
        state_vk_work.copy_(state_vk)

    def run() -> None:
        call_with_supported_kwargs(
            sglang_fused_update,
            A_log=A_log,
            a=a_3d,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q_4d,
            k=k_4d,
            v=v_4d,
            b=b_3d,
            initial_state_source=state_vk_work,
            initial_state_indices=state_indices,
            scale=config.head_k_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
            is_kda=False,
        )

    return benchmark_accel_fn(run, device=device, setup_fn=setup, warmup=warmup, rep=rep)


def bench_sglang_packed_layout_kda(
    tokens: int,
    device: torch.device,
    warmup: int,
    rep: int,
    seed: int,
    sglang_packed_decode: Callable,
    config: Qwen35LinearAttentionConfig,
) -> float:
    mixed_qkv_conv, a, b, A_log, dt_bias, state, _, state_indices, _ = make_fused_layout_kda_inputs(tokens, device, seed, config)
    state_vk = state.transpose(-1, -2).contiguous()
    state_vk_work = torch.empty_like(state_vk)
    out = torch.empty(tokens, 1, config.num_v_heads, config.head_v_dim, device=device, dtype=config.qkv_dtype)

    def setup() -> None:
        state_vk_work.copy_(state_vk)

    def run() -> None:
        call_with_supported_kwargs(
            sglang_packed_decode,
            mixed_qkv=mixed_qkv_conv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=config.head_k_dim**-0.5,
            initial_state=state_vk_work,
            out=out,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )

    return benchmark_accel_fn(run, device=device, setup_fn=setup, warmup=warmup, rep=rep)


def bench_full(tokens: int, device: torch.device, warmup: int, rep: int, seed: int, config: Qwen35LinearAttentionConfig) -> float:
    inputs = make_full_inputs(tokens, device, seed, config)
    mixed_qkv, a, b, conv_weight, conv_state, recurrent_state, A_log, dt_bias, state_indices = inputs
    conv_state_work = torch.empty_like(conv_state)
    recurrent_state_work = torch.empty_like(recurrent_state)

    def setup() -> None:
        conv_state_work.copy_(conv_state)
        recurrent_state_work.copy_(recurrent_state)

    def run() -> None:
        qwen35_linear_attention_decode(
            mixed_qkv,
            a,
            b,
            conv_weight,
            A_log,
            dt_bias,
            config=config,
            conv_state=conv_state_work,
            recurrent_state=recurrent_state_work,
            state_indices=state_indices,
            backend="cudac",
        )

    return benchmark_accel_fn(run, device=device, setup_fn=setup, warmup=warmup, rep=rep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cuLA Qwen3.5 decode.")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scope", choices=["core", "fused", "full", "both"], default="both")
    parser.add_argument("--tp-size", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--skip-triton", action="store_true", help="Skip the vendored Triton core timing.")
    parser.add_argument("--skip-sglang", action="store_true", help="Do not try the SGLang kernel provider.")
    parser.add_argument("--require-sglang", action="store_true", help="Fail if the SGLang kernel provider is unavailable.")
    parser.add_argument("--sglang-path", type=pathlib.Path, default=None, help="SGLang repo root or python/ directory.")
    parser.add_argument("--csv", type=pathlib.Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = local_config_from_tp_size(args.tp_size)
    device = accelerator_device()
    rows: list[dict[str, object]] = []
    sglang_fused_update = None
    sglang_core_source = None
    sglang_packed_decode = None
    sglang_packed_source = None
    if not args.skip_sglang:
        sglang_fused_update, sglang_core_source = resolve_sglang_core_update(args.sglang_path)
        sglang_packed_decode, sglang_packed_source = resolve_sglang_packed_decode(args.sglang_path)
    if args.require_sglang and (sglang_fused_update is None or sglang_packed_decode is None):
        raise RuntimeError("SGLang core and packed decode providers must both be available.")

    print(f"device={device} name={accelerator_name(device)} torch={torch.__version__}")
    print(
        f"qwen35: tp={args.tp_size} local_HK={config.num_k_heads} local_HV={config.num_v_heads} "
        f"K={config.head_k_dim} V={config.head_v_dim} conv_dim={config.conv_dim}"
    )
    print(f"sglang_core_provider={sglang_core_source or 'unavailable'}")
    print(f"sglang_packed_provider={sglang_packed_source or 'unavailable'}")
    print("| tokens | native_core_ms | triton_core_ms | sglang_core_ms | fused_layout_kda_ms | sglang_packed_ms | full_ms | triton/native | sglang/native | packed/fused | native_us_per_token | triton_us_per_token | sglang_us_per_token | fused_us_per_token | packed_us_per_token | full_us_per_token |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for tokens in args.tokens:
        native_core_ms = None
        triton_core_ms = None
        sglang_core_ms = None
        fused_layout_kda_ms = None
        sglang_packed_ms = None
        full_ms = None
        if args.scope in ("core", "both"):
            native_core_ms = bench_native_core(tokens, device, args.warmup, args.rep, args.seed, config)
            if not args.skip_triton:
                triton_core_ms = bench_triton_core(tokens, device, args.warmup, args.rep, args.seed, config)
            if sglang_fused_update is not None:
                sglang_core_ms = bench_sglang_core(
                    tokens,
                    device,
                    args.warmup,
                    args.rep,
                    args.seed,
                    sglang_fused_update,
                    config,
                )
        if args.scope in ("fused", "both"):
            fused_layout_kda_ms = bench_fused_layout_kda(tokens, device, args.warmup, args.rep, args.seed, config)
            if sglang_packed_decode is not None:
                sglang_packed_ms = bench_sglang_packed_layout_kda(
                    tokens,
                    device,
                    args.warmup,
                    args.rep,
                    args.seed,
                    sglang_packed_decode,
                    config,
                )
        if args.scope in ("full", "both"):
            full_ms = bench_full(tokens, device, args.warmup, args.rep, args.seed, config)

        native_core_us = None if native_core_ms is None else native_core_ms * 1000.0 / tokens
        triton_core_us = None if triton_core_ms is None else triton_core_ms * 1000.0 / tokens
        sglang_core_us = None if sglang_core_ms is None else sglang_core_ms * 1000.0 / tokens
        fused_layout_kda_us = None if fused_layout_kda_ms is None else fused_layout_kda_ms * 1000.0 / tokens
        sglang_packed_us = None if sglang_packed_ms is None else sglang_packed_ms * 1000.0 / tokens
        full_us = None if full_ms is None else full_ms * 1000.0 / tokens
        triton_ratio = None
        if native_core_ms is not None and triton_core_ms is not None and native_core_ms > 0:
            triton_ratio = triton_core_ms / native_core_ms
        sglang_ratio = None
        if native_core_ms is not None and sglang_core_ms is not None and native_core_ms > 0:
            sglang_ratio = sglang_core_ms / native_core_ms
        packed_ratio = None
        if fused_layout_kda_ms is not None and sglang_packed_ms is not None and fused_layout_kda_ms > 0:
            packed_ratio = sglang_packed_ms / fused_layout_kda_ms
        print(
            f"| {tokens} | "
            f"{'n/a' if native_core_ms is None else f'{native_core_ms:.4f}'} | "
            f"{'n/a' if triton_core_ms is None else f'{triton_core_ms:.4f}'} | "
            f"{'n/a' if sglang_core_ms is None else f'{sglang_core_ms:.4f}'} | "
            f"{'n/a' if fused_layout_kda_ms is None else f'{fused_layout_kda_ms:.4f}'} | "
            f"{'n/a' if sglang_packed_ms is None else f'{sglang_packed_ms:.4f}'} | "
            f"{'n/a' if full_ms is None else f'{full_ms:.4f}'} | "
            f"{'n/a' if triton_ratio is None else f'{triton_ratio:.2f}x'} | "
            f"{'n/a' if sglang_ratio is None else f'{sglang_ratio:.2f}x'} | "
            f"{'n/a' if packed_ratio is None else f'{packed_ratio:.2f}x'} | "
            f"{'n/a' if native_core_us is None else f'{native_core_us:.2f}'} | "
            f"{'n/a' if triton_core_us is None else f'{triton_core_us:.2f}'} | "
            f"{'n/a' if sglang_core_us is None else f'{sglang_core_us:.2f}'} | "
            f"{'n/a' if fused_layout_kda_us is None else f'{fused_layout_kda_us:.2f}'} | "
            f"{'n/a' if sglang_packed_us is None else f'{sglang_packed_us:.2f}'} | "
            f"{'n/a' if full_us is None else f'{full_us:.2f}'} |"
        )
        rows.append(
            {
                "tokens": tokens,
                "tp_size": args.tp_size,
                "local_k_heads": config.num_k_heads,
                "local_v_heads": config.num_v_heads,
                "conv_dim": config.conv_dim,
                "native_core_ms": native_core_ms,
                "triton_core_ms": triton_core_ms,
                "sglang_core_ms": sglang_core_ms,
                "fused_layout_kda_ms": fused_layout_kda_ms,
                "sglang_packed_ms": sglang_packed_ms,
                "full_ms": full_ms,
                "triton_over_native": triton_ratio,
                "sglang_over_native": sglang_ratio,
                "sglang_packed_over_fused": packed_ratio,
                "native_core_us_per_token": native_core_us,
                "triton_core_us_per_token": triton_core_us,
                "sglang_core_us_per_token": sglang_core_us,
                "fused_layout_kda_us_per_token": fused_layout_kda_us,
                "sglang_packed_us_per_token": sglang_packed_us,
                "full_us_per_token": full_us,
            }
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "tokens",
                    "tp_size",
                    "local_k_heads",
                    "local_v_heads",
                    "conv_dim",
                    "native_core_ms",
                    "triton_core_ms",
                    "sglang_core_ms",
                    "fused_layout_kda_ms",
                    "sglang_packed_ms",
                    "full_ms",
                    "triton_over_native",
                    "sglang_over_native",
                    "sglang_packed_over_fused",
                    "native_core_us_per_token",
                    "triton_core_us_per_token",
                    "sglang_core_us_per_token",
                    "fused_layout_kda_us_per_token",
                    "sglang_packed_us_per_token",
                    "full_us_per_token",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
