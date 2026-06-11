// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "qwen35_prefill_common.cuh"

#include <cute/tensor.hpp>
#include <cuda_runtime.h>

namespace cula::qwen35::prefill::kernel {

using namespace cute;

template <typename scalar_t>
struct Qwen35ScalarKdaPrefillKernel {
  static constexpr int kThreads = 128;
  static constexpr int kHeadDim = kHeadDimQK;
  // Keep the scalar CUDA fallback at one V row per CTA for correctness while
  // the SM90 chunk/TMA path is being wired in.  The previous multi-row V tile
  // version exposed a correctness bug with non-zero initial_state; the chunk
  // path should own the next parallelization step.
  static constexpr int kVTile = 1;
  static constexpr int kNumVTiles = kHeadDimV / kVTile;

  static_assert(kHeadDimQK == 128);
  static_assert(kHeadDimV == 128);
  static_assert(kHeadDimV % kVTile == 0);

  struct SharedStorage {
    float scratch[kThreads];
  };

  static dim3 block_shape() {
    return dim3(kThreads, 1, 1);
  }

  CUTE_HOST_DEVICE static auto make_v_work_tiles(int sequence_count) {
    auto problem_layout = make_layout(
        make_shape(Int<kHeadDimV>{}, Int<kNumVHeads>{}, sequence_count),
        make_stride(Int<1>{}, Int<kHeadDimV>{}, Int<kHeadDimV * kNumVHeads>{}));
    return zipped_divide(problem_layout, make_shape(Int<kVTile>{}, Int<1>{}, Int<1>{}));
  }

  static dim3 grid_shape(int sequence_count) {
    auto v_work_tiles = make_v_work_tiles(sequence_count);
    return dim3(static_cast<unsigned int>(size<1>(v_work_tiles)), 1, 1);
  }

  CUTE_DEVICE static float load_as_float(scalar_t value) {
    return static_cast<float>(value);
  }

  CUTE_DEVICE static scalar_t cast_output(float value) {
    return static_cast<scalar_t>(value);
  }

  CUTE_DEVICE static float softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
  }

  CUTE_DEVICE static float block_sum(float value, SharedStorage& storage, int tid) {
    storage.scratch[tid] = value;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        storage.scratch[tid] += storage.scratch[tid + stride];
      }
      __syncthreads();
    }
    return storage.scratch[0];
  }

  CUTE_DEVICE static void run_device(
      const scalar_t* __restrict__ q,
      const scalar_t* __restrict__ k,
      const scalar_t* __restrict__ v,
      const scalar_t* __restrict__ a,
      const scalar_t* __restrict__ b,
      const float* __restrict__ A_log,
      const float* __restrict__ dt_bias,
      const float* __restrict__ initial_state,
      const int32_t* __restrict__ cu_seqlens,
      scalar_t* __restrict__ out,
      float* __restrict__ final_state,
      int batch_size,
      int seq_len,
      int sequence_count,
      bool is_varlen,
      bool has_initial_state,
      SharedStorage& storage) {
    auto v_work_tiles = make_v_work_tiles(sequence_count);
    auto work_layout = make_layout(get<1>(v_work_tiles.shape()), LayoutLeft{});
    auto work_coord = work_layout.get_hier_coord(static_cast<int>(blockIdx.x));
    const int v_tile_idx = static_cast<int>(get<0>(work_coord));
    const int hv = static_cast<int>(get<1>(work_coord));
    const int seq_idx = static_cast<int>(get<2>(work_coord));
    const int v_base = v_tile_idx * kVTile;
    const int tid = static_cast<int>(threadIdx.x);

    if (hv >= kNumVHeads || seq_idx >= sequence_count) {
      return;
    }

    const int token_begin = is_varlen ? static_cast<int>(cu_seqlens[seq_idx]) : seq_idx * seq_len;
    const int token_end = is_varlen ? static_cast<int>(cu_seqlens[seq_idx + 1]) : token_begin + seq_len;
    const int state_base = ((seq_idx * kNumVHeads + hv) * kHeadDimQK) * kHeadDimV;

    const int kk = tid;
    float state_vals[kVTile];

#pragma unroll
    for (int lane = 0; lane < kVTile; ++lane) {
      const int v_row = v_base + lane;
      const int state_off = state_base + kk * kHeadDimV + v_row;
      state_vals[lane] = 0.0f;
      if (kk < kHeadDimQK && v_row < kHeadDimV) {
        state_vals[lane] = has_initial_state ? initial_state[state_off] : 0.0f;
      }
    }
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(kHeadDimQK));
    const float exp_A = expf(A_log[hv]);
    const float dt = dt_bias[hv];

    for (int token = token_begin; token < token_end; ++token) {
      const int local_t = is_varlen ? token : token - token_begin;
      const int qkv_base = ((token * kNumVHeads + hv) * kHeadDimQK);
      const int gate_base = token * kNumVHeads + hv;

      const float q_val = kk < kHeadDimQK ? load_as_float(q[qkv_base + kk]) : 0.0f;
      const float k_val = kk < kHeadDimQK ? load_as_float(k[qkv_base + kk]) : 0.0f;
      const float q_norm_sq = block_sum(q_val * q_val, storage, tid);
      const float k_norm_sq = block_sum(k_val * k_val, storage, tid);
      const float q_rnorm = rsqrtf(fmaxf(q_norm_sq, 1.0e-20f)) * scale;
      const float k_rnorm = rsqrtf(fmaxf(k_norm_sq, 1.0e-20f));

      const float decay = expf(-exp_A * softplus(load_as_float(a[gate_base]) + dt));
      const float beta = 1.0f / (1.0f + expf(-load_as_float(b[gate_base])));

      const float k_norm = k_val * k_rnorm;
      const float q_norm = q_val * q_rnorm;

#pragma unroll
      for (int lane = 0; lane < kVTile; ++lane) {
        const int v_row = v_base + lane;
        if (v_row < kHeadDimV) {
          const float proj_partial = kk < kHeadDimQK ? state_vals[lane] * k_norm : 0.0f;
          const float proj = block_sum(proj_partial, storage, tid);

          const float v_val = load_as_float(v[qkv_base + v_row]);
          const float v_new = beta * (v_val - decay * proj);

          float out_partial = 0.0f;
          if (kk < kHeadDimQK) {
            const float state_new = decay * state_vals[lane] + k_norm * v_new;
            state_vals[lane] = state_new;
            out_partial = state_new * q_norm;
          }
          const float out_acc = block_sum(out_partial, storage, tid);

          if (tid == 0) {
            const int out_off =
                (((is_varlen ? 0 : seq_idx) * seq_len + local_t) * kNumVHeads + hv) * kHeadDimV + v_row;
            out[out_off] = cast_output(out_acc);
          }
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (int lane = 0; lane < kVTile; ++lane) {
      const int v_row = v_base + lane;
      if (kk < kHeadDimQK && v_row < kHeadDimV) {
        const int state_off = state_base + kk * kHeadDimV + v_row;
        final_state[state_off] = state_vals[lane];
      }
    }

    (void)batch_size;
  }
};

template <typename scalar_t>
__global__ void qwen35_scalar_kda_prefill_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ initial_state,
    const int32_t* __restrict__ cu_seqlens,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    int batch_size,
    int seq_len,
    int sequence_count,
    bool is_varlen,
    bool has_initial_state) {
  __shared__ typename Qwen35ScalarKdaPrefillKernel<scalar_t>::SharedStorage storage;
  Qwen35ScalarKdaPrefillKernel<scalar_t>::run_device(
      q,
      k,
      v,
      a,
      b,
      A_log,
      dt_bias,
      initial_state,
      cu_seqlens,
      out,
      final_state,
      batch_size,
      seq_len,
      sequence_count,
      is_varlen,
      has_initial_state,
      storage);
}

template <typename scalar_t>
void launch_qwen35_scalar_kda_prefill_kernel(
    cudaStream_t stream,
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* a,
    const scalar_t* b,
    const float* A_log,
    const float* dt_bias,
    const float* initial_state,
    const int32_t* cu_seqlens,
    scalar_t* out,
    float* final_state,
    int batch_size,
    int seq_len,
    int sequence_count,
    bool is_varlen,
    bool has_initial_state) {
  const auto grid = Qwen35ScalarKdaPrefillKernel<scalar_t>::grid_shape(sequence_count);
  const auto block = Qwen35ScalarKdaPrefillKernel<scalar_t>::block_shape();
  qwen35_scalar_kda_prefill_kernel<scalar_t><<<grid, block, 0, stream>>>(
      q,
      k,
      v,
      a,
      b,
      A_log,
      dt_bias,
      initial_state,
      cu_seqlens,
      out,
      final_state,
      batch_size,
      seq_len,
      sequence_count,
      is_varlen,
      has_initial_state);
}

} // namespace cula::qwen35::prefill::kernel
