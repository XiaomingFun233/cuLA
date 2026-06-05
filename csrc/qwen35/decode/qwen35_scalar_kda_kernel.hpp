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

#include "qwen35_decode_common.cuh"
#include "qwen35_scalar_kda_mainloop.hpp"

#include <cute/tensor.hpp>

namespace cula::qwen35::decode::kernel {

using namespace cute;

template <typename scalar_t>
struct Qwen35ScalarKdaDecodeKernel {
  // Decode-first design:
  // - 1 CTA owns 1 (token_idx, hv)
  // - 1 warpgroup (128 threads) per CTA
  // - recurrent state stays fp32 and is traversed as 16x16 tiles over the
  //   internal [V, K] view
  // - the intended optimized path is fp32 FFMA on CUDA cores, not a forced
  //   Tensor Core lowering
  static constexpr int kThreads = 128;
  static constexpr int kWarpGroupThreads = 128;
  static constexpr int kTileV = 16;
  static constexpr int kTileK = 16;
  static constexpr int kTilesPerV = kHeadDimV / kTileV;
  static constexpr int kTilesPerK = kHeadDimQK / kTileK;

  static_assert(kNumQKHeads < kNumVHeads);
  static_assert(kHeadDimQK == 128);
  static_assert(kHeadDimV == 128);
  static_assert(kThreads == kWarpGroupThreads);
  static_assert(kHeadDimV % kTileV == 0);
  static_assert(kHeadDimQK % kTileK == 0);

  struct SharedStorage {
    // Shared staging plan for the fp32 decode path:
    // - q/k/v are staged once per CTA
    // - proj/out intermediates remain in fp32
    // - recurrent state itself remains in fp32 global storage
    alignas(16) scalar_t q_smem[kHeadDimQK];
    alignas(16) scalar_t k_smem[kHeadDimQK];
    alignas(16) scalar_t v_smem[kHeadDimV];
    alignas(16) float proj_smem[kHeadDimV];
    alignas(16) float out_smem[kHeadDimV];
  };

  static dim3 block_shape() {
    return dim3(kThreads, 1, 1);
  }

  static dim3 grid_shape(int token_count) {
    // One block owns one (token_idx, hv) pair in the first implementation.
    return dim3(static_cast<unsigned int>(kNumVHeads), static_cast<unsigned int>(token_count), 1);
  }

  template <typename Mainloop>
  CUTE_DEVICE static void run_device(
      const scalar_t* __restrict__ q_rep,
      const scalar_t* __restrict__ k_rep,
      const scalar_t* __restrict__ v,
      const scalar_t* __restrict__ a_kernel,
      const scalar_t* __restrict__ b_kernel,
      const float* __restrict__ A_log,
      const float* __restrict__ dt_bias,
      float* __restrict__ recurrent_state,
      const int32_t* __restrict__ pool_idx,
      scalar_t* __restrict__ out,
      int token_count,
      SharedStorage& storage) {
    const int hv = static_cast<int>(blockIdx.x);
    const int token_idx = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (token_idx >= token_count || hv >= kNumVHeads) {
      return;
    }

    // Internal tensor-view contract fixed for the first implementation pass:
    //
    // 1. q_rep / k_rep / v / out stay in their external contiguous layouts:
    //    - q_rep : [N, HV, K] with stride (HV*K, K, 1)
    //    - k_rep : [N, HV, K] with stride (HV*K, K, 1)
    //    - v     : [N, HV, V] with stride (HV*V, V, 1)
    //    - out   : [N, HV, V] with stride (HV*V, V, 1)
    //
    // 2. a_kernel / b_kernel are treated as:
    //    - [N, HV] with stride (HV, 1)
    //
    // 3. A_log / dt_bias are treated as:
    //    - [HV] with stride (1)
    //
    // 4. recurrent_state keeps the external physical storage contract:
    //    - [pool, HV, K, V]
    //    but the kernel's main computation will use an internal VK view:
    //    - [pool, HV, V, K]
    //
    // This lets the recurrent update consume one V-row of state against q/k
    // more naturally in the first mainloop design, while preserving the
    // existing external state ABI.
    //
    // The current block owns exactly one (token_idx, hv) pair. That means one
    // warpgroup-sized CTA updates one 128x128 recurrent-state tile for one
    // v-head.
    //
    // TODO(qwen35-scalar-kda-opt):
    // - Likely next optimization path: keep one CTA per (token_idx, hv), but
    //   tile the 128x128 state more aggressively inside the block (for example
    //   along V tiles or KxV subtiles assigned per warp).
    // - More complex alternative: split one (token_idx, hv) tile across
    //   multiple CTAs and coordinate updates. Not a first-pass target.
    // - After the fp32 decode path is stable, evaluate warp specialization:
    //   dedicated producer/load warp(s) vs consumer/compute warp(s), instead
    //   of introducing that complexity before the math path itself is stable.

    auto q_layout = make_layout(
        make_shape(token_count, Int<kNumVHeads>{}, Int<kHeadDimQK>{}),
        make_stride(kNumVHeads * kHeadDimQK, kHeadDimQK, Int<1>{}));
    auto v_layout = make_layout(
        make_shape(token_count, Int<kNumVHeads>{}, Int<kHeadDimV>{}),
        make_stride(kNumVHeads * kHeadDimV, kHeadDimV, Int<1>{}));
    auto head_layout = make_layout(
        make_shape(token_count, Int<kNumVHeads>{}),
        make_stride(kNumVHeads, Int<1>{}));
    auto hv_layout = make_layout(make_shape(Int<kNumVHeads>{}), make_stride(Int<1>{}));
    auto state_layout_kv = make_layout(
        make_shape(_, Int<kNumVHeads>{}, Int<kHeadDimQK>{}, Int<kHeadDimV>{}),
        make_stride(Int<kNumVHeads>{} * kHeadDimQK * kHeadDimV, kHeadDimQK * kHeadDimV, kHeadDimV, Int<1>{}));
    auto state_layout_vk = make_layout(
        make_shape(_, Int<kNumVHeads>{}, Int<kHeadDimV>{}, Int<kHeadDimQK>{}),
        make_stride(Int<kNumVHeads>{} * kHeadDimQK * kHeadDimV, kHeadDimQK * kHeadDimV, Int<1>{}, kHeadDimV));

    auto gQ = make_tensor(make_gmem_ptr(q_rep), q_layout);
    auto gK = make_tensor(make_gmem_ptr(k_rep), q_layout);
    auto gV = make_tensor(make_gmem_ptr(v), v_layout);
    auto gO = make_tensor(make_gmem_ptr(out), v_layout);
    auto gA = make_tensor(make_gmem_ptr(a_kernel), head_layout);
    auto gB = make_tensor(make_gmem_ptr(b_kernel), head_layout);
    auto gAlog = make_tensor(make_gmem_ptr(A_log), hv_layout);
    auto gDt = make_tensor(make_gmem_ptr(dt_bias), hv_layout);
    auto gH_kv = make_tensor(make_gmem_ptr(recurrent_state), state_layout_kv);
    auto gH_vk = make_tensor(make_gmem_ptr(recurrent_state), state_layout_vk);
    (void)gH_kv; // Keep the physical KV view documented and available.

    const int state_row = pool_idx[token_idx];
    if (state_row < 0) {
      return;
    }

    auto q_vec = gQ(token_idx, hv, _);
    auto k_vec = gK(token_idx, hv, _);
    auto v_vec = gV(token_idx, hv, _);
    auto out_vec = gO(token_idx, hv, _);
    auto a_scalar = gA(token_idx, hv);
    auto b_scalar = gB(token_idx, hv);
    auto A_log_scalar = gAlog(hv);
    auto dt_bias_scalar = gDt(hv);
    auto state_vk = gH_vk(state_row, hv, _, _);

    Mainloop::run(
        q_vec,
        k_vec,
        v_vec,
        a_scalar,
        b_scalar,
        A_log_scalar,
        dt_bias_scalar,
        state_vk,
        out_vec,
        storage,
        tid,
        kThreads);
  }
};

template <typename scalar_t, typename Mainloop = Qwen35ScalarKdaDecodeMainloop<scalar_t>>
__global__ void qwen35_scalar_kda_decode_kernel(
    const scalar_t* __restrict__ q_rep,
    const scalar_t* __restrict__ k_rep,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ a_kernel,
    const scalar_t* __restrict__ b_kernel,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ recurrent_state,
    const int32_t* __restrict__ pool_idx,
    scalar_t* __restrict__ out,
    int token_count) {
  __shared__ typename Qwen35ScalarKdaDecodeKernel<scalar_t>::SharedStorage storage;
  Qwen35ScalarKdaDecodeKernel<scalar_t>::template run_device<Mainloop>(
      q_rep,
      k_rep,
      v,
      a_kernel,
      b_kernel,
      A_log,
      dt_bias,
      recurrent_state,
      pool_idx,
      out,
      token_count,
      storage);
}

template <typename scalar_t, typename Mainloop = Qwen35ScalarKdaDecodeMainloop<scalar_t>>
void launch_qwen35_scalar_kda_decode_kernel(
    cudaStream_t stream,
    const scalar_t* q_rep,
    const scalar_t* k_rep,
    const scalar_t* v,
    const scalar_t* a_kernel,
    const scalar_t* b_kernel,
    const float* A_log,
    const float* dt_bias,
    float* recurrent_state,
    const int32_t* pool_idx,
    scalar_t* out,
    int token_count) {
  auto grid = Qwen35ScalarKdaDecodeKernel<scalar_t>::grid_shape(token_count);
  auto block = Qwen35ScalarKdaDecodeKernel<scalar_t>::block_shape();
  qwen35_scalar_kda_decode_kernel<scalar_t, Mainloop><<<grid, block, 0, stream>>>(
      q_rep,
      k_rep,
      v,
      a_kernel,
      b_kernel,
      A_log,
      dt_bias,
      recurrent_state,
      pool_idx,
      out,
      token_count);
}

} // namespace cula::qwen35::decode::kernel
