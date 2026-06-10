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

#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <cstdint>

namespace cula::qwen35::decode {

inline constexpr int kNumQKHeads = 16;
inline constexpr int kNumVHeads = 48;
inline constexpr int kHeadDimQK = 128;
inline constexpr int kHeadDimV = 128;
inline constexpr int kConvKernelSize = 4;
inline constexpr int kQDim = kNumQKHeads * kHeadDimQK;
inline constexpr int kKDim = kNumQKHeads * kHeadDimQK;
inline constexpr int kVDim = kNumVHeads * kHeadDimV;
inline constexpr int kMixedQKVDim = kQDim + kKDim + kVDim;

struct ConvDecodeParams {
  at::Tensor mixed_qkv;   // [B, 1, 10240]
  at::Tensor conv_state;  // [B, 10240, 4]
  at::Tensor conv_weight; // [10240, 4]
  at::Tensor out;         // [B, 1, 10240]
};

struct LayoutDecodeParams {
  at::Tensor mixed_qkv_conv; // [N, 10240]
  at::Tensor a;              // [N, 48]
  at::Tensor b;              // [N, 48]
  at::Tensor q_rep;          // [N, 48, 128]
  at::Tensor k_rep;          // [N, 48, 128]
  at::Tensor v;              // [N, 48, 128]
  at::Tensor a_kernel;       // [N, 48]
  at::Tensor b_kernel;       // [N, 48]
};

struct ScalarKdaDecodeParams {
  // Dtype contract for the first implementation:
  // - activations / outputs: half or bf16
  //     q_rep, k_rep, v, a_kernel, b_kernel, out
  // - recurrent parameters / state: float32
  //     A_log, dt_bias, recurrent_state
  at::Tensor q_rep;            // [N, 48, 128]
  at::Tensor k_rep;            // [N, 48, 128]
  at::Tensor v;                // [N, 48, 128]
  at::Tensor a_kernel;         // [N, 48]
  at::Tensor b_kernel;         // [N, 48]
  at::Tensor A_log;            // [48], float32
  at::Tensor dt_bias;          // [48], float32
  at::Tensor recurrent_state;  // [pool, 48, 128, 128], float32
  at::Tensor pool_idx;         // [N], int32
  at::Tensor out;              // [N, 48, 128]
};

struct LayoutScalarKdaDecodeParams {
  at::Tensor mixed_qkv_conv;    // [N, 10240]
  at::Tensor a;                 // [N, 48]
  at::Tensor b;                 // [N, 48]
  at::Tensor A_log;             // [48], float32
  at::Tensor dt_bias;           // [48], float32
  at::Tensor recurrent_state;   // [pool, 48, 128, 128], float32
  at::Tensor pool_idx;          // [N], int32
  at::Tensor out;               // [N, 48, 128]
};

void run_qwen35_conv1d_decode(ConvDecodeParams& params);
void run_qwen35_layout_decode(LayoutDecodeParams& params);
void run_qwen35_scalar_kda_decode(ScalarKdaDecodeParams& params);
void run_qwen35_layout_scalar_kda_decode(LayoutScalarKdaDecodeParams& params);

} // namespace cula::qwen35::decode
