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

#include "qwen35/decode/qwen35_decode_common.cuh"

#include <ATen/core/TensorBody.h>

namespace cula::qwen35::prefill {

using decode::kHeadDimQK;
using decode::kHeadDimV;
using decode::kKDim;
using decode::kMixedQKVDim;
using decode::kNumQKHeads;
using decode::kNumVHeads;
using decode::kQDim;
using decode::kVDim;

struct LayoutPrefillParams {
  at::Tensor mixed_qkv_conv; // [N, 10240]
  at::Tensor a;              // [N, 48]
  at::Tensor b;              // [N, 48]
  at::Tensor q_rep;          // [N, 48, 128]
  at::Tensor k_rep;          // [N, 48, 128]
  at::Tensor v;              // [N, 48, 128]
  at::Tensor a_kernel;       // [N, 48]
  at::Tensor b_kernel;       // [N, 48]
};

struct ScalarKdaPrefillParams {
  at::Tensor q;                // [B, T, 48, 128]
  at::Tensor k;                // [B, T, 48, 128]
  at::Tensor v;                // [B, T, 48, 128]
  at::Tensor a;                // [B, T, 48]
  at::Tensor b;                // [B, T, 48]
  at::Tensor A_log;            // [48], float32
  at::Tensor dt_bias;          // [48], float32
  at::Tensor initial_state;    // [N, 48, 128, 128], float32, may be empty
  at::Tensor cu_seqlens;       // [N + 1], int32, may be empty
  at::Tensor out;              // [B, T, 48, 128]
  at::Tensor final_state;      // [N, 48, 128, 128], float32
};

void run_qwen35_scalar_kda_prefill(ScalarKdaPrefillParams& params);
void run_qwen35_layout_prefill(LayoutPrefillParams& params);

} // namespace cula::qwen35::prefill

namespace cula::qwen35::prefill::sm90 {

void qwen35_chunk_qk_prefill_sm90(at::Tensor q, at::Tensor k, at::Tensor out);

} // namespace cula::qwen35::prefill::sm90
