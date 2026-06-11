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

#include "qwen35_prefill_common.cuh"
#include "qwen35_scalar_kda_prefill_kernel.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

namespace cula::qwen35::prefill {

namespace {

void check_tensor_device(const at::Tensor& tensor, const char* name, const at::Device& device) {
  if (tensor.defined() && tensor.numel() > 0) {
    TORCH_CHECK(tensor.device() == device, name, " must be on device ", device, ".");
  }
}

void check_contiguous(const at::Tensor& tensor, const char* name) {
  if (tensor.defined() && tensor.numel() > 0) {
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  }
}

template <typename scalar_t, int kLocalVHeads>
void dispatch_scalar_prefill_for_heads(
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
  kernel::launch_qwen35_scalar_kda_prefill_kernel<scalar_t, kLocalVHeads>(
      stream,
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

template <typename scalar_t>
void dispatch_scalar_prefill(
    int64_t local_v_heads,
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
  switch (local_v_heads) {
    case 48:
      dispatch_scalar_prefill_for_heads<scalar_t, 48>(stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state, batch_size, seq_len, sequence_count, is_varlen, has_initial_state);
      break;
    case 24:
      dispatch_scalar_prefill_for_heads<scalar_t, 24>(stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state, batch_size, seq_len, sequence_count, is_varlen, has_initial_state);
      break;
    case 12:
      dispatch_scalar_prefill_for_heads<scalar_t, 12>(stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state, batch_size, seq_len, sequence_count, is_varlen, has_initial_state);
      break;
    case 6:
      dispatch_scalar_prefill_for_heads<scalar_t, 6>(stream, q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens, out, final_state, batch_size, seq_len, sequence_count, is_varlen, has_initial_state);
      break;
  }
}

} // namespace

void run_qwen35_scalar_kda_prefill(ScalarKdaPrefillParams& params) {
  const at::Tensor& q = params.q;
  const at::Tensor& k = params.k;
  const at::Tensor& v = params.v;
  const at::Tensor& a = params.a;
  const at::Tensor& b = params.b;
  const at::Tensor& A_log = params.A_log;
  const at::Tensor& dt_bias = params.dt_bias;
  const at::Tensor& initial_state = params.initial_state;
  const at::Tensor& cu_seqlens = params.cu_seqlens;
  const at::Tensor& out = params.out;
  const at::Tensor& final_state = params.final_state;

  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor.");
  const at::Device device = q.device();

  check_tensor_device(k, "k", device);
  check_tensor_device(v, "v", device);
  check_tensor_device(a, "a", device);
  check_tensor_device(b, "b", device);
  check_tensor_device(A_log, "A_log", device);
  check_tensor_device(dt_bias, "dt_bias", device);
  check_tensor_device(initial_state, "initial_state", device);
  check_tensor_device(cu_seqlens, "cu_seqlens", device);
  check_tensor_device(out, "out", device);
  check_tensor_device(final_state, "final_state", device);

  check_contiguous(q, "q");
  check_contiguous(k, "k");
  check_contiguous(v, "v");
  check_contiguous(a, "a");
  check_contiguous(b, "b");
  check_contiguous(A_log, "A_log");
  check_contiguous(dt_bias, "dt_bias");
  check_contiguous(initial_state, "initial_state");
  check_contiguous(cu_seqlens, "cu_seqlens");
  check_contiguous(out, "out");
  check_contiguous(final_state, "final_state");

  TORCH_CHECK(
      q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() &&
          q.scalar_type() == a.scalar_type() && q.scalar_type() == b.scalar_type() &&
          q.scalar_type() == out.scalar_type(),
      "q/k/v/a/b/out must share the same dtype.");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "q must be float16 or bfloat16.");
  TORCH_CHECK(A_log.scalar_type() == at::kFloat, "A_log must be float32.");
  TORCH_CHECK(dt_bias.scalar_type() == at::kFloat, "dt_bias must be float32.");
  TORCH_CHECK(final_state.scalar_type() == at::kFloat, "final_state must be float32.");
  TORCH_CHECK(
      !initial_state.defined() || initial_state.numel() == 0 || initial_state.scalar_type() == at::kFloat,
      "initial_state must be float32 when provided.");
  TORCH_CHECK(
      !cu_seqlens.defined() || cu_seqlens.numel() == 0 || cu_seqlens.scalar_type() == at::kInt,
      "cu_seqlens must be int32 when provided.");

  TORCH_CHECK(q.dim() == 4, "q must be [B, T, 48, 128].");
  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t local_v_heads = q.size(2);
  TORCH_CHECK(decode::is_supported_local_v_heads(static_cast<int>(local_v_heads)), "local V heads must be one of {48, 24, 12, 6}, got ", local_v_heads, ".");
  TORCH_CHECK(
      q.sizes() == at::IntArrayRef({B, T, local_v_heads, kHeadDimQK}),
      "q must have shape [B, T, local_v_heads, 128].");
  TORCH_CHECK(k.sizes() == q.sizes(), "k must match q shape.");
  TORCH_CHECK(v.sizes() == q.sizes(), "v must match q shape.");
  TORCH_CHECK(a.dim() == 3 && a.sizes() == at::IntArrayRef({B, T, local_v_heads}), "a must be [B, T, local_v_heads].");
  TORCH_CHECK(b.sizes() == a.sizes(), "b must match a shape.");
  TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == local_v_heads, "A_log must be [local_v_heads].");
  TORCH_CHECK(dt_bias.dim() == 1 && dt_bias.size(0) == local_v_heads, "dt_bias must be [local_v_heads].");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape.");

  const bool is_varlen = cu_seqlens.defined() && cu_seqlens.numel() > 0;
  const int64_t sequence_count = is_varlen ? cu_seqlens.numel() - 1 : B;
  TORCH_CHECK(sequence_count > 0, "sequence_count must be positive.");
  if (is_varlen) {
    TORCH_CHECK(B == 1, "cu_seqlens mode expects flattened q/k/v with batch size 1.");
  }

  TORCH_CHECK(
      final_state.dim() == 4 &&
          final_state.sizes() == at::IntArrayRef({sequence_count, local_v_heads, kHeadDimQK, kHeadDimV}),
      "final_state must be [N, local_v_heads, 128, 128].");
  const bool has_initial_state = initial_state.defined() && initial_state.numel() > 0;
  if (has_initial_state) {
    TORCH_CHECK(initial_state.sizes() == final_state.sizes(), "initial_state must match final_state shape.");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device);
  cudaStream_t stream = at::cuda::getDefaultCUDAStream(device.index());

  if (q.scalar_type() == at::kHalf) {
    dispatch_scalar_prefill<c10::Half>(
        local_v_heads,
        stream,
        q.data_ptr<c10::Half>(),
        k.data_ptr<c10::Half>(),
        v.data_ptr<c10::Half>(),
        a.data_ptr<c10::Half>(),
        b.data_ptr<c10::Half>(),
        A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(),
        has_initial_state ? initial_state.data_ptr<float>() : nullptr,
        is_varlen ? cu_seqlens.data_ptr<int32_t>() : nullptr,
        out.data_ptr<c10::Half>(),
        final_state.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(T),
        static_cast<int>(sequence_count),
        is_varlen,
        has_initial_state);
  } else {
    dispatch_scalar_prefill<c10::BFloat16>(
        local_v_heads,
        stream,
        q.data_ptr<c10::BFloat16>(),
        k.data_ptr<c10::BFloat16>(),
        v.data_ptr<c10::BFloat16>(),
        a.data_ptr<c10::BFloat16>(),
        b.data_ptr<c10::BFloat16>(),
        A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(),
        has_initial_state ? initial_state.data_ptr<float>() : nullptr,
        is_varlen ? cu_seqlens.data_ptr<int32_t>() : nullptr,
        out.data_ptr<c10::BFloat16>(),
        final_state.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(T),
        static_cast<int>(sequence_count),
        is_varlen,
        has_initial_state);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace cula::qwen35::prefill
