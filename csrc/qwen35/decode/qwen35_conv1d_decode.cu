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

#include "qwen35_decode_common.cuh"

#include <c10/util/Exception.h>

namespace cula::qwen35::decode {

void run_qwen35_conv1d_decode(ConvDecodeParams& params) {
  (void)params;
  TORCH_CHECK(
      false,
      "run_qwen35_conv1d_decode is not implemented yet. "
      "Planned kernel: single-token depthwise causal conv1d + silu for 10240 channels.");
}

} // namespace cula::qwen35::decode
