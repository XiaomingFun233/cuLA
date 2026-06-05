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

"""CuTe DSL placeholder for Qwen3.5 depthwise causal conv1d prefill."""

from __future__ import annotations

import torch


def qwen35_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    """Depthwise causal conv1d over a full sequence.

    Expected shapes:
    - x: [B, C, S]
    - weight: [C, 1, 4] or [C, 4]
    """

    del x, weight, activation
    raise NotImplementedError("Qwen3.5 conv1d prefill kernel is not implemented yet.")
