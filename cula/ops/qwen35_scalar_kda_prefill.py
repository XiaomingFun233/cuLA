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

"""CuTe DSL placeholder for Qwen3.5 scalar-gated KDA prefill."""

from __future__ import annotations

import torch


def qwen35_scalar_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked scalar-gated delta-rule prefill for Qwen3.5."""

    del q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens
    raise NotImplementedError("Qwen3.5 scalar-gated KDA prefill kernel is not implemented yet.")
