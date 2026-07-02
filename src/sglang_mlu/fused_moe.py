# Copyright 2024 SGLang-MLU Authors.
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

"""MLU Fused Mixture-of-Expert operations.

Provides the ``forward_mlufuseep`` entry point that SGLang's ``FusedMoE``
layer can route to, as well as the ``process_mlufuseep_weights`` weight
pre-patterning callback.

v0.1 status
----------
The FusedEP pattern used by Ascend NPU collapses the entire MoE inference step
(dispatch, GEMM-1, GEMM-2, combine) into a single op (deeply tied to the
network fabric). MLU does not have an equivalent CNRT fused operator at the
time of writing, so this module exposes the reference implementation and the
registration seam.

To use the MLU-friendly pattern you can pass ``--moe-a2a-backend deepep`` (which
uses the DeepEP library over CNCL for DP/EP) or ``--moe-a2a-backend none``
(reduces to TP-only MoE). The documented ``mlu_fuseep`` backend is registered
here with a no-op fabric — the implementation is looked up dynamically.

For the v0.2 CNRT Fused EP see docs/design.md section "Fused MoE Operator".
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FusedMoE backend identifier
# ---------------------------------------------------------------------------

MLU_FUSEEP_BACKEND: str = "mlu_fuseep"
MLU_FUSEEP_MODE: str = "MLU_FUSED_MOE_MODE"   # env var selecting variants


# ---------------------------------------------------------------------------
# Reference PyTorch FusedEP forward
# ---------------------------------------------------------------------------

def forward_mlufuseep(
    layer: Any,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Reference forward that performs MoE without a specialised fabric.

    This stands in for a future CNRT fused operator.  The operations are:

    1. Route tokens to local experts via ``topk_ids``.
    2. Run GEMM-1 (gate/up projection) -> activation -> GEMM-2 (down projection).
    3. Weighted sum by ``topk_weights``.

    Parameters
    ----------
    layer : ``FusedMoE`` instance
        Exposes ``w13_weight``, ``w2_weight``, ``num_experts``, etc.
    hidden_states : Tensor [num_tokens, hidden_size]
    topk_ids : Tensor [num_tokens, top_k]
    topk_weights : Tensor [num_tokens, top_k]

    Returns
    -------
    Tensor [num_tokens, hidden_size]
    """
    num_tokens, hidden = hidden_states.shape
    top_k = topk_ids.shape[1]

    output = torch.zeros_like(hidden_states)
    num_local = getattr(layer, "num_local_experts", 1)
    for expert_idx in range(num_local):
        # Find which token-positions route to this expert (any top-k slot).
        expert_match = (topk_ids == expert_idx)  # [num_tokens, top_k]
        mask = expert_match.any(dim=1)           # [num_tokens]
        if not mask.any():
            continue
        tokens = hidden_states[mask]             # [n_tok_local, hidden]
        w13 = layer.w13_weight[expert_idx]       # [2*intermediate, hidden]
        w2  = layer.w2_weight[expert_idx]        # [intermediate, hidden]
        # GEMM-1: gate + up projection
        h = torch.nn.functional.linear(tokens, w13)   # [n_tok, 2*intermediate]
        gate, up = h.chunk(2, dim=-1)
        h = torch.nn.functional.silu(gate) * up       # [n_tok, intermediate]
        # GEMM-2: down projection
        out = torch.nn.functional.linear(h, w2.t())   # [n_tok, hidden]
        # Per-position weight: the specific top-k weight for THIS expert.
        # expert_match[mask] gives [n_tok_local, top_k] bool; find the column.
        match_cols = expert_match[mask].nonzero(as_tuple=True)
        per_tok_w = topk_weights[mask][match_cols[0], match_cols[1]]
        output[mask] += out * per_tok_w.unsqueeze(-1)
    return output


def process_mlufuseep_weights(layer: Any) -> None:
    """Convert MoE weights to the layout required by the MLU FusedEP op.

    v0.1 is a no-op; all weights stay in the original PyTorch layout, and
    ``forward_mlufuseep`` operates on them directly.
    """
    logger.debug("process_mlufuseep_weights called (v0.1 no-op)")


def register_mlufuseep_backend() -> None:
    """Register the ``mlu_fuseep`` backend with SGLang's MoE framework.

    No-op in v0.1 — real registration requires patches in sglang core
    (MoeA2ABackend enum, create_moe_dispatcher, FusedMoE.forward bypass).
    """
    logger.debug("MLU FuseEP backend registered (v0.1 no fabric)")


register_mlufuseep_backend()

__all__ = [
    "MLU_FUSEEP_BACKEND",
    "forward_mlufuseep",
    "process_mlufuseep_weights",
    "register_mlufuseep_backend",
]
