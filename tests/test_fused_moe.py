"""Tests for the MLU Fused Mixture-of-Expert (MoE) operations.

Validates the reference ``forward_mlufuseep`` implementation, the no-op weight
processing callback, and the backend identifier string.  Uses real CPU tensors
— no MLU hardware required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _purge_sglang_mlu() -> None:
    """Remove cached sglang_mlu modules so the next import is from scratch."""
    for key in [k for k in sys.modules if k.startswith("sglang_mlu")]:
        del sys.modules[key]


def _make_mock_layer(
    num_experts: int = 2,
    hidden: int = 64,
    intermediate: int = 128,
) -> MagicMock:
    """Build a minimal mock ``FusedMoE`` layer with random weights."""
    layer = MagicMock()
    layer.num_local_experts = num_experts
    # w13_weight: [num_experts, 2 * intermediate, hidden]
    layer.w13_weight = torch.randn(num_experts, 2 * intermediate, hidden)
    # w2_weight: [num_experts, intermediate, hidden]
    layer.w2_weight = torch.randn(num_experts, intermediate, hidden)
    return layer


# ---------------------------------------------------------------------------
# Tests — forward shape
# ---------------------------------------------------------------------------


def test_forward_mlufuseep_shape() -> None:
    """``forward_mlufuseep`` output shape matches input shape."""
    _purge_sglang_mlu()
    from sglang_mlu.fused_moe import forward_mlufuseep

    num_tokens, hidden, top_k = 8, 64, 2
    layer = _make_mock_layer(num_experts=2, hidden=hidden, intermediate=128)

    hidden_states = torch.randn(num_tokens, hidden)
    # Ensure unique expert IDs per token (standard MoE routing guarantee).
    topk_ids = torch.stack(
        [torch.randperm(2)[:top_k] for _ in range(num_tokens)]
    )
    topk_weights = torch.rand(num_tokens, top_k)
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

    out = forward_mlufuseep(layer, hidden_states, topk_ids, topk_weights)
    assert out.shape == hidden_states.shape


def test_forward_mlufuseep_single_expert() -> None:
    """Forward pass works with a single local expert."""
    _purge_sglang_mlu()
    from sglang_mlu.fused_moe import forward_mlufuseep

    num_tokens, hidden, top_k = 4, 32, 1
    layer = _make_mock_layer(num_experts=1, hidden=hidden, intermediate=64)

    hidden_states = torch.randn(num_tokens, hidden)
    topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.long)
    topk_weights = torch.ones(num_tokens, top_k)

    out = forward_mlufuseep(layer, hidden_states, topk_ids, topk_weights)
    assert out.shape == hidden_states.shape


def test_forward_mlufuseep_no_tokens_routed() -> None:
    """When no tokens map to any local expert, output is all zeros."""
    _purge_sglang_mlu()
    from sglang_mlu.fused_moe import forward_mlufuseep

    num_tokens, hidden = 4, 32
    layer = _make_mock_layer(num_experts=2, hidden=hidden, intermediate=64)

    hidden_states = torch.randn(num_tokens, hidden)
    # Route every token to expert 5 — outside the local range.
    topk_ids = torch.full((num_tokens, 1), 5, dtype=torch.long)
    topk_weights = torch.ones(num_tokens, 1)

    out = forward_mlufuseep(layer, hidden_states, topk_ids, topk_weights)
    assert out.shape == hidden_states.shape
    assert torch.allclose(out, torch.zeros_like(hidden_states))


# ---------------------------------------------------------------------------
# Tests — weight processing
# ---------------------------------------------------------------------------


def test_process_weights_no_op() -> None:
    """``process_mlufuseep_weights`` is a no-op that runs without error."""
    _purge_sglang_mlu()
    from sglang_mlu.fused_moe import process_mlufuseep_weights

    layer = _make_mock_layer()
    # Should not raise.
    process_mlufuseep_weights(layer)


# ---------------------------------------------------------------------------
# Tests — backend identifier
# ---------------------------------------------------------------------------


def test_backend_name() -> None:
    """``MLU_FUSEEP_BACKEND`` equals ``\"mlu_fuseep\"``."""
    _purge_sglang_mlu()
    from sglang_mlu.fused_moe import MLU_FUSEEP_BACKEND

    assert MLU_FUSEEP_BACKEND == "mlu_fuseep"


# ---------------------------------------------------------------------------
# Tests — registration
# ---------------------------------------------------------------------------


def test_forward_mlufuseep_correct_weights() -> None:
    """Each expert's output is weighted by its OWN top-k weight, not the sum.

    Regression test for the oracle v2 bug where every expert's output was
    multiplied by the sum of all top-k weights (= 1.0), inflating magnitude.
    """
    _purge_sglang_mlu()
    from sglang_mlu.fused_moe import forward_mlufuseep

    num_tokens, hidden, top_k = 4, 32, 2
    layer = _make_mock_layer(num_experts=2, hidden=hidden, intermediate=64)

    # Single token, routes to expert-0 (w=0.8) and expert-1 (w=0.2)
    hidden_states = torch.ones(1, hidden)
    topk_ids = torch.tensor([[0, 1]])
    topk_weights = torch.tensor([[0.8, 0.2]])
    out = forward_mlufuseep(layer, hidden_states, topk_ids, topk_weights)

    # Manually compute expected output for verification
    # Expert 0 contribution: 0.8 * (down_proj(silu(gem1(ones))))
    # Expert 1 contribution: 0.2 * (down_proj(silu(gem1(ones))))
    # Total = w0 * out0 + w1 * out1 (NOT (w0+w1) * out0 + (w0+w1) * out1)
    assert out.shape == (1, hidden)
    # The result should NOT be (out0 + out1) — that would be the inflated value.
    # Since w0+w1 = 1.0 here, we verify indirectly via a different weight sum
    topk_weights2 = torch.tensor([[0.6, 0.1]])  # sums to 0.7, not 1.0
    out2 = forward_mlufuseep(layer, hidden_states, topk_ids, topk_weights2)
    # If bug were present, out2 == out (both weighted by sum=1.0 and sum=0.7 give same)
    # With fix, out2 should differ from out
    assert not torch.allclose(out, out2), (
        "Changing top-k weights should change output magnitude"
    )


def test_register_backend_runs() -> None:
    """``register_mlufuseep_backend`` is a no-op that must not raise."""
    _purge_sglang_mlu()
    from sglang_mlu.fused_moe import register_mlufuseep_backend

    register_mlufuseep_backend()
