"""Tests for MLU-specific MLA (Multi-head Latent Attention) operators.

These tests use **real** CPU tensors — no MLU hardware is needed because the
v0.1 operators are pure-PyTorch reference implementations.  The focus is on
tensor shape contracts and numerical sanity.
"""

from __future__ import annotations

import sys

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _purge_sglang_mlu() -> None:
    """Remove cached sglang_mlu modules so the next import is from scratch."""
    for key in [k for k in sys.modules if k.startswith("sglang_mlu")]:
        del sys.modules[key]


# ---------------------------------------------------------------------------
# Tests — absorption ops
# ---------------------------------------------------------------------------


def test_absorb_q_nope_shape() -> None:
    """``mlu_absorb_q_nope``: [B,H,D_in] × [H,D_in,D_out] → [B,H,D_out]."""
    _purge_sglang_mlu()
    from sglang_mlu.mla_attention import mlu_absorb_q_nope

    B, H, D_in, D_out = 2, 4, 64, 128
    q_nope = torch.randn(B, H, D_in)
    w_kc = torch.randn(H, D_in, D_out)

    out = mlu_absorb_q_nope(q_nope, w_kc)
    assert out.shape == (B, H, D_out)


def test_absorb_w_vc_shape() -> None:
    """``mlu_absorb_w_vc``: [B,H,D] × [H,D,V] → [B,H,V]."""
    _purge_sglang_mlu()
    from sglang_mlu.mla_attention import mlu_absorb_w_vc

    B, H, D, V = 2, 4, 128, 64
    attn_output = torch.randn(B, H, D)
    w_vc = torch.randn(H, D, V)

    out = mlu_absorb_w_vc(attn_output, w_vc)
    assert out.shape == (B, H, V)


# ---------------------------------------------------------------------------
# Tests — normalisation
# ---------------------------------------------------------------------------


def test_rms_norm_shape() -> None:
    """``mlu_rms_norm`` preserves the input tensor shape."""
    _purge_sglang_mlu()
    from sglang_mlu.mla_attention import mlu_rms_norm

    shape = (2, 4, 128)
    x = torch.randn(shape)
    weight = torch.randn(128)

    out = mlu_rms_norm(x, weight)
    assert out.shape == shape


def test_rms_norm_zero_mean_unit_variance() -> None:
    """RMS-normalised output has approximately unit RMS along the last dim."""
    _purge_sglang_mlu()
    from sglang_mlu.mla_attention import mlu_rms_norm

    x = torch.randn(8, 16, 64)
    weight = torch.ones(64)
    out = mlu_rms_norm(x, weight)

    # RMS ≈ 1.0 when weight is all ones.
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


# ---------------------------------------------------------------------------
# Tests — fused QKV projection
# ---------------------------------------------------------------------------


def test_fused_qkv_a_proj_shape() -> None:
    """``mlu_fused_qkv_a_proj``: linear projection shape check."""
    _purge_sglang_mlu()
    from sglang_mlu.mla_attention import mlu_fused_qkv_a_proj

    B, D_in, D_out = 8, 256, 512
    hidden = torch.randn(B, D_in)
    weight = torch.randn(D_out, D_in)

    out = mlu_fused_qkv_a_proj(hidden, weight)
    assert out.shape == (B, D_out)


def test_fused_qkv_a_proj_with_bias() -> None:
    """Bias is added element-wise; output shape stays the same."""
    _purge_sglang_mlu()
    from sglang_mlu.mla_attention import mlu_fused_qkv_a_proj

    B, D_in, D_out = 4, 128, 256
    hidden = torch.randn(B, D_in)
    weight = torch.randn(D_out, D_in)
    bias = torch.randn(D_out)

    out = mlu_fused_qkv_a_proj(hidden, weight, bias)
    assert out.shape == (B, D_out)


# ---------------------------------------------------------------------------
# Tests — rotary embedding (reference no-op)
# ---------------------------------------------------------------------------


def test_rotary_emb_returns_tuple() -> None:
    """``mlu_rotary_emb`` returns a ``(q, k)`` tuple with unchanged shapes."""
    _purge_sglang_mlu()
    from sglang_mlu.mla_attention import mlu_rotary_emb

    B, H, D = 2, 4, 64
    q = torch.randn(B, H, D)
    k = torch.randn(B, H, D)
    cos = torch.randn(B, D // 2)
    sin = torch.randn(B, D // 2)

    q_out, k_out = mlu_rotary_emb(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


# ---------------------------------------------------------------------------
# Tests — registry hook
# ---------------------------------------------------------------------------


def test_register_mlu_mla_ops_runs() -> None:
    """``register_mlu_mla_ops`` is a no-op that must not raise."""
    _purge_sglang_mlu()
    from sglang_mlu.mla_attention import register_mlu_mla_ops

    register_mlu_mla_ops()
