"""Tests that every sglang_mlu module imports cleanly without sglang / torch_mlu.

The out-of-tree plugin relies on ``try/except ImportError`` guards in every
module so it can be imported on development machines that lack the MLU SDK
and/or the ``sglang`` package.  This file verifies that contract: importing
any submodule (or the top-level package) must **not** raise.
"""

from __future__ import annotations

import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _purge_sglang_mlu() -> None:
    """Remove cached sglang_mlu modules so the next import is from scratch."""
    for key in [k for k in sys.modules if k.startswith("sglang_mlu")]:
        del sys.modules[key]


# ---------------------------------------------------------------------------
# Top-level package
# ---------------------------------------------------------------------------


def test_activate_returns_none_without_mlu(mlu_inactive: None) -> None:
    """``activate()`` returns ``None`` when MLU hardware is unavailable.

    With ``torch_mlu`` blocked, ``is_mlu_available()`` returns ``False`` and
    ``activate()`` short-circuits to ``None``.
    """
    _purge_sglang_mlu()
    from sglang_mlu import activate

    assert activate() is None


def test_is_mlu_returns_false_without_mlu(mlu_inactive: None) -> None:
    """The ``is_mlu()`` helper exposed at the top level returns ``False``."""
    _purge_sglang_mlu()
    from sglang_mlu import is_mlu

    assert is_mlu() is False


# ---------------------------------------------------------------------------
# Per-module import-safety tests
# ---------------------------------------------------------------------------


def test_utils_import_safe() -> None:
    """``sglang_mlu.utils`` imports without sglang or torch_mlu."""
    _purge_sglang_mlu()
    from sglang_mlu import utils  # noqa: F401

    # Public surface area that must remain importable
    for name in (
        "is_mlu_available",
        "get_mlu_device_count",
        "init_mlu_backend",
        "apply_mlu_server_args_defaults",
        "is_mlu",
        "get_default_mlu_env",
    ):
        assert hasattr(utils, name), f"utils missing attribute: {name}"


def test_device_import_safe() -> None:
    """``sglang_mlu.device`` imports without sglang or torch_mlu."""
    _purge_sglang_mlu()
    from sglang_mlu.device import MLUDeviceMixin  # noqa: F401

    assert MLUDeviceMixin is not None


def test_attention_import_safe() -> None:
    """``sglang_mlu.attention`` imports without sglang or torch_mlu."""
    _purge_sglang_mlu()
    from sglang_mlu.attention import (  # noqa: F401
        MLUAttentionBackend,
        MLUAttentionInferenceSession,
    )

    assert MLUAttentionBackend is not None
    assert MLUAttentionInferenceSession is not None


def test_mla_import_safe() -> None:
    """``sglang_mlu.mla_attention`` imports without sglang or torch_mlu."""
    _purge_sglang_mlu()
    from sglang_mlu import mla_attention  # noqa: F401

    for name in (
        "mlu_fused_qkv_a_proj",
        "mlu_rms_norm",
        "mlu_rotary_emb",
        "mlu_absorb_q_nope",
        "mlu_absorb_w_vc",
        "mlu_sdpa_absorbed",
        "mlu_kv_rms_norm_rope_cache",
        "register_mlu_mla_ops",
    ):
        assert hasattr(mla_attention, name), f"mla_attention missing: {name}"


def test_fused_moe_import_safe() -> None:
    """``sglang_mlu.fused_moe`` imports without sglang or torch_mlu."""
    _purge_sglang_mlu()
    from sglang_mlu.fused_moe import (  # noqa: F401
        MLU_FUSEEP_BACKEND,
        forward_mlufuseep,
        process_mlufuseep_weights,
        register_mlufuseep_backend,
    )

    assert MLU_FUSEEP_BACKEND is not None
    assert callable(forward_mlufuseep)
    assert callable(process_mlufuseep_weights)
    assert callable(register_mlufuseep_backend)


def test_memory_import_safe() -> None:
    """``sglang_mlu.memory`` imports without sglang or torch_mlu."""
    _purge_sglang_mlu()
    from sglang_mlu.memory import (  # noqa: F401
        MLUMHATokenToKVPool,
        MLUMLATokenToKVPool,
        MLUDSATokenToKVPool,
        MLUPagedAllocator,
    )

    assert MLUMHATokenToKVPool is not None
    assert MLUMLATokenToKVPool is not None
    assert MLUDSATokenToKVPool is not None
    assert MLUPagedAllocator is not None


def test_cnnl_import_safe() -> None:
    """``sglang_mlu.kernels`` imports without sglang or torch_mlu."""
    _purge_sglang_mlu()
    from sglang_mlu.kernels import (  # noqa: F401
        CNNL_AVAILABLE,
        cnnl_flash_attention_score,
        cnnl_rms_norm,
        cnnl_matmul,
    )

    # Without ``torch_mlu`` the flag must be False.
    assert CNNL_AVAILABLE is False


def test_cnnl_stubs_raise_runtime_error(no_torch_mlu: None) -> None:
    """CNNL stubs raise ``RuntimeError`` (not ``ImportError``) when called."""
    _purge_sglang_mlu()
    from sglang_mlu.kernels import cnnl_wrappers

    assert cnnl_wrappers.is_cnnl_available() is False
    assert cnnl_wrappers.get_cnnl_version() == "uninstalled"

    with pytest.raises(RuntimeError, match="MLU SDK is not installed"):
        cnnl_wrappers.cnnl_flash_attention_score(None, None, None)


def test_platform_import_safe() -> None:
    """``sglang_mlu.platform`` imports without sglang or torch_mlu."""
    _purge_sglang_mlu()
    from sglang_mlu.platform import MLUPlatform  # noqa: F401

    assert MLUPlatform is not None


def test_graph_runner_import_safe() -> None:
    """``sglang_mlu.graph_runner`` imports without sglang or torch_mlu."""
    _purge_sglang_mlu()
    from sglang_mlu import graph_runner  # noqa: F401

    # When sglang is missing, ``MLUGraphRunner`` is ``None``.
    assert hasattr(graph_runner, "MLUGraphRunner")
