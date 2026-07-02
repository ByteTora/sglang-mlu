"""Tests for MLUAttentionBackend.

Validates the MLU attention backend flags, provider string, and registration
behaviour using a mocked ``runner`` object.  No real SGLang or MLU hardware
is required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _purge_sglang_mlu() -> None:
    """Remove cached sglang_mlu modules so the next import is from scratch."""
    for key in [k for k in sys.modules if k.startswith("sglang_mlu")]:
        del sys.modules[key]


# ---------------------------------------------------------------------------
# Tests — instantiation
# ---------------------------------------------------------------------------


def test_init_with_mock_runner(mock_runner: MagicMock) -> None:
    """``MLUAttentionBackend`` instantiates without error given a mock runner."""
    _purge_sglang_mlu()
    from sglang_mlu.attention import MLUAttentionBackend

    backend = MLUAttentionBackend(mock_runner)
    assert backend is not None


# ---------------------------------------------------------------------------
# Tests — capability flags
# ---------------------------------------------------------------------------


def test_flags() -> None:
    """MLU backend disables CUDA graph and CPU-side sequence lengths."""
    _purge_sglang_mlu()
    from sglang_mlu.attention import MLUAttentionBackend

    assert MLUAttentionBackend.use_cuda_graph is False
    assert MLUAttentionBackend.needs_cpu_seq_lens is False


# ---------------------------------------------------------------------------
# Tests — provider string
# ---------------------------------------------------------------------------


def test_provider() -> None:
    """The ``_provider`` class attribute equals ``\"mlu\"``."""
    _purge_sglang_mlu()
    from sglang_mlu.attention import MLUAttentionBackend

    assert MLUAttentionBackend._provider == "mlu"


# ---------------------------------------------------------------------------
# Tests — registration decorator
# ---------------------------------------------------------------------------


def test_register_decorator_runs() -> None:
    """``_register()`` executes without error when sglang is absent.

    In the stub path (no sglang installed) the function short-circuits and
    logs a debug message.  Calling it again must be safe.
    """
    _purge_sglang_mlu()
    from sglang_mlu.attention import _register

    # First call at import time already happened; calling again must not raise.
    _register()


def test_register_with_sglang_mock() -> None:
    """``_register()`` invokes the registry when sglang is available."""
    import types

    registry_state: dict[str, object] = {}

    def fake_register(name: str):
        """Return a no-op decorator that records the name."""
        registry_state["registered_name"] = name

        def decorator(func):
            registry_state["factory"] = func
            return func

        return decorator

    # Build fake module objects for every sglang submodule that attention.py
    # imports.  We use real ``ModuleType`` instances so that the ``try/except``
    # import-guard in attention.py sets ``_HAS_SGLANG = True``.
    backend_base = types.ModuleType("sglang.srt.layers.attention.base_attn_backend")
    backend_base.AttentionBackend = MagicMock()

    native_backend = types.ModuleType("sglang.srt.layers.attention.torch_native_backend")
    native_backend.TorchNativeAttnBackend = MagicMock()

    attention_registry = types.ModuleType("sglang.srt.layers.attention.attention_registry")
    attention_registry.register_attention_backend = fake_register

    modules_patch = {
        "sglang.srt.layers.attention.base_attn_backend": backend_base,
        "sglang.srt.layers.attention.torch_native_backend": native_backend,
        "sglang.srt.layers.attention.attention_registry": attention_registry,
    }

    with patch.dict(sys.modules, modules_patch):
        _purge_sglang_mlu()
        from sglang_mlu.attention import _register

        _register()

    assert registry_state["registered_name"] == "mlu"
    assert callable(registry_state["factory"])


# ---------------------------------------------------------------------------
# Tests — inference session alias
# ---------------------------------------------------------------------------


def test_inference_session_is_subclass() -> None:
    """``MLUAttentionInferenceSession`` is a subclass of ``MLUAttentionBackend``."""
    _purge_sglang_mlu()
    from sglang_mlu.attention import (
        MLUAttentionBackend,
        MLUAttentionInferenceSession,
    )

    assert issubclass(MLUAttentionInferenceSession, MLUAttentionBackend)
