"""Shared test fixtures for the SGLang-MLU test suite.

These fixtures provide:
- Automatic module isolation so each test re-imports sglang_mlu fresh.
- ``no_torch_mlu`` / ``no_sglang`` guards that simulate missing dependencies.
- A ``mock_runner`` that stands in for SGLang's ``ModelRunner``.

All tests are designed to run on development machines without MLU hardware.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Autouse isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_modules():
    """Remove cached sglang_mlu modules so each test re-imports fresh.

    Many sglang_mlu modules run registration logic at import time
    (e.g. ``attention._register()``, ``mla_attention.register_mlu_mla_ops()``).
    Clearing the cache ensures each test exercises the full import path and
    that module-level side-effects don't leak between tests.
    """
    saved = {}
    for mod_name in list(sys.modules):
        if mod_name.startswith("sglang_mlu"):
            saved[mod_name] = sys.modules.pop(mod_name)

    yield

    # Tear-down: drop any sglang_mlu modules the test imported, then restore
    # the pre-fixture state so subsequent tests start clean.
    for mod_name in list(sys.modules):
        if mod_name.startswith("sglang_mlu"):
            del sys.modules[mod_name]
    sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Dependency-blocking fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def no_torch_mlu():
    """Simulate an environment where ``torch_mlu`` is not installed.

    Patches ``builtins.__import__`` so any attempt to import ``torch_mlu``
    raises ``ImportError``.  The patch is active for the duration of the test.
    """
    import builtins

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch_mlu" or name.startswith("torch_mlu."):
            raise ImportError(f"torch_mlu stub: {name}")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded_import):
        yield


@pytest.fixture
def no_sglang():
    """Simulate an environment where ``sglang`` is not installed.

    Patches ``builtins.__import__`` so any attempt to import ``sglang``
    raises ``ImportError``.  The patch is active for the duration of the test.
    """
    import builtins

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sglang" or name.startswith("sglang."):
            raise ImportError(f"sglang stub: {name}")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded_import):
        yield


# ---------------------------------------------------------------------------
# Composite fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mlu_inactive(no_torch_mlu):
    """Convenience fixture: guarantees MLU is reported as unavailable.

    Depends on ``no_torch_mlu`` so ``activate()`` returns ``None``.
    """
    yield


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_runner():
    """Return a ``MagicMock`` that stands in for SGLang's ``ModelRunner``.

    The mock exposes the attributes that ``MLUAttentionBackend`` and
    ``MLUGraphRunner`` read during ``__init__``:
    ``model_config``, ``kv_cache_dtype``, ``req_to_token_pool``,
    ``token_to_kv_pool``.
    """
    runner = MagicMock()
    runner.model_config = MagicMock()
    runner.model_config.num_attention_heads = 32
    runner.model_config.head_dim = 128
    runner.model_config.num_kv_heads = 32
    runner.kv_cache_dtype = "float16"
    runner.req_to_token_pool = MagicMock()
    runner.token_to_kv_pool = MagicMock()
    return runner
