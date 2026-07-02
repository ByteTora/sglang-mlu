"""Tests for MLUDeviceMixin.

Validates device-level operations of the SGLang-MLU plugin using mocked
``torch.mlu`` calls.  All tests run without real MLU hardware.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

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
# Tests — class identity
# ---------------------------------------------------------------------------


def test_enum_and_attrs() -> None:
    """MLUDeviceMixin sets ``_enum``, ``device_name``, ``device_type`` correctly."""
    _purge_sglang_mlu()
    from sglang_mlu.device import MLUDeviceMixin

    assert MLUDeviceMixin._enum == "oot"
    assert MLUDeviceMixin.device_name == "mlu"
    assert MLUDeviceMixin.device_type == "mlu"


# ---------------------------------------------------------------------------
# Tests — safe defaults when torch_mlu is absent
# ---------------------------------------------------------------------------


def test_safe_defaults(no_torch_mlu: None) -> None:
    """Every device query degrades gracefully without ``torch_mlu``."""
    _purge_sglang_mlu()
    from sglang_mlu.device import MLUDeviceMixin

    mixin = MLUDeviceMixin()

    # get_device_name falls back to a placeholder.
    assert mixin.get_device_name() == "MLU-unknown"

    # Total-memory query returns 0.
    assert mixin.get_device_total_memory() == 0

    # Current-memory query returns 0.0.
    assert mixin.get_current_memory_usage() == 0.0

    # Available-memory query returns (0, 0).
    free, total = mixin.get_available_memory()
    assert free == 0
    assert total == 0

    # Device UUID falls back to empty string.
    assert mixin.get_device_uuid() == ""


# ---------------------------------------------------------------------------
# Tests — safe methods that log without error when torch_mlu is absent
# ---------------------------------------------------------------------------


def test_set_device_no_error(no_torch_mlu: None) -> None:
    """``set_device`` must not raise when ``torch_mlu`` is missing."""
    _purge_sglang_mlu()
    from sglang_mlu.device import MLUDeviceMixin

    mixin = MLUDeviceMixin()
    # Should simply log a warning and return.
    mixin.set_device(torch.device("cpu"))


def test_empty_cache_no_error(no_torch_mlu: None) -> None:
    """``empty_cache`` must not raise when ``torch_mlu`` is missing."""
    _purge_sglang_mlu()
    from sglang_mlu.device import MLUDeviceMixin

    mixin = MLUDeviceMixin()
    mixin.empty_cache()


def test_synchronize_no_error(no_torch_mlu: None) -> None:
    """``synchronize`` must not raise when ``torch_mlu`` is missing."""
    _purge_sglang_mlu()
    from sglang_mlu.device import MLUDeviceMixin

    mixin = MLUDeviceMixin()
    mixin.synchronize()


# ---------------------------------------------------------------------------
# Tests — get_device
# ---------------------------------------------------------------------------


def test_get_device_raises_without_torch_mlu(no_torch_mlu: None) -> None:
    """Without ``torch_mlu`` the device type is unregistered and ``get_device``
    raises a descriptive ``RuntimeError``."""
    _purge_sglang_mlu()
    from sglang_mlu.device import MLUDeviceMixin

    mixin = MLUDeviceMixin()
    with pytest.raises(RuntimeError, match="mlu"):
        mixin.get_device(0)


def test_get_device_returns_device_with_torch_mlu() -> None:
    """With ``torch_mlu`` mocked, ``get_device`` returns a valid device."""
    # Inject a fake torch_mlu into sys.modules so the import succeeds.
    import types

    fake_torch_mlu = types.ModuleType("torch_mlu")
    sys.modules["torch_mlu"] = fake_torch_mlu

    try:
        _purge_sglang_mlu()
        from sglang_mlu.device import MLUDeviceMixin

        mixin = MLUDeviceMixin()

        # Remove any cached MLUDeviceMixin class that was built before
        # we installed the torch_mlu stub, then patch torch.device so we
        # don't depend on the real "mlu" dispatch key.
        expected = torch.device("cpu")
        with patch("torch.device") as mock_device:
            mock_device.return_value = expected
            result = mixin.get_device(0)
            # Verify torch.device was called with the MLU device type.
            mock_device.assert_called_once_with(type="mlu", index=0)
            assert result is expected
    finally:
        sys.modules.pop("torch_mlu", None)


# ---------------------------------------------------------------------------
# Tests — distributed backend
# ---------------------------------------------------------------------------


def test_get_torch_distributed_backend_str() -> None:
    """``get_torch_distributed_backend_str`` returns ``\"cncl\"``."""
    _purge_sglang_mlu()
    from sglang_mlu.device import MLUDeviceMixin

    mixin = MLUDeviceMixin()
    assert mixin.get_torch_distributed_backend_str() == "cncl"


# ---------------------------------------------------------------------------
# Tests — memory helpers
# ---------------------------------------------------------------------------


def test_memory_methods(no_torch_mlu: None) -> None:
    """Memory methods return safe numeric defaults without ``torch_mlu``."""
    _purge_sglang_mlu()
    from sglang_mlu.device import MLUDeviceMixin

    mixin = MLUDeviceMixin()
    assert mixin.get_device_total_memory(0) == 0
    assert mixin.get_current_memory_usage(None) == 0.0


# ---------------------------------------------------------------------------
# Tests — compute capability
# ---------------------------------------------------------------------------


def test_get_device_capability() -> None:
    """``get_device_capability`` always returns ``None`` (MLU has no CC)."""
    _purge_sglang_mlu()
    from sglang_mlu.device import MLUDeviceMixin

    mixin = MLUDeviceMixin()
    assert mixin.get_device_capability() is None


# ---------------------------------------------------------------------------
# Tests — seeding
# ---------------------------------------------------------------------------


def test_seed_everything_none_noop() -> None:
    """``seed_everything(None)`` is a no-op."""
    _purge_sglang_mlu()
    from sglang_mlu.device import MLUDeviceMixin

    # Should not raise.
    MLUDeviceMixin.seed_everything(None)


def test_seed_everything_rejects_negative() -> None:
    """Negative seeds raise ``ValueError``."""
    _purge_sglang_mlu()
    from sglang_mlu.device import MLUDeviceMixin

    with pytest.raises(ValueError, match="non-negative"):
        MLUDeviceMixin.seed_everything(-1)
