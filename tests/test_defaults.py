"""Tests for ``apply_mlu_server_args_defaults``.

Validates that the MLU-specific server-argument defaults are applied correctly
to a mock ``ServerArgs`` object.  No real SGLang or MLU hardware is required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _purge_sglang_mlu() -> None:
    """Remove cached sglang_mlu modules so the next import is from scratch."""
    for key in [k for k in sys.modules if k.startswith("sglang_mlu")]:
        del sys.modules[key]


def _make_mock_server_args() -> MagicMock:
    """Return a ``MagicMock`` whose relevant attributes start as ``None``."""
    args = MagicMock()
    args.attention_backend = None
    args.prefill_attention_backend = None
    args.decode_attention_backend = None
    args.page_size = None
    args.disable_custom_all_reduce = None
    args.device = None
    args.mem_fraction_static = None
    return args


# ---------------------------------------------------------------------------
# Tests — individual defaults
# ---------------------------------------------------------------------------


def test_attention_backend() -> None:
    """``attention_backend`` is set to ``\"mlu\"``."""
    _purge_sglang_mlu()
    from sglang_mlu.utils import apply_mlu_server_args_defaults

    args = _make_mock_server_args()
    apply_mlu_server_args_defaults(args)
    assert args.attention_backend == "mlu"


def test_prefill_attention_backend() -> None:
    """``prefill_attention_backend`` is set to ``\"mlu\"``."""
    _purge_sglang_mlu()
    from sglang_mlu.utils import apply_mlu_server_args_defaults

    args = _make_mock_server_args()
    apply_mlu_server_args_defaults(args)
    assert args.prefill_attention_backend == "mlu"


def test_decode_attention_backend() -> None:
    """``decode_attention_backend`` is set to ``\"mlu\"``."""
    _purge_sglang_mlu()
    from sglang_mlu.utils import apply_mlu_server_args_defaults

    args = _make_mock_server_args()
    apply_mlu_server_args_defaults(args)
    assert args.decode_attention_backend == "mlu"


def test_device() -> None:
    """``device`` is set to ``\"mlu\"``."""
    _purge_sglang_mlu()
    from sglang_mlu.utils import apply_mlu_server_args_defaults

    args = _make_mock_server_args()
    apply_mlu_server_args_defaults(args)
    assert args.device == "mlu"


def test_page_size() -> None:
    """``page_size`` is set to ``128``."""
    _purge_sglang_mlu()
    from sglang_mlu.utils import apply_mlu_server_args_defaults

    args = _make_mock_server_args()
    apply_mlu_server_args_defaults(args)
    assert args.page_size == 128


def test_disable_custom_all_reduce() -> None:
    """``disable_custom_all_reduce`` is set to ``True``."""
    _purge_sglang_mlu()
    from sglang_mlu.utils import apply_mlu_server_args_defaults

    args = _make_mock_server_args()
    apply_mlu_server_args_defaults(args)
    assert args.disable_custom_all_reduce is True


def test_mem_fraction_static_defaults_to_08_when_missing() -> None:
    """``mem_fraction_static`` defaults to ``0.8`` when the attribute is absent.

    The implementation uses ``getattr(server_args, "mem_fraction_static", 0.8)``
    so the default only applies when the attribute does not already exist on
    the object.  We use a plain ``object`` subclass (not ``MagicMock``) so the
    attribute is genuinely absent.
    """

    class _MinimalArgs:
        attention_backend = None
        prefill_attention_backend = None
        decode_attention_backend = None
        page_size = None
        disable_custom_all_reduce = None
        device = None
        # mem_fraction_static is intentionally NOT defined.

    _purge_sglang_mlu()
    from sglang_mlu.utils import apply_mlu_server_args_defaults

    args = _MinimalArgs()
    apply_mlu_server_args_defaults(args)
    assert args.mem_fraction_static == 0.8


def test_mem_fraction_static_preserves_existing_value() -> None:
    """``mem_fraction_static`` keeps a pre-existing value when already set."""

    class _ArgsWithValue:
        attention_backend = None
        prefill_attention_backend = None
        decode_attention_backend = None
        page_size = None
        disable_custom_all_reduce = None
        device = None
        mem_fraction_static = 0.5

    _purge_sglang_mlu()
    from sglang_mlu.utils import apply_mlu_server_args_defaults

    args = _ArgsWithValue()
    apply_mlu_server_args_defaults(args)
    assert args.mem_fraction_static == 0.5
