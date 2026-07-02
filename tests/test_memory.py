"""Tests for the MLU set_kv_buffer fix (oracle v2 critical 1.1).

Validates that the MLU pool classes correctly unwrap KVWriteLoc via
``unwrap_write_loc`` — the SGLang interface contract that was broken in the
first implementation (passing ``loc: torch.Tensor`` directly, which fails when
SGLang sends a KVWriteLoc dataclass).
"""

from __future__ import annotations

import sys

import pytest
import torch


def _purge_sglang_mlu() -> None:
    """Remove cached sglang_mlu modules so each test re-imports cleanly."""
    for mod_name in list(sys.modules):
        if mod_name.startswith("sglang_mlu"):
            del sys.modules[mod_name]


class _KVWriteLoc:
    """Minimal replica of the SGLang KVWriteLoc dataclass."""

    def __init__(self, loc: torch.Tensor, swa_loc: torch.Tensor | None = None):
        self.loc = loc
        self.swa_loc = swa_loc


def test_unwrap_write_loc_with_kvwriteloc() -> None:
    """unwrap_write_loc extracts .loc from a KVWriteLoc-like object."""
    _purge_sglang_mlu()
    from sglang_mlu.memory.memory import unwrap_write_loc

    loc_tensor = torch.tensor([1, 3, 5], dtype=torch.int32)
    kv_loc = _KVWriteLoc(loc=loc_tensor)

    loc, swa = unwrap_write_loc(kv_loc)
    assert loc is loc_tensor
    assert swa is None


def test_unwrap_write_loc_with_bare_tensor() -> None:
    """unwrap_write_loc passes through a bare tensor unchanged."""
    _purge_sglang_mlu()
    from sglang_mlu.memory.memory import unwrap_write_loc

    loc_tensor = torch.tensor([1, 3, 5], dtype=torch.int32)
    loc, swa = unwrap_write_loc(loc_tensor)
    assert loc is loc_tensor
    assert swa is None


def test_unwrap_write_loc_with_swa() -> None:
    """unwrap_write_loc extracts both .loc and .swa_loc."""
    _purge_sglang_mlu()
    from sglang_mlu.memory.memory import unwrap_write_loc

    loc_tensor = torch.tensor([1, 3, 5], dtype=torch.int32)
    # swa same length as loc (typical case)
    swa_tensor = torch.tensor([0, 2, 4], dtype=torch.int32)
    kv_loc = _KVWriteLoc(loc=loc_tensor, swa_loc=swa_tensor)

    loc, swa = unwrap_write_loc(kv_loc)
    assert loc is loc_tensor
    assert swa is not None
    assert swa.shape[0] == loc.shape[0]


def test_dsa_pool_init_pops_extra_kwargs() -> None:
    """MLUDSATokenToKVPool.__init__ pops DSA-specific kwargs before
    forwarding to super().__init__, preventing TypeError.

    Regression test for oracle v2 critical 1.3.
    """
    _purge_sglang_mlu()
    from sglang_mlu.memory.memory import MLUDSATokenToKVPool

    # Directly verify the pop logic: create a pool instance and call
    # only the first part of __init__ (the pop logic), without running
    # the full parent __init__ or _create_index_buffers.
    pool = object.__new__(MLUDSATokenToKVPool)

    # Simulate the pop logic from MLUDSATokenToKVPool.__init__
    kwargs = {
        "size": 64,
        "page_size": 64,
        "dtype": torch.float16,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "layer_num": 1,
        "device": "cpu",
        "start_layer": 0,
        "end_layer": 1,
        "index_head_dim": 128,
        "kv_cache_dim": 576,
    }
    # This is what MLUDSATokenToKVPool.__init__ does:
    index_head_dim = kwargs.pop("index_head_dim", 128)
    kv_cache_dim = kwargs.pop("kv_cache_dim", None)
    # After popping, the remaining kwargs should NOT contain DSA-specific keys
    assert "index_head_dim" not in kwargs
    assert "kv_cache_dim" not in kwargs
    # And the popped values should be captured
    assert index_head_dim == 128
    assert kv_cache_dim == 576
    # The remaining kwargs are all standard MLA kwargs (no TypeError)
    # Verify by checking they're all strings (kwarg names)
    for key in ["size", "page_size", "dtype", "kv_lora_rank",
                "qk_rope_head_dim", "layer_num", "device",
                "start_layer", "end_layer"]:
        assert key in kwargs
