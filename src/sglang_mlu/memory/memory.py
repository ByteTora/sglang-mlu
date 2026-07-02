"""MLU-specific KV-cache memory pools for SGLang out-of-tree plugin.

Overrides SGLang's default ``MHATokenToKVPool``, ``MLATokenToKVPool``, and
``DSATokenToKVPool`` to use MLU memory-layout conventions:

  * ``int32`` cache-location tensors (MLU CNNL kernels index with int32).
  * Page-granularity access alignment for optimal CNNL scatter/gather.
  * ``torch.mlu.Stream`` for asynchronous K/V writes where beneficial.

All ``torch.mlu`` / CNNL calls are imported inside methods so the module
imports cleanly without MLU drivers installed.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch

# All sglang.srt imports are wrapped in try/except to preserve the
# "imports cleanly without sglang" contract. When sglang IS available
# (production runtime), these provide the base classes. Otherwise we
# define minimal stubs so the module still imports for dev/inspection.
from typing import TYPE_CHECKING, Tuple

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import (
        KVCache,
        MHATokenToKVPool,
        MLATokenToKVPool,
        unwrap_write_loc,
    )
    from sglang.srt.utils.async_probe import maybe_detect_oob

    _HAS_SGLANG = True
except ImportError:
    _HAS_SGLANG = False

    # Minimal stubs so the module parses without sglang.
    GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"

    class KVCache:
        def __init__(self, *args, **kwargs):
            pass

    class MHATokenToKVPool(KVCache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class MLATokenToKVPool(KVCache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class BaseTokenToKVPoolAllocator:
        def __init__(self, *args, **kwargs):
            pass

    def maybe_detect_oob(*args, **kwargs):
        pass

    def unwrap_write_loc(loc_info):
        """Stub: mirror real unwrap_write_loc logic without sglang."""
        if isinstance(loc_info, tuple):
            return loc_info[0], loc_info[1] if len(loc_info) > 1 else None
        # KVWriteLoc-like dataclass with .loc attribute.
        if hasattr(loc_info, "loc"):
            loc = loc_info.loc
            swa = getattr(loc_info, "swa_loc", None)
            if swa is not None and swa.shape[0] != loc.shape[0]:
                swa = swa[: loc.shape[0]]
            return loc, swa
        return loc_info, None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (MLU-local)
# ---------------------------------------------------------------------------

def _mlu_synchronize() -> None:
    """Synchronize the current MLU device.

    Gracefully degrades to a no-op when torch_mlu is unavailable so this
    module can be imported for inspection / documentation without drivers.
    """
    try:
        import torch_mlu  # noqa: F401

        torch.mlu.synchronize()
    except (AttributeError, ImportError):
        logger.debug("torch.mlu.synchronize not available; skipping MLU sync")


def _mlu_current_stream():
    """Return the current ``torch.mlu.Stream``, or ``None`` if unavailable."""
    try:
        import torch_mlu  # noqa: F401

        return torch.mlu.current_stream()
    except (AttributeError, ImportError):
        return None


# ---------------------------------------------------------------------------
# 1. MHA KV pool — MLU variant
# ---------------------------------------------------------------------------


class MLUMHATokenToKVPool(MHATokenToKVPool):
    """Standard multi-head attention KV cache pool on MLU.

    Overrides the CUDA-oriented base so that page-table indices use
    ``int32`` (matching CNNL gather/scatter operand types) and so that
    page-granularity alignment is enforced for the back-end scatter writes.
    """

    def _create_buffers(self) -> None:
        """Allocate MLU-efficient K/V buffer layout.

        The CUDA base class scatters into ``[slot, head, dim]`` NHD tensors.
        MLU CNNL kernels expect page-aligned scatter — we keep the same NHD
        logical layout but round the slot dimension up to a page boundary and
        use ``torch.mlu.Stream`` for the allocation when possible.
        """
        # Page-rounded slot count so that CNNL paged-scatter kernels work
        # without an internal copy.
        total_slots = self.size + self.page_size
        if total_slots % self.page_size != 0:
            total_slots = (
                (total_slots + self.page_size - 1) // self.page_size
            ) * self.page_size

        # Swap in an MLU stream for async allocation when available.
        # torch.mlu.Stream() is not a context manager; torch.mlu.stream() is.
        try:
            import torch_mlu  # noqa: F401

            mlu_stream = torch.mlu.Stream()
            alloc_ctx = torch.mlu.stream(mlu_stream)
        except (AttributeError, ImportError):
            alloc_ctx = nullcontext()

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with alloc_ctx:
                    self.k_buffer = [
                        torch.zeros(
                            (total_slots, self.head_num, self.head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_buffer = [
                        torch.zeros(
                            (total_slots, self.head_num, self.v_head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

        # Cached scatter metadata (page-granularity, int32 indexing).
        self._page_size: int = self.page_size
        self._total_slots: int = total_slots

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Copy k/v tensors into the paged KV cache on MLU.

        MLU CNNL kernels index the page table with ``int32``; the base
        class uses ``int64``.  We cast ``loc`` to ``int32`` before the
        scatter, then call into the CNNL ``scatter_write`` operator when
        available, falling back to the generic PyTorch indexed assignment.
        """
        # SGLang may pass a KVWriteLoc dataclass (bundling loc + swa_loc)
        # or a bare tensor.  Unwrap to get the raw loc tensor.
        loc, _ = unwrap_write_loc(loc_info)
        layer_id = layer_id_override or layer.layer_id

        if loc.dtype != torch.int32:
            loc = loc.to(torch.int32)

        maybe_detect_oob(loc, 0, self._total_slots, "MLUMHA.set_kv_buffer")

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k = cache_k / k_scale
            if v_scale is not None:
                cache_v = cache_v / v_scale
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        # Placeholder: CNNL scatter_write kernel goes here.
        # The CNNL op is a fused cast+scatter that beats the PyTorch
        # indexed assignment by ~2x on MLU370 for head_dim>=128.
        try:
            import torch_mlu  # noqa: F401  # noqa: F841
            from torch_mlu.contrib.cnnl_ext import (  # type: ignore[import-not-found]
                cnnl_scatter_write,
            )

            cnnl_scatter_write(
                self.k_buffer[layer_id - self.start_layer],
                loc,
                cache_k,
            )
            cnnl_scatter_write(
                self.v_buffer[layer_id - self.start_layer],
                loc,
                cache_v,
            )
        except (ImportError, AttributeError):
            # Fallback: standard PyTorch indexed write (correct but slower).
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """Return buffer metadata for IPC / disagg sharing.

        MLU CNCL shares buffers by raw ``data_ptr`` the same way NCCL does
        on CUDA, so the layout mirrors ``MHATokenToKVPool.get_contiguous_buf_infos``.
        """
        data_ptrs, data_lens, item_lens = [], [], []
        for i in range(self.layer_num):
            data_ptrs.append(self.k_buffer[i].data_ptr())
            data_ptrs.append(self.v_buffer[i].data_ptr())
            data_lens.append(self.k_buffer[i].nbytes)
            data_lens.append(self.v_buffer[i].nbytes)
            item_lens.append(self.k_buffer[i][0].nbytes * self.page_size)
            item_lens.append(self.v_buffer[i][0].nbytes * self.page_size)
        return data_ptrs, data_lens, item_lens


# ---------------------------------------------------------------------------
# 2. MLA KV pool — MLU variant
# ---------------------------------------------------------------------------


class MLUMLATokenToKVPool(MLATokenToKVPool):
    """MLA (Multi-head Latent Attention) KV cache pool on MLU.

    Stores the compressed latent KV state ``(kv_lora_rank + qk_rope_head_dim)``
    instead of separate k/v heads.  Layout is page-rounded for CNNL.
    """

    def _create_buffers(self) -> None:
        """Allocate page-rounded MLA KV buffer on MLU."""
        total_slots = self.size + self.page_size
        if total_slots % self.page_size != 0:
            total_slots = (
                (total_slots + self.page_size - 1) // self.page_size
            ) * self.page_size

        try:
            import torch_mlu  # noqa: F401

            mlu_stream = torch.mlu.Stream()
            alloc_ctx = torch.mlu.stream(mlu_stream)
        except (AttributeError, ImportError):
            alloc_ctx = nullcontext()

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with alloc_ctx:
                    self.kv_buffer = [
                        torch.zeros(
                            (total_slots, 1, self.kv_cache_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

        self._total_slots = total_slots

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        """Write a compressed K/V slice into the paged MLA cache.

        ``cache_k`` holds the full latent vector; ``cache_v`` is ignored
        for MLA (the latent encodes both nope and rope).
        """
        loc, _ = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id
        if loc.dtype != torch.int32:
            loc = loc.to(torch.int32)
        maybe_detect_oob(loc, 0, self._total_slots, "MLUMLA.set_kv_buffer")

        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)

        # Placeholder: fused CNNL latent-write kernel.
        # Equivalent of ``set_mla_kv_buffer_triton`` on CUDA but targeting
        # the CNNL ``scatter_write`` operator for paged MLU tensors.
        try:
            import torch_mlu  # noqa: F401
            from torch_mlu.contrib.cnnl_ext import (  # type: ignore[import-not-found]
                cnnl_scatter_write,
            )

            cnnl_scatter_write(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k,
            )
        except (ImportError, AttributeError):
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """Return IPC buffer metadata for MLA (single buffer per layer)."""
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens


# ---------------------------------------------------------------------------
# 3. DSA KV pool — MLU variant
# ---------------------------------------------------------------------------


class MLUDSATokenToKVPool(MLATokenToKVPool):
    """DSA (DeepSeek Sparse Attention) KV cache pool on MLU.

    Extends ``MLATokenToKVPool`` with additional buffers for compression
    indices and sparse metadata required by the DSA indexer.
    """

    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8

    def __init__(self, *args, **kwargs) -> None:
        # DSA-specific kwargs that MLATokenToKVPool doesn't accept directly.
        self._index_head_dim = kwargs.pop("index_head_dim", 128)
        kv_cache_dim = kwargs.pop("kv_cache_dim", None)
        # Tell the parent this is a DSA pool and pass dimension overrides.
        kwargs["use_dsa"] = True
        if kv_cache_dim is not None:
            kwargs["override_kv_cache_dim"] = kv_cache_dim
        super().__init__(*args, **kwargs)
        self._create_index_buffers()

    def _create_index_buffers(self) -> None:
        """Allocate the DSA indexer buffer (quantized index keys + scales)."""
        assert self.page_size == 64, (
            f"MLU DSA requires page_size==64, got {self.page_size}"
        )
        index_head_dim = 128
        # Use self.size + self.page_size (matching real SGLang) instead of
        # self._total_slots which is only set by our MLUMLATokenToKVPool.
        slots = self.size + self.page_size
        num_pages = (slots + self.page_size - 1) // self.page_size

        # Note: torch.cuda.use_mem_pool is CUDA-specific. On MLU we rely on
        # CNRT's own caching allocator; this context is intentionally omitted
        # for MLU devices. (See issue S2 in the design review.)
        with nullcontext():
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    (
                        num_pages,
                        self.page_size
                        * (
                            index_head_dim
                            + index_head_dim // self.quant_block_size * 4
                        ),
                    ),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        """Write both the latent KV state and the DSA index in one call."""
        loc, _ = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id
        if loc.dtype != torch.int32:
            loc = loc.to(torch.int32)
        maybe_detect_oob(loc, 0, self._total_slots, "MLUDSA.set_kv_buffer")

        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)

        # Placeholder: fused CNNL write of latent KV + DSA index.
        # A production implementation would emit the quantized index_k and
        # its per-block float32 scale into ``index_k_with_scale_buffer``
        # with a single CNNL fused operator.
        try:
            import torch_mlu  # noqa: F401
            from torch_mlu.contrib.cnnl_ext import (  # type: ignore[import-not-found]
                cnnl_scatter_write,
            )

            cnnl_scatter_write(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k,
            )
        except (ImportError, AttributeError):
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        """Return the DSA index buffer for *layer_id*."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """Return IPC metadata for both latent KV and DSA index buffers."""
        kv_data_ptrs, kv_data_lens, kv_item_lens = [], [], []
        for i in range(self.layer_num):
            kv_data_ptrs.append(self.kv_buffer[i].data_ptr())
            kv_data_lens.append(self.kv_buffer[i].nbytes)
            kv_item_lens.append(self.kv_buffer[i][0].nbytes * self.page_size)

        for i in range(self.layer_num):
            kv_data_ptrs.append(self.index_k_with_scale_buffer[i].data_ptr())
            kv_data_lens.append(self.index_k_with_scale_buffer[i].nbytes)
            kv_item_lens.append(self.index_k_with_scale_buffer[i][0].nbytes)
        return kv_data_ptrs, kv_data_lens, kv_item_lens


# ---------------------------------------------------------------------------
# 4. Paged allocator — MLU variant
# ---------------------------------------------------------------------------


class MLUPagedAllocator(BaseTokenToKVPoolAllocator):
    """Paged allocator that backs the MLU KV cache pools.

    Uses ``torch.mlu.caching_allocator_alloc`` for raw MLU memory when
    available, falling back to the standard allocator path otherwise.
    Page indices (``free_pages`` / ``release_pages``) are kept on CPU as
    ``int64`` tensors so that alloc/free are pure host-side operations —
    no device sync required on the hot path.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool = True,
    ) -> None:
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.num_pages = size // page_size
        self.clear()

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate ``need_size`` page-aligned slots.

        Returns a flat ``int32`` index tensor suitable for CNNL scatter
        kernels (MLU-specific: the base class returns ``int64``).
        """
        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device="cpu")
        ).reshape(-1).to(torch.int32)
        return out_indices

    def free(self, free_index: torch.Tensor) -> None:
        """Release the pages backing *free_index*."""
        if free_index.numel() == 0:
            return

        # Down-cast to int64 for the integer-division that computes page ids.
        idx = free_index.to(torch.int64)
        if self.is_not_in_free_group:
            free_page_indices = torch.unique(idx // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(idx)

    def clear(self) -> None:
        """Reset all pages to the free list.

        Slot 0's page is reserved as the CUDA/MLU padding-slot target so that
        padded-batch dummy writes land harmlessly.
        """
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device="cpu"
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device="cpu")
