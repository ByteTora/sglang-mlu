"""SGLang-MLU hardware platform plugin for Cambricon MLU.

This module is the heart of the OOT plugin. It composes `MLUDeviceMixin`
(device operations) with SRTPlatform (factory methods) to register MLU
as a first-class hardware backend inside SGLang.

Startup flow:
  1. SGLang imports entry_points("sglang.srt.platforms")
  2. Calls `sglang_mlu.activate()` → if MLU is available, returns this class name
  3. Instantiates `MLUPlatform` and installs it as `current_platform`
  4. SGLang calls factory methods on this instance to obtain MLU-specific
     attention backend, KV cache pool, graph runner, etc.

Design follows the Ascend NPU backend under:
  python/sglang/srt/hardware_backend/npu/
and the plugin pattern documented in:
  docs/platforms/plugin.md
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

# ── Lazy base-class imports ───────────────────────────────────────────────
# SGLang's plugin loader resolves the class name returned by activate()
# via importlib. At that point sglang IS available, so this import succeeds.
# During local dev/CI we use TYPE_CHECKING to provide type hints without
# actually importing sglang, and guard the runtime import with try/except.
try:
    from sglang.srt.platforms.interface import SRTPlatform
except ImportError:
    if not TYPE_CHECKING:
        # Provide a stand-in base so the module still imports on dev boxes.
        class SRTPlatform:  # type: ignore[no-redef]
            """Stub for environments without sglang installed."""

            def __init_subclass__(cls, **kwargs):
                pass


if TYPE_CHECKING:
    from sglang.srt.platforms.interface import SRTPlatform  # for type checkers

from sglang_mlu.device import MLUDeviceMixin

__all__ = ["MLUPlatform"]


class MLUPlatform(SRTPlatform, MLUDeviceMixin):
    """Cambricon MLU hardware platform implementation.

    Inherits device operations from MLUDeviceMixin and SRT glue logic
    from the abstract SRTPlatform. Every method that returns a class
    or a string is a factory method the SGLang engine uses to
    instantiate MLU-specific subsystems at startup.
    """

    # ── Capability flags ────────────────────────────────────────────────

    def support_cuda_graph(self) -> bool:
        """MLU has its own graph capture; we do NOT reuse CUDA graph."""
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        """MLU piecewise capture is handled by MLUGraphRunner, not piecewise compile."""
        return False

    def supports_fp8(self) -> bool:
        """FP8 support depends on MLU generation; MLU370+ supports it."""
        return True

    def is_pin_memory_available(self) -> bool:
        """Host-side pinned memory is available via CNRT."""
        return True

    # ── Attention backend factory ───────────────────────────────────────

    def get_default_attention_backend(self) -> str:
        """Return `"mlu"` so the attention registry dispatches to our backend."""
        return "mlu"

    # ── Graph runner factory ────────────────────────────────────────────

    def get_graph_runner_cls(self) -> type:
        """Return the MLU graph runner class for decode-phase graph capture."""
        from sglang_mlu.graph_runner import MLUGraphRunner

        return MLUGraphRunner

    # ── KV cache pool factories ────────────────────────────────────────

    def get_mha_kv_pool_cls(self) -> type:
        """MHA KV pool for standard transformer models (Llama, Qwen, Gemma, etc.)."""
        from sglang_mlu.memory import MLUMHATokenToKVPool

        return MLUMHATokenToKVPool

    def get_mla_kv_pool_cls(self) -> type:
        """MLA KV pool for DeepSeek / Kimi style compressed-KV models."""
        from sglang_mlu.memory import MLUMLATokenToKVPool

        return MLUMLATokenToKVPool

    def get_dsa_kv_pool_cls(self) -> type:
        """DSA KV pool for DeepSeek V3.2+ sparse attention models."""
        from sglang_mlu.memory import MLUDSATokenToKVPool

        return MLUDSATokenToKVPool

    def get_paged_allocator_cls(self) -> type:
        """Paged allocator that backs all MLU KV cache pools."""
        from sglang_mlu.memory import MLUPagedAllocator

        return MLUPagedAllocator

    # ── Compilation & quantisation factories ───────────────────────────

    def get_compile_backend(self, mode: Optional[str] = None) -> str:
        return "inductor"

    def get_piecewise_backend_cls(self) -> type:
        """Piecewise backend for CUDA-graph-style compilation on MLU."""
        from sglang_mlu.graph_runner import MLUGraphRunner

        return MLUGraphRunner

    def get_dispatch_key_name(self) -> str:
        return "mlu"

    # ── Quantisation config ─────────────────────────────────────────────

    def get_quantization_config(self, quantization: str):
        """Quantisation config hook for OOT platforms.

        Returning ``None`` lets SGLang stock ``QuantizationConfig`` handle the
        quant string.  In v0.1 our MLU plugin relies on:

        * ``UnquantizedLinearMethod`` → ``F.linear`` (BF16) — works natively
          via torch_mlu dispatch when no quant config is present.
        * ``Fp8Config`` / ``AWQConfig`` — stock PyTorch reference kernels
          (Marlin, CPU fallback) that work on any device; MLU-specific fused
          quant kernels are left to a future release, documented in
          ``docs/design.md`` section "Quantization".
        """
        return None

    # ── Lifecycle hooks ─────────────────────────────────────────────────

    def apply_server_args_defaults(self, server_args) -> None:
        """Set MLU-specific defaults on the SGLang `ServerArgs` dataclass."""
        from sglang_mlu.utils import apply_mlu_server_args_defaults

        apply_mlu_server_args_defaults(server_args)

    def init_backend(self) -> None:
        """One-time backend initialization — called in each worker process.

        Also triggers the import of the attention backend module so its
        @register_attention_backend("mlu") decorator runs and "mlu" becomes
        resolvable in SGLang's ATTENTION_BACKENDS registry.
        """
        from sglang_mlu.utils import init_mlu_backend

        init_mlu_backend()

        # Ensure attention backend is registered (C4 fix).
        import sglang_mlu.attention  # noqa: F401

        logger.info("MLUPlatform backend initialized")

    # ── String representation ──────────────────────────────────────────

    def __repr__(self) -> str:
        return f"MLUPlatform(device={self.device_name})"
