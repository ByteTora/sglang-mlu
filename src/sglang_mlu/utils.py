"""MLU hardware detection, initialization, and server-argument defaults.

Most logic mirrors what Ascend NPU does in sglang.srt.hardware_backend.npu.utils,
but targeting the torch_mlu / CNRT / CNNL stack instead of torch_npu / CANN.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MLU_AVAILABLE: Optional[bool] = None
_MLU_BACKEND_INITIALIZED = False


def is_mlu_available() -> bool:
    """Return True if at least one Cambricon MLU device is reachable.

    The check is performed once and cached.
    """
    global _MLU_AVAILABLE
    if _MLU_AVAILABLE is not None:
        return _MLU_AVAILABLE

    try:
        import torch
        import torch_mlu  # noqa: F401 — registers the MLU backend

        _MLU_AVAILABLE = torch.mlu.is_available() and torch.mlu.device_count() > 0
    except ImportError:
        _MLU_AVAILABLE = False

    if _MLU_AVAILABLE:
        logger.info(
            "MLU available — %d device(s) detected",
            __import__("torch").mlu.device_count(),
        )
    else:
        logger.debug("MLU not available via torch_mlu — platform plugin inactive")

    return _MLU_AVAILABLE


def get_mlu_device_count() -> int:
    """Return the number of visible MLU devices (0 if unavailable)."""
    if not is_mlu_available():
        return 0
    import torch

    return torch.mlu.device_count()


def init_mlu_backend() -> None:
    """One-time MLU backend initialization, called before model construction.

    Mirrors Ascend's `init_npu_backend()`:
      - Prevent accidental CUDA fallback.
      - Set MLU environment knobs for stability.
    """
    global _MLU_BACKEND_INITIALIZED
    if _MLU_BACKEND_INITIALIZED:
        return

    import torch
    import torch_mlu  # noqa: F401

    # Prevent torch from silently using CUDA when MLU runs into an
    # unimplemented op (mirrors torch.cuda.is_available = lambda: False on NPU).
    torch.cuda.is_available = lambda: False  # type: ignore[method-assign]

    # Enable internal memory format optimizations (analogous to
    # torch_npu.npu.config.allow_internal_format = True on Ascend).
    if hasattr(torch.mlu, "config") and hasattr(torch.mlu.config, "allow_internal_format"):
        torch.mlu.config.allow_internal_format = True

    _MLU_BACKEND_INITIALIZED = True
    logger.info("MLU backend initialized (CUDA fallback disabled)")


def apply_mlu_server_args_defaults(server_args: Any) -> None:
    """Apply MLU-specific default overrides to a `ServerArgs` instance.

    Called by `MLUPlatform.apply_server_args_defaults()` during SGLang startup.
    Kept separate so it can be tested without a full SRTPlatform instance.
    """
    server_args.attention_backend = "mlu"
    server_args.prefill_attention_backend = "mlu"
    server_args.decode_attention_backend = "mlu"
    server_args.page_size = 128
    server_args.disable_custom_all_reduce = True
    server_args.device = "mlu"
    server_args.mem_fraction_static = getattr(server_args, "mem_fraction_static", 0.8)
    logger.info(
        "MLU defaults applied — attention_backend=mlu, page_size=128, "
        "disable_custom_all_reduce=True"
    )


def is_mlu() -> bool:
    """Return True if at least one MLU device is reachable.

    Adds the `device == "mlu"` short-circuit so callers don't have to
    check both separately.
    """
    return is_mlu_available()


def get_default_mlu_env() -> dict[str, str]:
    """Return recommended environment variables for MLU stability."""
    return {
        **os.environ,
        "MLU_DEVICE_DEFAULT_TIMEOUT": "60",
        "CNCL_TIMEOUT": "200",
        "CNCL_BUFFSIZE": "200",
        "STREAMS_PER_DEVICE": "32",
    }
