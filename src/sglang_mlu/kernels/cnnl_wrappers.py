"""Python interface to Cambricon CNNL (Neuware Core Library) operations.

**Important architectural note** (v0.1): CNNL operators are NOT exposed as
a Python module. Instead, torch_mlu registers them as ``torch.library``
custom ops that integrate with ``torch.compile`` and ATen dispatch.

For the initial MLU backend we rely on ``F.scaled_dot_product_attention``
which PyTorch routes to CNNL's FlashAttention kernel automatically on
MLU hardware. No custom Python→CNNL bridge is required at this stage.

Future MLU-custom kernels (fused MLA-preprocess, optimized scatter-write,
fused-RoPE) will be registered as ``torch.ops.sgl_kernel.*`` in the
main SGLang repo and called from C++ — see
``sglang/srt/layers/attention/utils.py`` for the Triton/CUDA pattern.

When torch_mlu is not installed the module provides ``_CnnlStubOp``
objects that raise ``RuntimeError`` instead of ``ImportError`` so the
package imports without hardware.
"""

from __future__ import annotations

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class _CnnlStubOp:
    """Raises a helpful RuntimeError when a CNNL op is called without MLU."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"CNNL op `{self.name}` was called but MLU SDK is not installed.\n"
            "Either:\n"
            "  1. Run this on a machine with Cambricon MLU hardware + SDK, or\n"
            "  2. Install CNToolkit >= v4.1.0 and CNNL >= v1.28.0."
        )


try:
    from torch_mlu.cnnl import (  # type: ignore[import-untyped]
        flash_attention_score,
        rms_norm,
        matmul,
    )

    assert callable(flash_attention_score), "CNNL flash_attention_score not callable"
    CNNL_AVAILABLE = True
except ImportError:
    flash_attention_score = _CnnlStubOp("flash_attention_score")  # type: ignore[assignment]
    rms_norm = _CnnlStubOp("rms_norm")  # type: ignore[assignment]
    matmul = _CnnlStubOp("matmul")  # type: ignore[assignment]
    CNNL_AVAILABLE = False
    logger.debug("CNNL not available — cnnl_wrappers exports stubs")


def is_cnnl_available() -> bool:
    return CNNL_AVAILABLE


def get_cnnl_version() -> str:
    try:
        from torch_mlu import cnnl  # type: ignore[import-untyped]

        return getattr(cnnl, "__version__", "unknown")
    except ImportError:
        return "uninstalled"


def cnnl_flash_attention_score(
    query: Any, key: Any, value: Any,
    attn_mask: Any = None, causal: bool = True,
    scale: Optional[float] = None, dropout_p: float = 0.0,
) -> Any:
    kwargs: dict[str, Any] = {"causal": causal, "dropout_p": dropout_p}
    if attn_mask is not None:
        kwargs["attn_mask"] = attn_mask
    if scale is not None:
        kwargs["scale"] = scale
    return flash_attention_score(query, key, value, **kwargs)


def cnnl_rms_norm(x: Any, gamma: Any, epsilon: float = 1e-6) -> Any:
    return rms_norm(x, gamma, epsilon=epsilon)


def cnnl_matmul(a: Any, b: Any, trans_a: bool = False, trans_b: bool = False) -> Any:
    return matmul(a, b, trans_a=trans_a, trans_b=trans_b)


__all__ = [
    "CNNL_AVAILABLE", "is_cnnl_available", "get_cnnl_version",
    "cnnl_flash_attention_score", "cnnl_rms_norm", "cnnl_matmul",
]
