"""MLU kernel abstractions wrapping CNNL and BANG C operators.

This package provides Python-side wrappers around Cambricon's CNNL (Cambricon
Neuware Core Library) for tensor operations used by the SGLang-MLU backend.
When CNNL is not available (e.g., development without MLU hardware), stubs
raise informative errors.
"""

from sglang_mlu.kernels.cnnl_wrappers import (
    CNNL_AVAILABLE,
    cnnl_flash_attention_score,
    cnnl_rms_norm,
    cnnl_matmul,
)

__all__ = [
    "CNNL_AVAILABLE",
    "cnnl_flash_attention_score",
    "cnnl_rms_norm",
    "cnnl_matmul",
]
