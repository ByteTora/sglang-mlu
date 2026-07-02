"""SGLang-MLU: Out-of-tree hardware plugin for Cambricon MLU inference.

Entry point registration: this module's `activate()` function is referenced by
pyproject.toml under [project.entry-points."sglang.srt.platforms"].
SGLang's platform loader discovers and calls it at startup.
"""

from sglang_mlu.utils import is_mlu_available


def activate():
    """Entry point called by SGLang's platform plugin loader.

    Returns the fully-qualified class name of the MLU platform if MLU hardware is
    available; otherwise returns None so SGLang falls back to other platforms.

    Returns:
        str or None: Fully qualified name of MLUPlatform class, or None.
    """
    if is_mlu_available():
        return "sglang_mlu.platform.MLUPlatform"
    return None


__version__ = "0.1.0"
__all__ = ["activate", "is_mlu", "mla_attention", "fused_moe"]

from sglang_mlu.utils import is_mlu  # noqa: E402
from sglang_mlu import mla_attention  # noqa: E402
from sglang_mlu import fused_moe  # noqa: E402
