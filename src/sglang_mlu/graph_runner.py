"""MLU graph runner for decode-phase inference.

v0.1 status — eager fallback, NO graph capture
------------------------------------------------
MLU graph capture uses CNRT's graph API (``torch.mlu.graphs.MLUGraph``)
which differs substantially from CUDA graphs. The SGLang base class
``DecodeCudaGraphRunner`` has many CUDA-specific code paths (CUDA streams,
``torch.cuda.CUDAGraph`` objects, ``alloc_static`` buffer pools, etc.)
that would need to be extracted into a common ``DecodeRunnerBase``
abstraction before MLU can be plugged in cleanly.

For v0.1 we therefore delegate this class to SGLang's eager runner
and log a clear warning if the user did not explicitly opt out of
graph mode.

If ``support_cuda_graph()`` is True on the platform AND the user set
``--enable-cuda-graph``, SGLang may still instantiate this runner —
in which case we handle it gracefully by delegating to the parent
``forward_batch`` eager path.

See docs/design.md section "CNRT Graph Capture" for the v0.2 plan.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

_GRAPH_CAPTURE_IMPLEMENTED = False


def _log_graph_warning_once() -> None:
    """Log a one-time warning that MLU graph capture is not yet available."""
    logger.warning(
        "MLU graph capture is not yet implemented (v0.1). "
        "Decode will run in eager mode. "
        "For better prefill/decode throughput, set --disable-cuda-graph "
        "and use the eager path. "
        "See sglang-mlu/docs/design.md for the v0.2 graph capture plan."
    )


try:
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    _HAS_SGLANG = True
except ImportError:
    _HAS_SGLANG = False
    logger.debug("sglang not available — MLUGraphRunner stub")

if _HAS_SGLANG:

    class MLUGraphRunner(DecodeCudaGraphRunner):
        """MLUGraphRunner — currently delegates every operation to the
        parent ``DecodeCudaGraphRunner`` eager path.

        No MLU-specific graph capture is performed in this version.
        """

        _graph_warning_logged: bool = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Always disable CUDA graph; we are in eager mode.
            # The parent class may still allocate CUDA buffers during
            # __init__; we accept this minor overhead for v0.1.
            if not MLUGraphRunner._graph_warning_logged:
                _log_graph_warning_once()
                MLUGraphRunner._graph_warning_logged = True
            super().__init__(*args, **kwargs)

        def _cache_loc_dtype(self) -> torch.dtype:
            """MLU scatter ops prefer int32 page-table indices."""
            return torch.int32

        def _init_profile_context_and_memory_record(self) -> Any:
            """No MLU profiling context is attached in v0.1 eager mode."""
            return None  # type: ignore[return-value]

        def _create_device_graph(self) -> None:
            """No-op in v0.1; CNRT graph capture deferred to v0.2."""
            self.device_graph = None  # type: ignore[assignment]

        def _capture_graph(self) -> None:
            """No-op in v0.1; no operations are recorded."""

        def capture(self) -> None:
            """Skip graph capture — fall back to eager."""
            logger.debug("MLUGraphRunner.capture() called — skipping (no graph)")

        def execute(self, forward_batch: Any, pp_proxy_tensors: Any = None) -> "Tensor":
            """Run eager decode (SDPA + cache ops).

            v0.1 fallback: delegate to ModelRunner.forward() which routes
            through the EagerRunner.  This is slower than a captured graph
            but functionally correct.
            """
            logger.debug("MLUGraphRunner.execute() — eager path")
            return self.model_runner.forward(forward_batch)

else:
    MLUGraphRunner = None  # type: ignore[assignment,misc]

__all__ = [
    "MLUGraphRunner",
    "_GRAPH_CAPTURE_IMPLEMENTED",
]
