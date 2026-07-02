# Copyright 2024 SGLang-MLU Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGLang-MLU attention backend.

Design
------
The initial MLU backend delegates all heavy lifting to SGLang's
``TorchNativeAttnBackend`` which uses ``torch.nn.functional.scaled_dot_product_attention``.
When running on MLU hardware, ``torch_mlu`` installs PyTorch dispatch hooks
that route ``scaled_dot_product_attention`` to CNNL's FlashAttention
implementation — no custom Plugins are needed.

``MLUAttentionBackend`` / ``MLUAttentionInferenceSession`` subclass ``TorchNativeAttnBackend``
only to:

  * set ``needs_cpu_seq_lens = False`` (MLU kernels do not read CPU seq lens)
  * set ``use_cuda_graph = False`` (MLU uses its own CNRT graph path)
  * document the MLU-specific contract

When the ``torch_mlu`` / ``sglang`` packages are not available,
stubs are substituted so the module imports cleanly for dev / documentation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy SGLang imports
# ---------------------------------------------------------------------------
try:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.torch_native_backend import (
        TorchNativeAttnBackend,
    )

    _HAS_SGLANG = True
except ImportError:
    _HAS_SGLANG = False

    class AttentionBackend:  # type: ignore[no-redef]
        use_cuda_graph: bool = False
        needs_cpu_seq_lens: bool = True
        use_captured_forward_metadata_for_breakable_cuda_graph: bool = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def init_forward_metadata(self, forward_batch: Any) -> None:
            pass

        def forward_extend(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            raise NotImplementedError("stub base")

        def forward_decode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            raise NotImplementedError("stub base")

        def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            raise NotImplementedError("stub base")

    class TorchNativeAttnBackend(AttentionBackend):  # type: ignore[no-redef]
        """Stub that masquerades as TorchNativeAttnBackend for dev boxes."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


if TYPE_CHECKING:
    from sglang.srt.layers.attention.torch_native_backend import (
        TorchNativeAttnBackend,
    )


# ---------------------------------------------------------------------------
# MLUAttentionBackend
# ---------------------------------------------------------------------------

class MLUAttentionBackend(TorchNativeAttnBackend):
    """MLU attention backend.

    Inherits every numerical kernel from SGLang's ``TorchNativeAttnBackend``
    (which calls ``F.scaled_dot_product_attention``).
    On MLU hardware ``torch_mlu``'s dispatch hooks send this op to CNNL's
    FlashAttention kernel automatically.

    The only MLU-specific overrides are the class-level capability flags:
    * ``use_cuda_graph = False`` — MLU uses its own CNRT graph capture.
    * ``needs_cpu_seq_lens = False`` — MLU kernels do not read CPU-side
      sequence-length tensors.
    """

    use_cuda_graph: bool = False
    needs_cpu_seq_lens: bool = False

    _provider: str = "mlu"

    def __init__(self, runner: Any) -> None:
        super().__init__(runner)  # type: ignore[call-arg]
        logger.debug(
            "MLUAttentionBackend created — SDPA will route to CNNL on MLU hardware",
        )


class MLUAttentionInferenceSession(MLUAttentionBackend):
    """Typed alias for callers that want an explicit "session" name."""


# ---------------------------------------------------------------------------
# Backend registration
# ---------------------------------------------------------------------------

def _register() -> None:
    """Register ``"mlu"`` in SGLang's global attention backend map."""
    if not _HAS_SGLANG:
        logger.debug(
            "sglang not available — skipping attention backend registration",
        )
        return

    from sglang.srt.layers.attention.attention_registry import (
        register_attention_backend,
    )

    @register_attention_backend("mlu")
    def _mlu_backend_factory(runner: Any) -> MLUAttentionBackend:
        return MLUAttentionBackend(runner)


_register()

__all__ = [
    "MLUAttentionBackend",
    "MLUAttentionInferenceSession",
]
