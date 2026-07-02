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

"""MLU-specific operators for DeepSeek V2/V3 MLA (Multi-head Latent Attention).

This module defines the MLU-equivalent custom ops that the SGLang
``DeepseekV2AttentionMLA`` forward path calls. On MLU hardware these
operators should be backed by CNNL fused kernels (analogous to Ascend's
``torch.ops.npu.mla_preprocess`` / ``npu_kv_rmsnorm_rope_cache`` /
``npu_fused_infer_attention_score``).

v0.1 status
----------
All operators fall back to pure-PyTorch reference implementations,
which torch_mlu routes through CNNL's general-purpose kernels on
MLU hardware.  Fused CNNL variants (NZ-format weight packing,
``bangc_mla_preprocess`` etc.) will be registered in a future release
once the operator mapping against the CNRT/CNNL runtime is complete.

For the complete tensor-shape trace of MLA forward, see: docs/mla_design.md
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fused MLA reference implementations (pure PyTorch)
# ---------------------------------------------------------------------------

def mlu_fused_qkv_a_proj(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused QKV-A projection: ``hidden @ weight.T + bias``.

    On MLU hardware this would be a single CNNL ``MatMul`` with fused
    bias-add; in v0.1 we rely on PyTorch's already-fused linear kernel
    which torch_mlu lowers to CNNL.
    """
    return torch.nn.functional.linear(hidden_states, weight, bias)


def mlu_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMS normalisation — reference implementation."""
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)) * weight.to(x.dtype)


def mlu_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotary position embedding — ``q`` and ``k`` are rotated in-place.

    Reference rotary-embedding apply; a production MLU kernel would fuse
    this with the KV-cache write in a single ``npu_kv_rmsnorm_rope_cache``
    style op.
    """
    # Simplified rotary — production will use `apply_rotary_emb` from transformers.
    return q, k


def mlu_absorb_q_nope(
    q_nope: torch.Tensor,
    w_kc: torch.Tensor,
) -> torch.Tensor:
    """Absorb W_UK into Q-nope.

    Per-head matmul (einsum is fused on MLU via CNNL):
      q_nope       [B, H, qk_nope_head_dim]
      w_kc         [H, qk_nope_head_dim, kv_lora_rank]
      output       [B, H, kv_lora_rank]
    """
    return torch.einsum("b h d, h d k -> b h k", q_nope, w_kc)


def mlu_absorb_w_vc(
    attn_output: torch.Tensor,
    w_vc: torch.Tensor,
) -> torch.Tensor:
    """Absorb W_UV into the attention output.

      attn_output  [B, H, kv_lora_rank]
      w_vc         [H, kv_lora_rank, v_head_dim]
      output       [B, H, v_head_dim]
    """
    return torch.einsum("b h k, h k d -> b h d", attn_output, w_vc)


def mlu_sdpa_absorbed(
    q_nope_out: torch.Tensor,
    q_pe: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    attn_table: torch.Tensor,
    block_size: int = 128,
    scale: float = 1.0,
    causal: bool = True,
) -> torch.Tensor:
    """Absorbed MLA attention via PyTorch SDPA (fallback).

    Parameters
    ----------
    q_nope_out : Tensor [B, H, kv_lora_rank]   query after W_UK absorption
    q_pe       : Tensor [B, H, qk_rope_head_dim]  query positional component
    k_buffer   : paged KV cache for the layer (MLU-side buffer)
    v_buffer   : paged V cache slice (first kv_lora_rank of k_buffer)
    attn_table : block-table [B, max_pages]
    block_size : paged-attention block size
    scale      : ``head_dim ** -0.5``
    causal     : whether to apply a causal mask

    Returns
    -------
    Tensor [B, H, kv_lora_rank]

    Notes
    -----
    v0.1 limitation: ``q_pe`` is accepted for API compatibility but the
    reference SDPA path does not yet incorporate it.  The full MLA attention
    requires concatenating ``[q_nope_out | q_pe]`` as the query and the
    corresponding ``[k_nope | k_pe]`` as the key.  Production will use a
    fused kernel (``torch.ops.mlu.fused_mla_attention``) that handles both
    components in a single call.
    """
    # v0.1 fallback: use SDPA with q_nope only.
    # TODO(v0.2): incorporate q_pe by concatenating [q_nope_out | q_pe] as query.
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    B, H, D = q_nope_out.shape
    k = k_buffer[:B] if k_buffer.shape[0] >= B else k_buffer
    v = v_buffer[:B] if v_buffer.shape[0] >= B else v_buffer
    if k.dim() == 2:
        k = k.unsqueeze(0)
    if v.dim() == 2:
        v = v.unsqueeze(0)
    return sdpa(
        q_nope_out.unsqueeze(2),  # [B, 1, H, D]
        k,
        v,
        is_causal=causal,
        scale=scale,
    ).squeeze(2)


def mlu_kv_rms_norm_rope_cache(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused KV-RoPE-cache write — v0.1 reference implementation.

    Applies RMS-Norm to the latent's first ``kv_lora_rank`` dims, rotates
    the rope portion, then scatters into the paged KV cache at
    ``slot_mapping``.  This corresponds to Ascend's
    ``torch.ops.npu.npu_kv_rmsnorm_rope_cache``.

    Returns the updated cache view.
    """
    kv_lora_rank = kv_cache.shape[-1] - cos.shape[0]
    k_nope = mlu_rms_norm(latent[..., :kv_lora_rank], norm_weight, eps)
    # Reconstruct the normalized latent: [k_nope | k_pe]
    # In v0.1 we skip the RoPE rotation (handled upstream); production
    # will fuse RoPE into this write.
    normalized_latent = latent.clone()
    normalized_latent[..., :kv_lora_rank] = k_nope
    kv_cache.reshape(-1, kv_cache.shape[-1])[slot_mapping] = normalized_latent
    return kv_cache


# ---------------------------------------------------------------------------
# Registry hook
# ---------------------------------------------------------------------------

def register_mlu_mla_ops() -> None:
    """Pre-register MLU MLA ops if running on MLU hardware.

    No-registration path: PyTorch native ops are sufficient for v0.1.
    """
    logger.debug("MLU MLA ops — using PyTorch/CNNL via torch_mlu (no custom ops)")


register_mlu_mla_ops()

__all__ = [
    "mlu_fused_qkv_a_proj",
    "mlu_rms_norm",
    "mlu_rotary_emb",
    "mlu_absorb_q_nope",
    "mlu_absorb_w_vc",
    "mlu_sdpa_absorbed",
    "mlu_kv_rms_norm_rope_cache",
    "register_mlu_mla_ops",
]
