# MLU Quantization Support — v0.1

## Current strategy

The MLU plugin relies on **stock SGLang quantization** — no custom kernels or
method overrides are needed for the initial release.  torch_mlu's ATen-op
hooks transparently lower standard PyTorch ops (``F.linear``, ``F.silu``, etc.)
to CNNL kernels on MLU hardware.

## Decision tree

```
server --quantization <METHOD>  (or None)
   │
   ├─ None (auto) ─→ model has quantization_config.json?
   │                   ├─ yes → stock method (Fp8/AWQ/GPTQ) via HF config
   │                   └─ no  → None → UnquantizedLinearMethod → F.linear BF16
   │
   ├─ "fp8"        → Fp8Config.get_quant_method()
   │                   ├─ stock Fp8LinearMethod (Marlin fallback)
   │                   └─ on MLU: uses cutlass/Marlin-CPU, works on any device
   │
   ├─ "gptq"/"awq" → QuantizedMarlin / custom CUDA kernel
   │                   └─ on MLU: requires CPU fallback or custom MLU kernel
   │                              (v0.2 — see FUTURE WORK below)
   │
   └─ <unknown>    → UnquantizedLinearMethod (safe BF16 F.linear)
```

## What works today

| Quant method | Linear layer | MoE layer | Notes |
|--------------|--------------|-----------|-------|
| **BF16 / None** | `F.linear` | Triton fused MoE | full CNNL acceleration |
| **FP8 (online)** | `Fp8Config` | stock | Marlin fallback works on MLU |

## What needs future work

| Quant method | Blocker | Target |
|-------------|---------|--------|
| GPTQ / AWQ (Marlin) | Custom CUDA kernel lacks MLU port | v0.2 |
| W4A4 dynamic (int4) | Requires CNNL `bangc_quant_matmul` | v0.2 |
| FP8 (block-quant, cutlass) | Needs CNNL fused FP8 kernels | v0.2 |
| MLA FP8 absorption | W_UK / W_UV dequant fusion kernel | v0.3 |

## Why this works

1. SGLang's `get_quantization_config()` calls `current_platform.get_quantization_config()` first for OOT platforms.
2. `MLUPlatform.get_quantization_config()` returns `None` → SGLang falls through to its stock lookup.
3. `Fp8LinearMethod.apply()` ultimately calls `F.linear()` (via Marlin or per-channel fallback).
4. `F.linear` on MLU-device tensors is dispatched by torch_mlu to CNNL's MatMul kernel.

## Environment variables

None required.  If you want to force BF16 even when FP8 weights are present:

```bash
export SGLANG_MLU_FORCE_BF16=1    # future use
```
