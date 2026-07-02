# SGLang-MLU

[![CI](https://github.com/ByteTora/sglang-mlu/actions/workflows/ci.yml/badge.svg)](https://github.com/ByteTora/sglang-mlu/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen)](https://www.python.org/)

Out-of-tree hardware plugin that enables [SGLang](https://github.com/sgl-project/sglang) to run LLM inference on **Cambricon MLU** accelerators (MLU370, MLU590, and newer).

## Why SGLang-MLU

SGLang is a high-performance serving framework for large language models. SGLang-MLU extends it to Cambricon's MLU hardware through SGLang's official [OOT plugin system](https://docs.sglang.io/docs/hardware-platforms/plugin) — no SGLang source changes required.

**Key design principles:**

- **Zero fork** — installs alongside upstream SGLang via `pip install -e .`
- **Lazy dispatch** — PyTorch native ops (`F.scaled_dot_product_attention`, `F.linear`) are automatically routed to CNNL kernels by `torch_mlu`
- **Graceful degradation** — every module imports cleanly without MLU hardware or drivers installed

## Quick Start

### Prerequisites

- Cambricon MLU370 or newer accelerator
- MLU driver ≥ v6.0.3
- CNToolkit ≥ v4.1.0, CNNL ≥ v1.28.0
- Python ≥ 3.9

### Install

```bash
# 1. Install Cambricon's PyTorch fork and upstream SGLang
pip install torch_mlu               # from Cambricon's PyPI
pip install sglang                  # upstream SGLang

# 2. Install the MLU plugin (dev mode)
git clone https://github.com/ByteTora/sglang-mlu.git
cd sglang-mlu
pip install -e ".[dev]"
```

### Launch a Server

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --device mlu \
    --attention-backend mlu \
    --tp-size 1 \
    --mem-fraction-static 0.8 \
    --host 127.0.0.1 \
    --port 8000
```

When MLU hardware is detected, the plugin activates automatically:

```
[MLUPlatform] MLU backend initialized (CUDA fallback disabled)
```

### Send a Request

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct",
       "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Architecture

```
SGLang Engine
  │  calls factory methods on current_platform
  │
  ▼
MLUPlatform (SRTPlatform + MLUDeviceMixin)
  │
  ├─ get_default_attention_backend()  → "mlu"
  ├─ get_graph_runner_cls()           → MLUGraphRunner
  ├─ get_mha_kv_pool_cls()            → MLUMHATokenToKVPool
  ├─ get_mla_kv_pool_cls()            → MLUMLATokenToKVPool
  ├─ get_dsa_kv_pool_cls()            → MLUDSATokenToKVPool
  └─ get_paged_allocator_cls()        → MLUPagedAllocator
        │
        ├─ MLUAttentionBackend → F.scaled_dot_product_attention → CNNL FlashAttention
        ├─ MLUGraphRunner      → eager decode (v0.1) / CNRT graph capture (v0.2)
        └─ KV cache pools      → paged allocator on CNRT runtime
```

### Stack

| Layer | Component | Backend |
|-------|-----------|---------|
| Model serving | SGLang | Python |
| Hardware plugin | sglang-mlu (this repo) | Python |
| MLU PyTorch | torch_mlu | C++/Python |
| Tensor kernels | CNNL | C++ |
| Collective comms | CNCL | C++ |
| Resource management | CNRT | C / kernel driver |
| Hardware | MLU370 / MLU590 / … | Silicon |

## Supported Models

| Model Family | Architecture | Status |
|-------------|-------------|--------|
| Llama 2/3/3.1 | MHA | ✅ Verified |
| Qwen 2/2.5 | MHA | ✅ Verified |
| Gemma 2/3 | MHA | ✅ Verified |
| DeepSeek V2/V3 | MLA | ⚠️ Functional (RoPE deferred to v0.2) |
| DeepSeek V3.2+ | DSA | ⚠️ Functional (index write deferred to v0.2) |
| Mixtral / Qwen MoE | MoE | ⚠️ Reference path (FusedEP deferred to v0.2) |

## Project Layout

```
sglang-mlu/
├── pyproject.toml                  # build config, entry-point registration
├── README.md
├── .github/workflows/ci.yml        # Python 3.9–3.12 CI matrix
├── tools/
│   └── benchmark_mlu.py            # synthetic + MLU benchmark harness
├── docs/
│   └── quantization.md             # v0.1 quantization strategy
├── src/sglang_mlu/
│   ├── __init__.py                 # activate() entry point
│   ├── utils.py                    # device detection, server-arg defaults
│   ├── device.py                   # MLUDeviceMixin — device operations
│   ├── platform.py                 # MLUPlatform — factory methods
│   ├── attention.py                # MLUAttentionBackend (SDPA → CNNL)
│   ├── graph_runner.py             # MLUGraphRunner (v0.1 eager)
│   ├── mla_attention.py            # MLA absorb/norm/cache ops
│   ├── fused_moe.py                # MoE reference + registration seam
│   ├── kernels/
│   │   └── cnnl_wrappers.py       # CNNL operator documentation
│   └── memory/
│       └── memory.py               # MHA/MLA/DSA pools + allocator
└── tests/                          # 61 tests, all passing
    ├── conftest.py
    ├── test_import_safety.py       # 12 tests
    ├── test_device.py              # 15 tests
    ├── test_attention.py           # 6 tests
    ├── test_mla.py                 # 9 tests
    ├── test_fused_moe.py           # 7 tests
    ├── test_memory.py              # 4 tests
    └── test_defaults.py            # 8 tests
```

## Development

```bash
# Run tests
pip install -e ".[dev]"
pytest tests/ -v

# Lint
ruff check src/sglang_mlu tests/

# Type check (non-blocking — sglang/torch_mlu not required)
mypy src/sglang_mlu --ignore-missing-imports

# Benchmark (CPU dummy mode, no hardware needed)
python3 tools/benchmark_mlu.py --dummy

# Benchmark (on MLU hardware)
python3 tools/benchmark_mlu.py --model-path meta-llama/Llama-3.2-1B-Instruct
```

## How It Works

SGLang discovers the MLU plugin via `setuptools` entry-points at startup:

1. `sglang_mlu.activate()` checks for MLU hardware
2. If available, returns `"sglang_mlu.platform.MLUPlatform"`
3. SGLang instantiates `MLUPlatform` as `current_platform`
4. Factory methods return MLU-specific attention backend, KV cache pools, and graph runner
5. PyTorch native ops (`F.scaled_dot_product_attention`, `F.linear`) are dispatched by `torch_mlu` to CNNL kernels automatically

## Roadmap

| Version | Milestone |
|---------|-----------|
| **v0.1** | MHA models (Llama, Qwen, Gemma) — production ready |
| **v0.2** | CNRT graph capture, MLA RoPE fusion, FusedEP MoE, FP8/INT4 quantization |
| **v0.3** | DSA index kernel, MLA weight-absorption fusion, multi-LoRA |

## License

[Apache 2.0](LICENSE)

## Acknowledgments

- [SGLang](https://github.com/sgl-project/sglang) — the base serving framework
- [Cambricon](https://www.cambricon.com) — MLU hardware and torch_mlu
- [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) — Ascend NPU reference implementation
