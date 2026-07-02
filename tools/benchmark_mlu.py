#!/usr/bin/env python3
"""MLU benchmark harness for SGLang-MLU plugin.

Measures compute throughput of MLU hardware for LLM inference workloads.
When used with --server, launches an SGLang server and benchmarks it end-to-end.
Otherwise runs a smoke test with synthetic tensors.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MLU benchmark harness for SGLang-MLU plugin",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Path or HuggingFace ID of the model to benchmark",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=128,
        help="Input sequence length for benchmark prompts",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Number of output tokens to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=10,
        help="Number of iterations to run for timing",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations before timing",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Run synthetic tensor benchmark without a real model or MLU",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Launch an SGLang server and benchmark end-to-end",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you today?",
        help="Prompt text for server benchmark",
    )
    return parser.parse_args()


def detect_mlu() -> Optional[dict]:
    """Check if MLU hardware is available.

    Returns a dict with device info if MLU is found, None otherwise.
    The detection logic is defensive and runs standalone (no sglang required).
    """
    info = {
        "available": False,
        "device_name": "Unknown",
        "memory_total_gb": 0.0,
        "driver_version": "Unknown",
        "device_count": 0,
    }

    # Try torch_mlu first (native Cambricon PyTorch extension)
    try:
        import torch_mlu  # noqa: F401

        info["device_count"] = torch_mlu.mlu.device_count()
        if info["device_count"] > 0:
            info["available"] = True
            info["device_name"] = torch_mlu.mlu.get_device_name(0)
            # torch_mlu may expose memory info via cnrt or similar
            try:
                info["memory_total_gb"] = (
                    torch_mlu.mlu.get_device_properties(0).total_memory / (1024**3)
                )
            except AttributeError:
                info["memory_total_gb"] = 0.0
            try:
                info["driver_version"] = torch_mlu.mlu.driver_version
            except AttributeError:
                info["driver_version"] = "Unknown"
            return info
    except ImportError:
        pass
    except RuntimeError:
        pass

    # Fallback: check via torch with 'mlu' device type
    try:
        import torch

        if hasattr(torch, "mlu") and torch.mlu.is_available():
            info["available"] = True
            info["device_count"] = torch.mlu.device_count()
            if info["device_count"] > 0:
                try:
                    info["device_name"] = torch.cuda.get_device_name(
                        0
                    )  # MLU may reuse CUDA API
                except RuntimeError:
                    pass
                try:
                    mem = torch.cuda.get_device_properties(0).total_memory
                    info["memory_total_gb"] = mem / (1024**3)
                except RuntimeError:
                    pass
                try:
                    info["driver_version"] = torch.mlu.get_device_capability(0)
                except AttributeError:
                    pass
            return info
    except ImportError:
        pass

    return None


def _benchmark_gemm(
    device: str, warmup: int, num_iters: int
) -> dict:
    """Run GEMM (linear projection simulation) and return timing stats."""
    import torch

    # Simulate a typical LLM linear projection: [batch, seq, hidden] x [hidden, out]
    hidden = 4096
    out = 12288  # ~FFN intermediate size for Llama-7B scale

    a = torch.randn(1, hidden, device=device, dtype=torch.float16)
    b = torch.randn(hidden, out, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(warmup):
        _ = torch.mm(a, b)

    if device != "cpu":
        torch.mlu.synchronize() if hasattr(torch, "mlu") else torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = torch.mm(a, b)
    if device != "cpu":
        torch.mlu.synchronize() if hasattr(torch, "mlu") else torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # FLOPs for [1, hidden] x [hidden, out]: 2 * hidden * out
    flops_per_iter = 2 * hidden * out
    total_flops = flops_per_iter * num_iters
    tflops = total_flops / elapsed / 1e12

    return {
        "name": "GEMM[projection]",
        "elapsed_s": elapsed,
        "avg_ms": elapsed / num_iters * 1000,
        "tflops": tflops,
        "gflops": total_flops / elapsed / 1e9,
    }


def _benchmark_sdpa(
    device: str, warmup: int, num_iters: int
) -> dict:
    """Run scaled dot-product attention and return timing stats."""
    import torch
    import torch.nn.functional as F

    batch = 1
    num_heads = 32
    seq_len = 256
    head_dim = 128

    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(q, k, v)

    if device != "cpu":
        torch.mlu.synchronize() if hasattr(torch, "mlu") else torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = F.scaled_dot_product_attention(q, k, v)
    if device != "cpu":
        torch.mlu.synchronize() if hasattr(torch, "mlu") else torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Approximate FLOPs for attention: 2 * batch * heads * seq^2 * head_dim (QK^T)
    # plus 2 * batch * heads * seq^2 * head_dim (softmax @ V)
    flops_per_iter = 2 * batch * num_heads * seq_len * seq_len * head_dim
    flops_per_iter *= 2  # QK^T + softmaxV
    total_flops = flops_per_iter * num_iters
    tflops = total_flops / elapsed / 1e12

    return {
        "name": "SDPA[attention]",
        "elapsed_s": elapsed,
        "avg_ms": elapsed / num_iters * 1000,
        "tflops": tflops,
        "gflops": total_flops / elapsed / 1e9,
    }


def _benchmark_moe_dispatch(
    device: str, warmup: int, num_iters: int
) -> dict:
    """Simulate MoE dispatch + GEMM sequence and return timing stats."""
    import torch

    seq_len = 64
    hidden = 1024
    ffn_hidden = 3072  # Reduced for CPU smoke testing
    num_experts = 4
    top_k = 2

    x = torch.randn(seq_len, hidden, device=device, dtype=torch.float16)
    # Expert weights: simulate top-k routed experts
    w1 = torch.randn(num_experts, hidden, ffn_hidden, device=device, dtype=torch.float16)
    w2 = torch.randn(num_experts, ffn_hidden, hidden, device=device, dtype=torch.float16)

    # Simulate token-to-expert assignment (random top-k)
    router_logits = torch.randn(seq_len, num_experts, device=device, dtype=torch.float16)
    _, top_experts = torch.topk(router_logits, top_k, dim=-1)

    def moe_forward():
        out = torch.zeros_like(x)
        for expert_idx in range(num_experts):
            mask = (top_experts == expert_idx).any(dim=-1)
            if mask.any():
                selected = x[mask]
                # Up projection
                activated = torch.mm(selected, w1[expert_idx])
                # Activation (silu approximation via relu for speed)
                activated = torch.relu(activated)
                # Down projection
                out[mask] += torch.mm(activated, w2[expert_idx])
        return out

    # Warmup
    for _ in range(warmup):
        _ = moe_forward()

    if device != "cpu":
        torch.mlu.synchronize() if hasattr(torch, "mlu") else torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = moe_forward()
    if device != "cpu":
        torch.mlu.synchronize() if hasattr(torch, "mlu") else torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Approximate FLOPs: top_k seq tokens through 2 matmuls per expert
    tokens_per_iter = seq_len * top_k
    flops_per_iter = tokens_per_iter * (2 * hidden * ffn_hidden + 2 * ffn_hidden * hidden)
    total_flops = flops_per_iter * num_iters
    tflops = total_flops / elapsed / 1e12

    return {
        "name": "MoE[dispatch+GEMM]",
        "elapsed_s": elapsed,
        "avg_ms": elapsed / num_iters * 1000,
        "tflops": tflops,
        "gflops": total_flops / elapsed / 1e9,
    }


def _get_memory_stats(device: str) -> dict:
    """Return memory usage stats for the given device."""
    stats = {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}

    if device == "cpu":
        return stats

    try:
        import torch

        if hasattr(torch, "mlu"):
            stats["allocated_mb"] = torch.mlu.memory_allocated(device) / (1024**2)
            stats["reserved_mb"] = torch.mlu.memory_reserved(device) / (1024**2)
            stats["max_allocated_mb"] = torch.mlu.max_memory_allocated(device) / (1024**2)
        elif hasattr(torch, "cuda"):
            stats["allocated_mb"] = torch.cuda.memory_allocated(device) / (1024**2)
            stats["reserved_mb"] = torch.cuda.memory_reserved(device) / (1024**2)
            stats["max_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
    except Exception:
        pass

    return stats


def benchmark_synthetic(args: argparse.Namespace) -> None:
    """Run a synthetic GEMM + attention benchmark using MLU tensors."""
    import torch

    # Determine device
    device = "cpu"
    if args.dummy:
        device = "cpu"
        print("=" * 60)
        print("  SGLang-MLU Synthetic Benchmark (Dummy Mode)")
        print("=" * 60)
        print("  Running on CPU with synthetic tensors (no MLU required)")
    else:
        mlu_info = detect_mlu()
        if mlu_info is None:
            print("=" * 60)
            print("  ERROR: No MLU device detected!")
            print("=" * 60)
            print()
            print("  This benchmark requires Cambricon MLU hardware with")
            print("  torch_mlu installed. To run on CPU without MLU, use")
            print("  the --dummy flag:")
            print()
            print("    python tools/benchmark_mlu.py --dummy")
            print()
            sys.exit(1)

        print("=" * 60)
        print("  SGLang-MLU Synthetic Benchmark")
        print("=" * 60)
        print()
        print("  MLU Hardware Information:")
        print(f"    Device Name       : {mlu_info['device_name']}")
        print(f"    Device Count      : {mlu_info['device_count']}")
        print(f"    Memory Total      : {mlu_info['memory_total_gb']:.1f} GB")
        print(f"    Driver Version    : {mlu_info['driver_version']}")
        device = "mlu:0"

    print()
    print(f"  Configuration:")
    print(f"    Device            : {device}")
    print(f"    Warmup Iters      : {args.warmup}")
    print(f"    Benchmark Iters   : {args.num_iters}")
    print()

    # Run benchmarks
    results = []
    results.append(_benchmark_gemm(device, args.warmup, args.num_iters))
    results.append(_benchmark_sdpa(device, args.warmup, args.num_iters))
    results.append(_benchmark_moe_dispatch(device, args.warmup, args.num_iters))

    # Memory stats
    mem_stats = _get_memory_stats(device)

    # Print results
    print("-" * 60)
    print(f"  {'Benchmark':<22} {'Avg Time':>12} {'Throughput':>16}")
    print("-" * 60)
    for r in results:
        print(f"  {r['name']:<22} {r['avg_ms']:>10.2f}ms {r['tflops']:>10.2f} TFLOPS")
    print("-" * 60)
    print()

    if device != "cpu":
        print(f"  Memory Usage:")
        print(f"    Allocated         : {mem_stats['allocated_mb']:.1f} MB")
        print(f"    Reserved          : {mem_stats['reserved_mb']:.1f} MB")
        print(f"    Max Allocated     : {mem_stats['max_allocated_mb']:.1f} MB")
    else:
        print("  (Memory stats not available in dummy/CPU mode)")

    print()
    print("  Done.")


def benchmark_server(args: argparse.Namespace) -> None:
    """Launch SGLang server and benchmark end-to-end."""
    try:
        from sglang import launch_server, ServerArgs  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: sglang is not installed. Cannot run server benchmark.")
        print("Install sglang with: pip install sglang")
        sys.exit(1)

    mlu_info = detect_mlu()
    if mlu_info is None and not args.dummy:
        print("ERROR: No MLU device detected for server benchmark.")
        sys.exit(1)

    print("=" * 60)
    print("  SGLang-MLU Server Benchmark")
    print("=" * 60)
    print()
    print(f"  Model              : {args.model_path}")
    print(f"  Input Length       : {args.input_len}")
    print(f"  Output Length      : {args.output_len}")
    print(f"  Batch Size         : {args.batch_size}")
    print()

    # Launch server with MLU backend
    server_args = ServerArgs(
        model_path=args.model_path,
        device="mlu",
        tp_size=1,
        mem_fraction_static=0.8,
    )
    server_process = launch_server(server_args, timeout=300)

    try:
        import sglang  # type: ignore[import-untyped]
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        # Wait for server to be ready
        print("  Waiting for server to be ready...")
        time.sleep(10)

        # Send benchmark requests
        import requests

        url = f"http://{server_args.host}:{server_args.url_port}/generate"
        payload = {
            "text": args.prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": args.output_len,
            },
        }

        # Warmup
        print(f"  Running {args.warmup} warmup iterations...")
        for _ in range(args.warmup):
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()

        # Benchmark
        print(f"  Running {args.num_iters} benchmark iterations...")
        latency_results = []
        tokens_per_sec_results = []

        start_total = time.perf_counter()
        for i in range(args.num_iters):
            iter_start = time.perf_counter()
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            iter_elapsed = time.perf_counter() - iter_start

            latency_results.append(iter_elapsed)
            output_tokens = resp.json().get("meta_info", {}).get(
                "completion_tokens", args.output_len
            )
            tps = output_tokens / iter_elapsed
            tokens_per_sec_results.append(tps)

        total_elapsed = time.perf_counter() - start_total

        # Report
        avg_latency = sum(latency_results) / len(latency_results)
        avg_tps = sum(tokens_per_sec_results) / len(tokens_per_sec_results)
        p95_latency = sorted(latency_results)[int(0.95 * len(latency_results))]

        print()
        print("-" * 60)
        print("  Server Benchmark Results:")
        print("-" * 60)
        print(f"  Total Time         : {total_elapsed:.2f}s")
        print(f"  Avg Latency/Req    : {avg_latency:.3f}s")
        print(f"  P95 Latency        : {p95_latency:.3f}s")
        print(f"  Avg Tokens/Sec     : {avg_tps:.1f}")
        print(f"  Avg Req/Sec        : {1.0 / avg_latency:.2f}")
        print("-" * 60)

    finally:
        server_process.terminate()
        server_process.wait(timeout=30)
        print("  Server stopped.")


def main() -> None:
    """Main entry point for the MLU benchmark harness."""
    args = parse_args()

    if args.server:
        benchmark_server(args)
    else:
        benchmark_synthetic(args)


if __name__ == "__main__":
    main()
