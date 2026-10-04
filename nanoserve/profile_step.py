#!/usr/bin/env python3
"""
Profile Decode Steps

Fills the engine with a batch of requests, runs their prefill, then profiles
a few pure-decode steps with torch.profiler and reports where the time goes.

On a GPU, the key number is GPU busy time / wall time per step. nanoserve
launches hundreds of small kernels per step from Python (per layer, per
sequence in the attention loop), so the GPU idles between launches. vLLM
removes most of that gap with fused kernels (one PagedAttention kernel for
the whole batch) and CUDA graphs (the whole decode step replayed as one launch).

    python -m nanoserve.profile_step --device cuda --dtype bfloat16 --batch 32
"""

import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile

from .engine import LLMEngine, SchedulerConfig
from .loader import load_model


def main():
    parser = argparse.ArgumentParser(description="Profile nanoserve decode steps.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--trace", help="Write a Chrome trace (open in chrome://tracing or Perfetto)")
    args = parser.parse_args()

    model = load_model(args.model, getattr(torch, args.dtype), args.device)
    engine = LLMEngine(model, SchedulerConfig(max_num_seqs=args.batch, num_blocks=4096,
                                              max_num_batched_tokens=args.batch * args.prompt_len))
    g = torch.Generator().manual_seed(0)
    for _ in range(args.batch):
        prompt = torch.randint(0, model.config.vocab_size, (args.prompt_len,), generator=g).tolist()
        engine.add_request(prompt, max_tokens=10_000)

    engine.step()  # prefill everything in one step
    for _ in range(3):  # warm up decode
        engine.step()
    sync = torch.cuda.synchronize if args.device.startswith("cuda") else (lambda: None)

    sync()
    start = time.perf_counter()
    for _ in range(args.steps):
        engine.step()
    sync()
    step_ms = 1000 * (time.perf_counter() - start) / args.steps

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if args.device.startswith("cuda") else [])
    with profile(activities=activities) as prof:
        for _ in range(args.steps):
            engine.step()
        sync()

    print(f"\nDecode step: batch {args.batch}, context ~{args.prompt_len} tokens, "
          f"{step_ms:.1f} ms/step = {1000 * args.batch / step_ms:.0f} tok/s\n")

    events = prof.key_averages()
    if args.device.startswith("cuda"):
        def dev_time(e):
            return getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
        gpu_us = sum(dev_time(e) for e in events)
        wall_us = step_ms * 1000 * args.steps
        kernels = sum(e.count for e in events if dev_time(e) > 0)
        print(f"GPU busy {gpu_us / 1000 / args.steps:.1f} ms of {step_ms:.1f} ms per step "
              f"({100 * gpu_us / wall_us:.0f}% utilization), ~{kernels / args.steps:.0f} kernel launches/step\n")
        sort_by = "self_cuda_time_total"
    else:
        sort_by = "self_cpu_time_total"

    for label in ("attention", "mlp"):
        ev = next((e for e in events if e.key == label), None)
        if ev:
            print(f"{label:>10}: {ev.cpu_time_total / 1000 / args.steps:.1f} ms/step (CPU time incl. children)")
    print()
    print(events.table(sort_by=sort_by, row_limit=15))
    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"Trace written to {args.trace}")


if __name__ == "__main__":
    main()
