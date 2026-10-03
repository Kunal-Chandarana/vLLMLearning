#!/usr/bin/env python3
"""
Serving Benchmark

Load generator for any OpenAI-compatible completions server (`vllm serve`,
SGLang, TGI, ...). Streams every request so it can measure the latency
metrics that matter for LLM serving -- TTFT, TPOT, ITL, E2E -- at p50/p90/p95/p99,
plus throughput and goodput, across a sweep of load levels.

Two load modes:
- Closed loop (--concurrency 1,4,16,...): N requests in flight at all times.
  Good for finding the latency/throughput trade-off curve.
- Open loop (--request-rate 2,4,8,...): Poisson arrivals at a fixed rate,
  regardless of how fast the server responds. Closer to real traffic, and
  shows where queueing makes TTFT blow up.

Example (on a GPU box):
    vllm serve Qwen/Qwen2.5-7B-Instruct --disable-log-requests
    python benchmarks/serving_benchmark.py \\
        --model Qwen/Qwen2.5-7B-Instruct --tokenizer Qwen/Qwen2.5-7B-Instruct \\
        --input-len 1024 --output-len 256 --concurrency 1,4,16,64 \\
        --tag baseline

Example (locally, against the mock server, no GPU needed):
    python benchmarks/mock_server.py &
    python benchmarks/serving_benchmark.py --base-url http://localhost:8001 \\
        --concurrency 1,8,32 --num-requests 64
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import aiohttp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metrics import RequestResult, results_to_dicts, summarize


WORDS = ("the model server batch token cache memory latency request stream decode "
         "prefill attention kernel layer weight tensor block page queue schedule").split()


class PromptGenerator:
    """Builds random prompts of a target token length.

    Random content matters: repeated prompts would hit vLLM's prefix cache and
    make prefill look free. --shared-prefix-len deliberately adds a common
    prefix to every prompt, for measuring what prefix caching buys you.
    """

    def __init__(self, tokenizer_name: Optional[str], seed: int):
        self.rng = random.Random(seed)
        self.tokenizer = None
        if tokenizer_name:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            special = set(self.tokenizer.all_special_ids)
            self.vocab = [i for i in range(self.tokenizer.vocab_size) if i not in special]

    def _random_text(self, num_tokens: int) -> str:
        if num_tokens <= 0:
            return ""
        if self.tokenizer is None:
            # Without a tokenizer, roughly one word per token.
            return " ".join(self.rng.choice(WORDS) for _ in range(num_tokens))
        ids = [self.rng.choice(self.vocab) for _ in range(num_tokens)]
        return self.tokenizer.decode(ids)

    def generate(self, n: int, input_len: int, range_ratio: float,
                 shared_prefix_len: int) -> List[str]:
        prefix = self._random_text(shared_prefix_len)
        prompts = []
        for _ in range(n):
            low = max(1, int(input_len * (1 - range_ratio)))
            high = max(low, int(input_len * (1 + range_ratio)))
            body = self._random_text(self.rng.randint(low, high))
            prompts.append(prefix + " " + body if prefix else body)
        return prompts


async def send_request(session: aiohttp.ClientSession, url: str, model: str,
                       prompt: str, output_len: int) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # vLLM extension: keep generating to max_tokens so output length is
        # controlled by the benchmark, not by when the model emits EOS.
        "ignore_eos": True,
    }
    result = RequestResult(success=False)
    start = time.perf_counter()
    result.start_time = start
    last_token_time = None
    chunks = 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return result
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                now = time.perf_counter()
                if chunk.get("usage"):
                    result.prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                    result.output_tokens = chunk["usage"].get("completion_tokens", 0)
                choices = chunk.get("choices") or []
                if not choices or not choices[0].get("text"):
                    continue
                chunks += 1
                if last_token_time is None:
                    result.ttft = now - start
                else:
                    result.itls.append(now - last_token_time)
                last_token_time = now
        result.e2e_latency = time.perf_counter() - start
        if result.output_tokens == 0:
            # Server didn't report usage; fall back to one token per chunk.
            result.output_tokens = chunks
        result.success = chunks > 0
        if not result.success:
            result.error = "no tokens received"
    except Exception as e:  # network errors, timeouts, bad JSON
        result.error = f"{type(e).__name__}: {e}"
    return result


async def run_load(url: str, model: str, prompts: List[str], output_len: int,
                   concurrency: Optional[int], request_rate: Optional[float],
                   seed: int) -> Tuple[List[RequestResult], float]:
    """Fire all prompts at the server under one load setting."""
    rng = random.Random(seed)
    semaphore = asyncio.Semaphore(concurrency) if concurrency else None
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def limited(prompt: str) -> RequestResult:
            if semaphore is None:
                return await send_request(session, url, model, prompt, output_len)
            async with semaphore:
                return await send_request(session, url, model, prompt, output_len)

        tasks = []
        start = time.perf_counter()
        for prompt in prompts:
            tasks.append(asyncio.create_task(limited(prompt)))
            if request_rate:
                # Poisson process: exponential gaps between arrivals.
                await asyncio.sleep(rng.expovariate(request_rate))
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start
    return list(results), duration


async def resolve_model(base_url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/v1/models") as resp:
            resp.raise_for_status()
            return (await resp.json())["data"][0]["id"]


def print_summary_row(label: str, s: Dict):
    print(f"{label:<14} {s['completed']:>4}/{s['num_requests']:<4} "
          f"{s['request_throughput']:>7.2f} {s['output_token_throughput']:>9.1f} "
          f"{s['ttft_ms']['p50']:>9.1f} {s['ttft_ms']['p99']:>9.1f} "
          f"{s['tpot_ms']['p50']:>8.2f} {s['tpot_ms']['p99']:>8.2f} "
          f"{s['itl_ms']['p99']:>8.2f} {s['e2e_ms']['p99']:>10.1f}")


def parse_list(value: str, cast):
    return [cast(v) for v in value.split(",") if v.strip()]


async def main_async(args):
    base_url = args.base_url.rstrip("/")
    url = f"{base_url}/v1/completions"
    model = args.model or await resolve_model(base_url)

    if args.concurrency:
        levels = [("concurrency", c) for c in parse_list(args.concurrency, int)]
    else:
        levels = [("request_rate", r) for r in parse_list(args.request_rate, float)]

    gen = PromptGenerator(args.tokenizer, args.seed)

    print(f"Model: {model}   Server: {base_url}")
    print(f"Input ~{args.input_len} tok (+/-{args.range_ratio:.0%}), output {args.output_len} tok, "
          f"shared prefix {args.shared_prefix_len} tok")

    if args.warmup > 0:
        warm = gen.generate(args.warmup, args.input_len, 0.0, args.shared_prefix_len)
        await run_load(url, model, warm, min(args.output_len, 16), args.warmup, None, args.seed)

    print()
    print(f"{'load':<14} {'ok':>9} {'req/s':>7} {'out tok/s':>9} "
          f"{'TTFT p50':>9} {'TTFT p99':>9} {'TPOT p50':>8} {'TPOT p99':>8} "
          f"{'ITL p99':>8} {'E2E p99':>10}")
    print(f"{'':<14} {'':>9} {'':>7} {'':>9} {'(ms)':>9} {'(ms)':>9} {'(ms)':>8} {'(ms)':>8} "
          f"{'(ms)':>8} {'(ms)':>10}")

    runs = []
    for kind, value in levels:
        # Enough requests that the steady state dominates ramp-up and drain.
        n = max(args.num_requests, 4 * value) if kind == "concurrency" else args.num_requests
        prompts = gen.generate(int(n), args.input_len, args.range_ratio, args.shared_prefix_len)
        results, duration = await run_load(
            url, model, prompts, args.output_len,
            concurrency=value if kind == "concurrency" else args.max_concurrency,
            request_rate=value if kind == "request_rate" else None,
            seed=args.seed,
        )
        summary = summarize(results, duration, args.slo_ttft_ms, args.slo_tpot_ms)
        print_summary_row(f"{kind[:4]}={value:g}", summary)
        if summary["errors"]:
            print(f"   errors: {summary['errors']}")
        runs.append({"load_type": kind, "load_value": value, "summary": summary,
                     "requests": results_to_dicts(results)})

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.output_dir, f"{args.tag}-{stamp}.json")
    config = {k: v for k, v in vars(args).items()}
    config["model"] = model
    with open(path, "w") as f:
        json.dump({"config": config, "runs": runs}, f, indent=2)
    print(f"\nSaved results to {path}")
    print(f"Plot with: python benchmarks/plot_results.py {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible LLM server.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", help="Model name to request (default: first from /v1/models)")
    parser.add_argument("--tokenizer", help="HF tokenizer for exact prompt lengths (default: ~1 word per token)")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--range-ratio", type=float, default=0.0,
                        help="Vary input length uniformly by +/- this fraction")
    parser.add_argument("--shared-prefix-len", type=int, default=0,
                        help="Tokens of common prefix on every prompt (prefix caching experiments)")
    load = parser.add_mutually_exclusive_group()
    load.add_argument("--concurrency", help="Comma-separated closed-loop concurrency levels, e.g. 1,4,16,64")
    load.add_argument("--request-rate", help="Comma-separated Poisson arrival rates in req/s, e.g. 1,2,4,8")
    parser.add_argument("--max-concurrency", type=int, help="Cap in-flight requests in request-rate mode")
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--slo-ttft-ms", type=float, help="TTFT SLO for goodput")
    parser.add_argument("--slo-tpot-ms", type=float, help="TPOT SLO for goodput")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="run", help="Label used in the results filename")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = parser.parse_args()
    if not args.concurrency and not args.request_rate:
        args.concurrency = "1,4,16,64"
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
