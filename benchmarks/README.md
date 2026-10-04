# Serving Benchmark

A load generator for OpenAI-compatible LLM servers (`vllm serve`, SGLang, TGI)
that measures what inference teams actually track:

| Metric | What it is | What drives it |
|---|---|---|
| **TTFT** | Time to first token | Queueing + prefill (compute-bound) |
| **TPOT** | Time per output token, per request | Decode step time (memory-bandwidth-bound), batch size |
| **ITL** | Gap between streamed tokens, pooled | Decode jitter, e.g. other requests' prefills stalling the batch |
| **E2E** | Request sent to last token | TTFT + TPOT × output length |
| **Goodput** | Req/s that met the TTFT/TPOT SLOs | The honest capacity number |

Every latency metric is reported as mean / p50 / p90 / p95 / p99. Averages hide
the tail, and the tail is what users notice.

## Files

- `serving_benchmark.py`: streams requests at a sweep of load levels and saves JSON results
- `metrics.py`: metric definitions and aggregation (unit tested in `tests/test_metrics.py`)
- `plot_results.py`: 2×2 figure: throughput, TTFT, TPOT, and the throughput vs per-user speed trade-off
- `mock_server.py`: fake streaming server for testing the harness without a GPU
- `compare_results.py`: two result files side by side, with ratios
- `compare_engines.sh`: vLLM vs nanoserve on one GPU, end to end (see [nanoserve/README.md](../nanoserve/README.md#vllm-vs-nanoserve-on-a-gpu-step-5))

## Try it locally (no GPU)

```bash
pip install aiohttp fastapi uvicorn matplotlib numpy
python benchmarks/mock_server.py &
python benchmarks/serving_benchmark.py --base-url http://localhost:8001 \
    --concurrency 1,4,16,64 --num-requests 32 --tag mock
python benchmarks/plot_results.py benchmarks/results/mock-*.json
```

The mock server's numbers are made up. It only has the right *shape*: serialized
prefill makes TTFT queue up under load, and decode steps slow down as the batch grows.

## Run it for real (rented GPU)

vLLM needs an NVIDIA GPU. A single L4 / A10G (24 GB) is enough for a 7–8B model
in BF16; an H100 makes the numbers more interesting.

```bash
# On the GPU machine
pip install vllm aiohttp matplotlib
vllm serve Qwen/Qwen2.5-7B-Instruct --disable-log-requests &

python benchmarks/serving_benchmark.py \
    --model Qwen/Qwen2.5-7B-Instruct --tokenizer Qwen/Qwen2.5-7B-Instruct \
    --input-len 1024 --output-len 256 --concurrency 1,2,4,8,16,32,64,128 \
    --slo-ttft-ms 500 --slo-tpot-ms 50 --tag baseline
```

Use `--request-rate 1,2,4,8,16` instead of `--concurrency` for Poisson arrivals.
That mode is closer to real traffic, and it shows where queueing makes TTFT
blow up.

## Experiments to run

Change one thing per run, give it a `--tag`, then plot the runs together:

```bash
python benchmarks/plot_results.py benchmarks/results/baseline-*.json benchmarks/results/fp8-*.json \
    --out benchmarks/results/baseline_vs_fp8.png
```

| Experiment | Server flag | Benchmark flag | What to look for |
|---|---|---|---|
| Prefix caching | on by default in vLLM V1; compare with `--no-enable-prefix-caching` | `--shared-prefix-len 900` | TTFT drop when most of the prompt is shared |
| Chunked prefill | on by default in V1; vary `--max-num-batched-tokens` (e.g. 512 vs 8192) | long `--input-len` | ITL p99 vs TTFT trade-off |
| Quantization | `--quantization fp8` or an AWQ checkpoint | same as baseline | Throughput gain; check quality separately |
| KV cache size | `--gpu-memory-utilization 0.5` vs `0.9` | high concurrency | Where preemption starts and TTFT p99 jumps |
| Batch cap | `--max-num-seqs 16` vs `256` | high concurrency | Throughput ceiling vs TPOT |
| Speculative decoding | `--speculative-config ...` | low concurrency | TPOT gain at low load, loss at high load |
| Tensor parallelism | `--tensor-parallel-size 2` | same as baseline | Per-GPU efficiency vs latency |

For each experiment, write down *why* the numbers moved. For example: prefill
is compute-bound, so caching it removes FLOPs from TTFT. Decode is
memory-bandwidth-bound, so a bigger batch spreads each weight read across more
tokens, until the KV cache runs out.

## Notes on method

- Prompts are random tokens, so vLLM's prefix cache can't make prefill look free.
  `--shared-prefix-len` adds sharing on purpose.
- Requests send `ignore_eos: true` (a vLLM extension) so every response is
  exactly `--output-len` tokens. Other servers may ignore the flag.
- In concurrency mode each level runs at least 4 × concurrency requests, so
  steady state dominates ramp-up and drain.
- Throughput is measured over wall-clock time from the first request sent to the
  last response finished.
- A few warmup requests run first, so CUDA graph capture and other one-off
  startup costs don't land in the first measurement.
