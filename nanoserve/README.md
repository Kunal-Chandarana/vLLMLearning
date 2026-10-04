# nanoserve

A small LLM inference engine, built step by step to learn how vLLM works
inside. Each step is checked against the Hugging Face reference
implementation.

| Step | What | Status |
|---|---|---|
| 1 | Qwen2 written from scratch (GQA, RoPE, SwiGLU, RMSNorm), loading HF safetensors | ✅ |
| 2 | Paged KV cache: fixed-size blocks, block allocator, per-sequence block tables | ✅ |
| 3 | Continuous-batching scheduler: mixed prefill + decode, preemption when blocks run out | ✅ |
| 4 | OpenAI-compatible streaming server, measured with [`benchmarks/`](../benchmarks/README.md) | |
| 5 | Head-to-head with `vllm serve` on the same GPU, with a write-up of the gap | |

## Design

- **Flat token batches.** The model takes `input_ids` and `positions` of shape
  `[num_tokens]`, not padded `[batch, seq_len]`. One forward pass can then mix
  prompt tokens from new requests with single decode tokens from running ones,
  which is what continuous batching needs.
- **Stateless model, pluggable attention.** The KV cache lives in an attention
  backend passed into `forward`. Step 1 uses `ContiguousKVCache` (one sequence,
  `max_len` slots reserved up front). Step 2 swaps in a paged backend without
  touching the model.
- **Logits only where needed.** `compute_logits` runs the LM head on just the
  rows being sampled. For a 1,000-token prompt that skips a `[1000, 151936]`
  matmul.
- **RoPE tables stay float32** when weights are cast to bf16/fp16, because
  low-precision cos/sin drift at long positions.

## Paged KV cache (step 2)

`paged_attention.py` splits KV memory into fixed-size blocks of `block_size`
tokens. Each sequence has a **block table** mapping its logical blocks to
physical ones. Token `p` lives in block `block_table[p // block_size]` at
offset `p % block_size`. Blocks are allocated as a sequence grows and freed
the moment it finishes.

Why it matters, in numbers for Qwen2.5-0.5B in bf16:
- **KV cache per token:** 2 (K and V) × 24 layers × 2 KV heads × 64 dims ×
  2 bytes = **12 KiB**.
- **Contiguous cache:** must reserve the full context for every sequence:
  32,768 tokens × 12 KiB = **384 MiB per sequence**, even for a 50-token
  reply.
- **Paged cache, 16-token blocks:** wastes at most 15 slots (180 KiB) per
  sequence.

That difference is how many sequences fit in the batch at once, and batch size
is where throughput comes from.

One forward pass can carry a mix of sequences, each adding any number of
tokens: a whole prompt (prefill) or one token (decode). `begin_step` takes
`(seq_id, num_new_tokens)` pairs, allocates the blocks, and returns token
positions plus a **slot mapping**: where each new token's K/V gets written.

`generate_batch` prefills all prompts in one pass, then decodes every running
sequence together, freeing blocks as sequences hit EOS. On an M-series CPU
with 8 prompts, it produces output identical to running them one at a time,
**3.1× faster** (97 vs 31 tok/s).

**Limitation:** attention gathers each sequence's blocks into a contiguous
tensor, then runs SDPA per sequence in a Python loop. That's correct but slow.
vLLM's PagedAttention kernel reads K/V straight from the scattered blocks
inside the kernel, for the whole batch at once.

## Continuous batching (step 3)

`engine.py` has an `LLMEngine` that accepts requests at any time. Each
`step()` the scheduler builds a fresh batch under a token budget
(`max_num_batched_tokens`), in the same order as vLLM's V1 scheduler:

1. **Running requests first, oldest first.** A request still prefilling gets
   its next prompt chunk; a decoding request gets one token. If no KV block
   is free, the **newest** running request is **preempted**: its blocks are
   freed and it goes to the front of the waiting queue. When it's rescheduled,
   it recomputes its KV from prompt + output so far (vLLM's default
   "recompute" mode, rather than swapping to CPU).
2. **Then waiting requests**, first come first served, while the budget,
   `max_num_seqs` and free blocks allow. A prompt bigger than the remaining
   budget is split across steps (**chunked prefill**), so a 4,000-token
   prompt can't stall every running decode for a whole step.

A request samples a token only in a step where its scheduled tokens reach the
end of everything it knows: the last prompt chunk, or a decode step.

**Measured on the real model:** Qwen2.5-0.5B on an M-series CPU, 16 requests,
at most 8 running at once, output lengths mixed between 8 and 128 tokens:

| | Wall time | Useful tok/s | Mean TTFT |
|---|---|---|---|
| Static batching (groups of 8, each group runs to its longest request) | 12.7 s | 69 | ~5.1 s |
| Continuous batching (finished requests replaced immediately) | 9.8 s | 90 | 1.8 s |

The two give identical outputs. Static batching wastes decode slots on
requests that already finished, and makes new requests wait for the slowest
request in the group. The more uneven the output lengths, the bigger the gap.

**Tests** (`tests/test_nanoserve_engine.py`) put the scheduler under pressure
and require greedy output identical to one-request-at-a-time generation:
- a `max_num_seqs` cap, which leaves requests queued;
- a tiny token budget, which forces chunked prefill;
- a cache too small for the workload, which forces preemption and recompute;
- requests arriving mid-flight.

They also check invariants: the token budget is never exceeded, decodes keep
running while a long prompt is chunked in, every block is freed at the end,
and seeded sampling is reproducible.

## Usage

```bash
pip install torch transformers safetensors huggingface_hub
python -m nanoserve.generate --chat --prompt "What is a KV cache?" --max-tokens 128
python -m nanoserve.generate --device mps --dtype float16   # Apple GPU
```

## Tests

```bash
pytest tests/test_nanoserve_model.py                          # tiny random model, no download
NANOSERVE_REAL_MODEL=1 pytest tests/test_nanoserve_model.py   # + Qwen2.5-0.5B-Instruct (~1 GB)
pytest tests/test_nanoserve_paged.py                          # paged cache vs contiguous, batching
pytest tests/test_nanoserve_engine.py                         # scheduler: chunking, preemption, arrivals
```

The tests check prefill logits, prefill followed by token-by-token decode, and
greedy generation against `transformers`. For the real model, the full-sequence
logits match exactly in float32.

Two traps in comparing against `transformers.generate`:
- **Pass an `attention_mask`.** Qwen's pad token is its EOS token, so
  `transformers` can't infer the mask.
- **Set `repetition_penalty=1.0`.** The checkpoint's `generation_config.json`
  sets it to 1.1, which applies even under greedy decoding.
