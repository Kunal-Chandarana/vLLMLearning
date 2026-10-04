#!/usr/bin/env python3
"""
Single-sequence Generation

Prefill the prompt in one forward pass, then decode one token per pass,
reusing the KV cache. This is the baseline loop that the continuous-batching
scheduler (step 3) generalizes to many sequences at once.

    python -m nanoserve.generate --prompt "The capital of France is" --max-tokens 32
"""

import argparse
import time
from typing import Iterator, List, Optional

import torch

from .attention import ContiguousKVCache
from .model import Qwen2Model


def sample(logits: torch.Tensor, temperature: float, generator: Optional[torch.Generator] = None) -> int:
    if temperature <= 0:
        return int(logits.argmax(-1))
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


def generate(model: Qwen2Model, prompt_ids: List[int], max_new_tokens: int,
             temperature: float = 0.0, eos_token_id: Optional[int] = None,
             seed: Optional[int] = None) -> Iterator[int]:
    """Yields generated token ids one at a time."""
    device = model.embed_tokens.weight.device
    dtype = model.embed_tokens.weight.dtype
    cache = ContiguousKVCache(model.config, len(prompt_ids) + max_new_tokens, dtype, device)
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

    tokens = torch.tensor(prompt_ids, device=device)
    positions = torch.arange(len(prompt_ids), device=device)
    for _ in range(max_new_tokens):
        cache.begin_step(len(tokens))
        hidden = model(tokens, positions, cache)
        cache.end_step()
        # Only the last position's logits are needed to pick the next token.
        next_id = sample(model.compute_logits(hidden[-1:])[0].cpu(), temperature, generator)
        yield next_id
        if next_id == eos_token_id:
            return
        tokens = torch.tensor([next_id], device=device)
        positions = torch.tensor([cache.seq_len], device=device)


def generate_batch(model: Qwen2Model, prompts: List[List[int]], max_new_tokens: int,
                   num_blocks: int, block_size: int = 16,
                   eos_token_id: Optional[int] = None) -> List[List[int]]:
    """Greedy-decode several prompts together on a paged KV cache.

    Pass 1 prefills every prompt in a single forward. After that, each pass
    decodes one token for every sequence still running. A sequence that hits
    EOS leaves the batch and its blocks go straight back to the free list.
    (Step 3's scheduler adds the missing half: admitting new requests mid-flight.)
    """
    from .paged_attention import PagedKVCache

    device = model.embed_tokens.weight.device
    dtype = model.embed_tokens.weight.dtype
    cache = PagedKVCache(model.config, num_blocks, block_size, dtype, device)
    outputs: List[List[int]] = [[] for _ in prompts]
    pending = {i: list(p) for i, p in enumerate(prompts)}  # tokens to feed next pass
    for i in pending:
        cache.add_sequence(i)

    while pending:
        step = [(i, len(toks)) for i, toks in pending.items()]
        positions = cache.begin_step(step)
        input_ids = torch.tensor([t for toks in pending.values() for t in toks], device=device)
        hidden = model(input_ids, positions, cache)
        cache.end_step()

        # Each sequence samples from the hidden state of its last token this pass.
        last_rows = torch.tensor([n for _, n in step], device=device).cumsum(0) - 1
        next_ids = model.compute_logits(hidden[last_rows]).argmax(-1).tolist()

        pending = {}
        for (seq_id, _), token in zip(step, next_ids):
            outputs[seq_id].append(token)
            if token == eos_token_id or len(outputs[seq_id]) == max_new_tokens:
                cache.free_sequence(seq_id)
            else:
                pending[seq_id] = [token]
    return outputs


def main():
    from transformers import AutoTokenizer
    from .loader import load_model

    parser = argparse.ArgumentParser(description="Generate text with nanoserve.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="Explain what a KV cache is in one paragraph.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device", default="cpu", help="cpu, mps or cuda")
    parser.add_argument("--chat", action="store_true", help="Wrap the prompt in the model's chat template")
    args = parser.parse_args()

    model = load_model(args.model, getattr(torch, args.dtype), args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.chat:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True)
    else:
        prompt_ids = tokenizer.encode(args.prompt)

    print(args.prompt if not args.chat else f"[user] {args.prompt}\n[assistant]", end="", flush=True)
    start = time.perf_counter()
    first_token_time = None
    out_ids = []
    for token_id in generate(model, prompt_ids, args.max_tokens, args.temperature, tokenizer.eos_token_id):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        out_ids.append(token_id)
        print(tokenizer.decode([token_id]), end="", flush=True)
    end = time.perf_counter()

    decode_tokens = max(len(out_ids) - 1, 1)
    print(f"\n\n[{len(prompt_ids)} prompt tokens, {len(out_ids)} generated | "
          f"TTFT {1000 * (first_token_time - start):.0f} ms, "
          f"TPOT {1000 * (end - first_token_time) / decode_tokens:.1f} ms]")


if __name__ == "__main__":
    main()
