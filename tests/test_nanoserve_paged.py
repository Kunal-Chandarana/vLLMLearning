"""
Paged KV cache tests. The reference is ContiguousKVCache, which step 1
already checked against transformers, so paged attention must produce the
same logits for every sequence -- alone, batched, and across block boundaries.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from nanoserve.attention import ContiguousKVCache
from nanoserve.generate import generate, generate_batch
from nanoserve.paged_attention import BlockAllocator, OutOfBlocksError, PagedKVCache, bytes_per_token

from test_nanoserve_model import tiny_pair


@pytest.fixture(scope="module")
def model():
    return tiny_pair(True)[1]


def contiguous_logits(model, ids):
    cache = ContiguousKVCache(model.config, len(ids), torch.float32, "cpu")
    cache.begin_step(len(ids))
    return model.compute_logits(model(torch.tensor(ids), torch.arange(len(ids)), cache))


# ---- allocator ---------------------------------------------------------------

def test_allocator_hands_out_and_reclaims_blocks():
    alloc = BlockAllocator(3)
    blocks = [alloc.allocate() for _ in range(3)]
    assert sorted(blocks) == [0, 1, 2] and alloc.num_free == 0
    with pytest.raises(OutOfBlocksError):
        alloc.allocate()
    alloc.free(blocks[1])
    assert alloc.allocate() == blocks[1]


def test_allocator_rejects_double_free():
    alloc = BlockAllocator(2)
    b = alloc.allocate()
    alloc.free(b)
    with pytest.raises(ValueError):
        alloc.free(b)


# ---- correctness vs contiguous cache ----------------------------------------

def test_single_sequence_matches_contiguous_across_blocks(model):
    """Block size 4 with a 10-token prefill and 7 decode steps crosses several block boundaries."""
    ids = torch.randint(0, 512, (17,)).tolist()
    expected = contiguous_logits(model, ids)

    cache = PagedKVCache(model.config, num_blocks=8, block_size=4, dtype=torch.float32, device="cpu")
    cache.add_sequence(0)
    rows = []
    for chunk in [ids[:10]] + [[t] for t in ids[10:]]:
        positions = cache.begin_step([(0, len(chunk))])
        rows.append(model.compute_logits(model(torch.tensor(chunk), positions, cache)))
        cache.end_step()
    torch.testing.assert_close(torch.cat(rows), expected, atol=1e-4, rtol=1e-4)


def test_mixed_prefill_and_decode_in_one_batch(model):
    """One pass carrying a new prompt alongside sequences mid-decode -- the
    shape of batch continuous batching produces -- with blocks of different
    sequences interleaved in memory."""
    seqs = {s: torch.randint(0, 512, (length,)).tolist() for s, length in [(0, 9), (1, 14), (2, 6)]}
    expected = {s: contiguous_logits(model, ids) for s, ids in seqs.items()}
    cache = PagedKVCache(model.config, num_blocks=16, block_size=4, dtype=torch.float32, device="cpu")
    got = {s: [] for s in seqs}

    def run(step, tokens):
        positions = cache.begin_step(step)
        logits = model.compute_logits(model(torch.tensor(tokens), positions, cache))
        cache.end_step()
        start = 0
        for seq_id, n in step:
            got[seq_id].append(logits[start:start + n])
            start += n

    # Pass 1: prefill seqs 0 and 1 (all but their last 3 tokens).
    cache.add_sequence(0); cache.add_sequence(1)
    run([(0, 6), (1, 11)], seqs[0][:6] + seqs[1][:11])
    # Pass 2: seq 2 arrives and prefills while 0 and 1 decode.
    cache.add_sequence(2)
    run([(0, 1), (2, 5), (1, 1)], [seqs[0][6]] + seqs[2][:5] + [seqs[1][11]])
    # Passes 3-4: everyone decodes.
    run([(0, 1), (1, 1), (2, 1)], [seqs[0][7], seqs[1][12], seqs[2][5]])
    run([(0, 1), (1, 1)], [seqs[0][8], seqs[1][13]])

    for s in seqs:
        torch.testing.assert_close(torch.cat(got[s]), expected[s], atol=1e-4, rtol=1e-4)


def test_batch_generation_matches_one_at_a_time(model):
    prompts = [torch.randint(0, 512, (n,)).tolist() for n in (5, 12, 8, 3)]
    expected = [list(generate(model, p, 15)) for p in prompts]
    assert generate_batch(model, prompts, 15, num_blocks=32, block_size=4) == expected


# ---- memory accounting -------------------------------------------------------

def test_blocks_allocated_lazily_and_freed(model):
    cache = PagedKVCache(model.config, num_blocks=10, block_size=4, dtype=torch.float32, device="cpu")
    cache.add_sequence(0)
    cache.begin_step([(0, 5)])  # 5 tokens -> 2 blocks
    cache.end_step()
    assert cache.memory_stats()["used_blocks"] == 2
    assert cache.memory_stats()["wasted_slots"] == 3
    cache.begin_step([(0, 3)])  # 8 tokens -> still 2 blocks
    cache.end_step()
    assert cache.memory_stats()["used_blocks"] == 2
    cache.free_sequence(0)
    assert cache.memory_stats()["free_blocks"] == 10


def test_out_of_blocks_is_reported_before_any_allocation(model):
    cache = PagedKVCache(model.config, num_blocks=3, block_size=4, dtype=torch.float32, device="cpu")
    cache.add_sequence(0); cache.add_sequence(1)
    assert not cache.can_append([(0, 8), (1, 8)])
    with pytest.raises(OutOfBlocksError):
        cache.begin_step([(0, 8), (1, 8)])
    assert cache.allocator.num_free == 3  # nothing leaked by the failed step


def test_bytes_per_token(model):
    # tiny model: 2 (K,V) x 3 layers x 2 kv heads x 16 head_dim x 4 bytes
    assert bytes_per_token(model.config, torch.float32) == 2 * 3 * 2 * 16 * 4


@pytest.mark.skipif(not __import__("os").environ.get("NANOSERVE_REAL_MODEL"),
                    reason="set NANOSERVE_REAL_MODEL=1 to download a real model")
def test_real_model_batch_matches_one_at_a_time():
    from nanoserve.loader import download, load_model

    model_dir = download("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    real = load_model(model_dir)
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True)
               for q in ["What is a KV cache?", "Define TTFT.", "Why is decode memory-bound?"]]
    expected = [list(generate(real, p, 24, eos_token_id=tokenizer.eos_token_id)) for p in prompts]
    assert generate_batch(real, prompts, 24, num_blocks=64, block_size=16,
                          eos_token_id=tokenizer.eos_token_id) == expected
