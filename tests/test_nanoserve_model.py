"""
Checks nanoserve's Qwen2 against the Hugging Face reference implementation.

The tiny-model tests use randomly initialized weights, so they need no
download and run in seconds. The real-model test downloads
Qwen2.5-0.5B-Instruct (~1 GB); enable it with NANOSERVE_REAL_MODEL=1.
"""

import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from nanoserve.attention import ContiguousKVCache
from nanoserve.config import ModelConfig
from nanoserve.generate import generate
from nanoserve.loader import load_weights
from nanoserve.model import Qwen2Model


def test_config_reads_rope_theta_from_either_layout():
    base = dict(vocab_size=8, hidden_size=8, intermediate_size=8, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=8)
    assert ModelConfig.from_dict({**base, "rope_theta": 10000.0}).rope_theta == 10000.0  # transformers 4
    assert ModelConfig.from_dict({**base, "rope_parameters": {"rope_type": "default", "rope_theta": 500.0}}
                                 ).rope_theta == 500.0  # transformers 5


def tiny_pair(tie_word_embeddings: bool):
    """A random tiny Qwen2 in both implementations, with identical weights."""
    torch.manual_seed(0)
    hf_config = transformers.Qwen2Config(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        rope_theta=10000.0, tie_word_embeddings=tie_word_embeddings)
    hf = transformers.Qwen2ForCausalLM(hf_config).eval()
    ours = Qwen2Model(ModelConfig.from_dict(hf_config.to_dict())).eval()
    assert ours.config.rope_theta == 10000.0
    load_weights(ours, hf.state_dict())
    return hf, ours


@pytest.mark.parametrize("tied", [False, True])
def test_prefill_logits_match_reference(tied):
    hf, ours = tiny_pair(tied)
    ids = torch.randint(0, 512, (40,))
    expected = hf(ids.unsqueeze(0)).logits[0]

    cache = ContiguousKVCache(ours.config, 64, torch.float32, "cpu")
    cache.begin_step(len(ids))
    hidden = ours(ids, torch.arange(len(ids)), cache)
    actual = ours.compute_logits(hidden)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_incremental_decode_matches_full_prefill():
    """Prefill part of the sequence, then feed the rest one token at a time:
    every position's logits must equal a single full forward pass."""
    hf, ours = tiny_pair(False)
    ids = torch.randint(0, 512, (30,))
    expected = hf(ids.unsqueeze(0)).logits[0]

    cache = ContiguousKVCache(ours.config, 64, torch.float32, "cpu")
    split = 12
    cache.begin_step(split)
    rows = [ours.compute_logits(ours(ids[:split], torch.arange(split), cache))]
    cache.end_step()
    for pos in range(split, len(ids)):
        cache.begin_step(1)
        rows.append(ours.compute_logits(ours(ids[pos:pos + 1], torch.tensor([pos]), cache)))
        cache.end_step()
    torch.testing.assert_close(torch.cat(rows), expected, atol=1e-4, rtol=1e-4)


def test_greedy_generation_matches_reference():
    hf, ours = tiny_pair(True)
    prompt = torch.randint(0, 512, (10,)).tolist()
    expected = hf.generate(torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
                           max_new_tokens=20, do_sample=False, pad_token_id=0)[0, len(prompt):].tolist()
    assert list(generate(ours, prompt, 20)) == expected


def test_cache_overflow_raises():
    _, ours = tiny_pair(False)
    cache = ContiguousKVCache(ours.config, 8, torch.float32, "cpu")
    with pytest.raises(ValueError):
        cache.begin_step(9)


@pytest.mark.skipif(not os.environ.get("NANOSERVE_REAL_MODEL"), reason="set NANOSERVE_REAL_MODEL=1 to download a real model")
def test_real_model_greedy_matches_reference():
    from nanoserve.loader import download, load_model

    repo = "Qwen/Qwen2.5-0.5B-Instruct"
    model_dir = download(repo)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    ours = load_model(model_dir)

    prompt = tokenizer.encode("The three most important ideas in LLM inference are")
    # The checkpoint's generation_config.json sets repetition_penalty=1.1, which
    # HF applies even when greedy; turn it off to get plain argmax decoding.
    expected = hf.generate(torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
                           max_new_tokens=32, do_sample=False, repetition_penalty=1.0)[0, len(prompt):].tolist()
    actual = list(generate(ours, prompt, 32, eos_token_id=tokenizer.eos_token_id))
    assert actual == expected[:len(actual)]
