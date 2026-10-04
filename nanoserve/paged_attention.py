"""
Paged KV Cache

The idea behind vLLM's PagedAttention, in plain PyTorch.

GPU memory for the KV cache is carved into fixed-size blocks of block_size
tokens. Each sequence owns a block table: a list of block ids, in order. Token
at position p of a sequence lives in block block_table[p // block_size], at
offset p % block_size. Blocks are handed out on demand, one at a time, and
returned the moment a sequence finishes. So:

- No reservation of max_len slots per sequence. Waste is at most one partly
  filled block per sequence, instead of (max_len - actual_len) tokens.
- No fragmentation. Any free block can serve any sequence; a sequence's
  blocks need not be adjacent.
- More sequences fit in the same memory, so the batch can be bigger, which is
  where the throughput comes from.

One forward pass can carry several sequences, each adding any number of
tokens: a whole prompt for a new sequence (prefill) or one token for a running
one (decode). Tokens arrive flattened, in the order the sequences were passed
to begin_step.

Attention here gathers each sequence's blocks into a contiguous tensor and
calls SDPA, one sequence at a time. That is correct but slow; vLLM's CUDA
kernel reads K/V straight out of the scattered blocks instead of copying them.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .config import ModelConfig


class OutOfBlocksError(RuntimeError):
    pass


class BlockAllocator:
    """Free list of physical block ids."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        # Pop from the end, so blocks are handed out 0, 1, 2, ...
        self._free = list(range(num_blocks - 1, -1, -1))
        self._allocated = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocksError("no free KV cache blocks")
        block = self._free.pop()
        self._allocated.add(block)
        return block

    def free(self, block: int):
        if block not in self._allocated:
            raise ValueError(f"block {block} is not allocated")
        self._allocated.remove(block)
        self._free.append(block)


@dataclass
class SequenceState:
    block_table: List[int] = field(default_factory=list)
    num_tokens: int = 0  # tokens whose K/V are in the cache


class PagedKVCache:
    def __init__(self, config: ModelConfig, num_blocks: int, block_size: int,
                 dtype: torch.dtype, device: str):
        shape = (num_blocks, block_size, config.num_key_value_heads, config.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)]
        self.block_size = block_size
        self.num_queries_per_kv = config.num_queries_per_kv
        self.device = device
        self.allocator = BlockAllocator(num_blocks)
        self.seqs: Dict[int, SequenceState] = {}
        self._step: List[Tuple[int, int]] = []  # (seq_id, num_new_tokens)
        self._slot_mapping = None

    # ---- sequence lifecycle -------------------------------------------------

    def add_sequence(self, seq_id: int):
        if seq_id in self.seqs:
            raise ValueError(f"sequence {seq_id} already exists")
        self.seqs[seq_id] = SequenceState()

    def free_sequence(self, seq_id: int):
        for block in self.seqs.pop(seq_id).block_table:
            self.allocator.free(block)

    def blocks_needed(self, seq_id: int, num_new_tokens: int) -> int:
        """Extra blocks a sequence needs to grow by num_new_tokens."""
        seq = self.seqs[seq_id]
        total = -(-(seq.num_tokens + num_new_tokens) // self.block_size)  # ceil division
        return max(0, total - len(seq.block_table))

    def can_append(self, step: Sequence[Tuple[int, int]]) -> bool:
        return sum(self.blocks_needed(s, n) for s, n in step) <= self.allocator.num_free

    # ---- one forward pass ---------------------------------------------------

    def begin_step(self, step: Sequence[Tuple[int, int]]) -> torch.Tensor:
        """Reserve blocks for this pass and return the positions of its tokens.

        step: (seq_id, num_new_tokens) per sequence, in the same order as the
        tokens will be concatenated in input_ids.
        """
        if not self.can_append(step):
            raise OutOfBlocksError(f"need {sum(self.blocks_needed(s, n) for s, n in step)} blocks, "
                                   f"{self.allocator.num_free} free")
        positions, slots = [], []
        for seq_id, n in step:
            seq = self.seqs[seq_id]
            for _ in range(self.blocks_needed(seq_id, n)):
                seq.block_table.append(self.allocator.allocate())
            for pos in range(seq.num_tokens, seq.num_tokens + n):
                block = seq.block_table[pos // self.block_size]
                slots.append(block * self.block_size + pos % self.block_size)
                positions.append(pos)
        self._step = list(step)
        self._slot_mapping = torch.tensor(slots, dtype=torch.long, device=self.device)
        return torch.tensor(positions, dtype=torch.long, device=self.device)

    def end_step(self):
        for seq_id, n in self._step:
            self.seqs[seq_id].num_tokens += n
        self._step = []
        self._slot_mapping = None

    def forward(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        k_cache, v_cache = self.k[layer_idx], self.v[layer_idx]
        # Write this pass's K/V into their slots. Viewing the cache as
        # [num_blocks * block_size, ...] makes a slot id a plain row index.
        k_cache.view(-1, *k_cache.shape[2:])[self._slot_mapping] = k
        v_cache.view(-1, *v_cache.shape[2:])[self._slot_mapping] = v

        out = torch.empty_like(q)
        start = 0
        for seq_id, n in self._step:
            seq = self.seqs[seq_id]
            context_len = seq.num_tokens + n
            blocks = torch.tensor(seq.block_table[:-(-context_len // self.block_size)], device=self.device)
            keys = k_cache[blocks].flatten(0, 1)[:context_len]
            values = v_cache[blocks].flatten(0, 1)[:context_len]
            keys = keys.repeat_interleave(self.num_queries_per_kv, dim=1)
            values = values.repeat_interleave(self.num_queries_per_kv, dim=1)

            # The n new tokens sit at the end of the context; new token i may
            # see context positions 0 .. seq.num_tokens + i.
            mask = torch.ones(n, context_len, dtype=torch.bool, device=q.device).tril(diagonal=seq.num_tokens)
            q_seq = q[start:start + n]
            out[start:start + n] = F.scaled_dot_product_attention(
                q_seq.transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1),
                attn_mask=mask).transpose(0, 1)
            start += n
        return out

    # ---- introspection ------------------------------------------------------

    def memory_stats(self) -> Dict[str, int]:
        used_blocks = self.allocator.num_blocks - self.allocator.num_free
        tokens = sum(s.num_tokens for s in self.seqs.values())
        return {
            "num_blocks": self.allocator.num_blocks,
            "used_blocks": used_blocks,
            "free_blocks": self.allocator.num_free,
            "cached_tokens": tokens,
            # Slots allocated but not yet holding a token (partly filled last blocks).
            "wasted_slots": used_blocks * self.block_size - tokens,
        }


def bytes_per_token(config: ModelConfig, dtype: torch.dtype) -> int:
    """KV cache bytes for one token across all layers: 2 (K and V) x layers x kv_heads x head_dim."""
    elem = torch.tensor([], dtype=dtype).element_size()
    return 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * elem
