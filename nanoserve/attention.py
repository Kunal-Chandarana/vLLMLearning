"""
Attention Backends

ContiguousKVCache is the simplest possible backend: one sequence, one
preallocated [max_len, num_kv_heads, head_dim] buffer per layer. It is the
baseline the paged backend (step 2) gets checked against, and it shows the
problem paging solves: every sequence reserves max_len slots up front, used
or not.
"""

import torch
import torch.nn.functional as F

from .config import ModelConfig


class ContiguousKVCache:
    def __init__(self, config: ModelConfig, max_len: int, dtype: torch.dtype, device: str):
        shape = (max_len, config.num_key_value_heads, config.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)]
        self.max_len = max_len
        self.num_queries_per_kv = config.num_queries_per_kv
        self.seq_len = 0      # tokens already committed to the cache
        self.step_len = 0     # tokens being added in the current forward pass

    def begin_step(self, num_new_tokens: int):
        if self.seq_len + num_new_tokens > self.max_len:
            raise ValueError(f"KV cache full: {self.seq_len} + {num_new_tokens} > {self.max_len}")
        self.step_len = num_new_tokens

    def end_step(self):
        self.seq_len += self.step_len
        self.step_len = 0

    def forward(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        start, n = self.seq_len, q.shape[0]
        end = start + n
        self.k[layer_idx][start:end] = k
        self.v[layer_idx][start:end] = v
        keys = self.k[layer_idx][:end]
        values = self.v[layer_idx][:end]

        # GQA: each KV head serves num_queries_per_kv consecutive query heads.
        keys = keys.repeat_interleave(self.num_queries_per_kv, dim=1)
        values = values.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Query i sits at absolute position start + i and may see keys 0..start+i.
        mask = torch.ones(n, end, dtype=torch.bool, device=q.device).tril(diagonal=start)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1), attn_mask=mask)
        return out.transpose(0, 1)
