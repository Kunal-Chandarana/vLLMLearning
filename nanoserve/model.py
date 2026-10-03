"""
Qwen2 Model

A from-scratch Qwen2 decoder (no transformers model classes), written the way
serving engines lay it out:

- Inputs are a flat list of tokens, shape [num_tokens], with an explicit
  position for each token -- not a padded [batch, seq_len] tensor. That lets
  one forward pass mix tokens from many sequences (continuous batching).
- Attention over past tokens is delegated to an attention backend that owns
  the KV cache. The model itself is stateless, so swapping the simple
  contiguous cache for a paged one changes nothing here.

Architecture: token embedding -> N x [RMSNorm -> GQA attention with RoPE ->
residual -> RMSNorm -> SwiGLU MLP -> residual] -> RMSNorm -> LM head.
"""

from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class AttentionBackend(Protocol):
    def forward(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        """Store k/v for these tokens in the cache and attend over the cache.

        q: [num_tokens, num_heads, head_dim]
        k, v: [num_tokens, num_kv_heads, head_dim]
        returns: [num_tokens, num_heads, head_dim]
        """
        ...


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in float32 for stability, as the reference implementation does.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class RotaryEmbedding(nn.Module):
    """RoPE: rotates each query/key pair of dims by an angle proportional to
    the token's position, so q.k depends on relative position."""

    def __init__(self, head_dim: int, max_positions: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = torch.outer(torch.arange(max_positions, dtype=torch.float32), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_positions, head_dim]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        cos = self.cos[positions].unsqueeze(1)  # [num_tokens, 1, head_dim]
        sin = self.sin[positions].unsqueeze(1)
        q_out = q.float() * cos + self._rotate_half(q.float()) * sin
        k_out = k.float() * cos + self._rotate_half(k.float()) * sin
        return q_out.to(q.dtype), k_out.to(k.dtype)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        # Qwen2 has biases on q/k/v but not on the output projection.
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, rope: RotaryEmbedding,
                backend: AttentionBackend) -> torch.Tensor:
        n = x.shape[0]
        q = self.q_proj(x).view(n, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(n, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(n, self.num_kv_heads, self.head_dim)
        q, k = rope(positions, q, k)
        out = backend.forward(self.layer_idx, q, k, v)
        return self.o_proj(out.reshape(n, self.num_heads * self.head_dim))


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, positions, rope, backend):
        x = x + self.self_attn(self.input_layernorm(x), positions, rope, backend)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rope = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                backend: AttentionBackend) -> torch.Tensor:
        """input_ids, positions: [num_tokens]. Returns hidden states [num_tokens, hidden]."""
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, positions, self.rope, backend)
        return self.norm(x)

    @torch.inference_mode()
    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        # Callers pass only the rows they will sample from (e.g. each sequence's
        # last token), so prefill doesn't pay for a [prompt_len, vocab] matmul.
        return self.lm_head(hidden).float()
