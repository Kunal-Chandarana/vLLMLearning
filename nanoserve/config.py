"""
Model Configuration

The subset of a Hugging Face config.json that the Qwen2 architecture needs.
"""

import json
import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_queries_per_kv(self) -> int:
        """Grouped-query attention: query heads sharing each KV head."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "ModelConfig":
        with open(os.path.join(model_dir, "config.json")) as f:
            return cls.from_dict(json.load(f))
