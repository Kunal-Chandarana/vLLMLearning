"""
Weight Loading

Downloads a Qwen2 checkpoint from the Hugging Face Hub and loads its
safetensors weights into nanoserve's Qwen2Model by parameter name.
"""

import glob
import os

import torch
from safetensors.torch import load_file

from .config import ModelConfig
from .model import Qwen2Model


def download(repo_id: str) -> str:
    """Fetch config, tokenizer and safetensors weights. Returns the local dir."""
    if os.path.isdir(repo_id):
        return repo_id
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, allow_patterns=["*.json", "*.safetensors", "*.txt"])


def _strip_prefix(name: str, prefix: str = "model.") -> str:
    return name[len(prefix):] if name.startswith(prefix) else name


def load_weights(model: Qwen2Model, state: dict):
    """Load a Hugging Face Qwen2 state dict (with its "model." prefixes)."""
    state = {_strip_prefix(k): v for k, v in state.items()}
    tied = model.config.tie_word_embeddings
    if tied:
        state.pop("lm_head.weight", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) - ({"lm_head.weight"} if tied else set()) or unexpected:
        raise RuntimeError(f"Weight mismatch. Missing: {missing}, unexpected: {unexpected}")


def cast_weights(model: Qwen2Model, dtype: torch.dtype) -> Qwen2Model:
    """Cast parameters only. The RoPE cos/sin tables stay float32: in bf16 they
    lose enough precision to drift at long positions."""
    for p in model.parameters():
        p.data = p.data.to(dtype)
    return model


def load_model(repo_id: str, dtype: torch.dtype = torch.float32, device: str = "cpu") -> Qwen2Model:
    model_dir = download(repo_id)
    model = Qwen2Model(ModelConfig.from_pretrained(model_dir))
    state = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        state.update(load_file(path))
    load_weights(model, state)
    return cast_weights(model, dtype).to(device).eval()
