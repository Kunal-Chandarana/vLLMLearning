"""
Model Utilities

Helper functions for working with vLLM models.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get comprehensive information about a model."""
    try:
        from transformers import AutoConfig, AutoTokenizer
        
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Calculate approximate parameters
        if hasattr(config, 'n_parameters'):
            params = config.n_parameters
        else:
            # Rough estimation
            hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', 768))
            num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 12))
            vocab_size = getattr(config, 'vocab_size', 50257)
            params = (hidden_size * hidden_size * num_layers * 12) + (vocab_size * hidden_size * 2)
        
        return {
            "name": model_name,
            "architecture": getattr(config, 'model_type', 'unknown'),
            "parameters": params,
            "vocab_size": getattr(config, 'vocab_size', 'unknown'),
            "max_length": getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', 'unknown')),
            "hidden_size": getattr(config, 'hidden_size', getattr(config, 'd_model', 'unknown')),
            "num_layers": getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 'unknown')),
            "num_attention_heads": getattr(config, 'num_attention_heads', getattr(config, 'n_head', 'unknown')),
            "tokenizer_type": type(tokenizer).__name__,
            "estimated_memory_gb": (params * 4) / (1024**3) * 1.5,  # Rough estimate
        }
    except Exception as e:
        return {"name": model_name, "error": str(e)}


def estimate_memory_requirements(model_name: str, batch_size: int = 1, max_tokens: int = 512) -> Dict[str, float]:
    """Estimate memory requirements for a model."""
    info = get_model_info(model_name)
    
    if "error" in info:
        return {"error": info["error"]}
    
    params = info.get("parameters", 0)
    if not isinstance(params, int):
        return {"error": "Could not determine parameter count"}
    
    # Base model memory (parameters * 4 bytes for float32)
    model_memory_gb = (params * 4) / (1024**3)
    
    # KV cache memory (rough estimate)
    hidden_size = info.get("hidden_size", 768)
    num_layers = info.get("num_layers", 12)
    kv_cache_gb = (batch_size * max_tokens * hidden_size * num_layers * 2 * 4) / (1024**3)
    
    # Activation memory (rough estimate)
    activation_gb = (batch_size * max_tokens * hidden_size * 4) / (1024**3)
    
    # Total with overhead
    total_gb = (model_memory_gb + kv_cache_gb + activation_gb) * 1.2  # 20% overhead
    
    return {
        "model_memory_gb": model_memory_gb,
        "kv_cache_gb": kv_cache_gb,
        "activation_gb": activation_gb,
        "total_estimated_gb": total_gb,
        "recommended_gpu_memory_gb": total_gb * 1.5,  # Safety margin
    }


def get_optimal_vllm_config(model_name: str, available_gpus: int = 1, gpu_memory_gb: int = 16) -> Dict[str, Any]:
    """Get optimal vLLM configuration for a model and hardware setup."""
    info = get_model_info(model_name)
    
    if "error" in info:
        return {"error": info["error"]}
    
    # Calculate optimal parameters
    estimated_memory = info.get("estimated_memory_gb", 8)
    
    # Tensor parallel size
    tensor_parallel_size = min(available_gpus, max(1, int(estimated_memory / gpu_memory_gb) + 1))
    
    # GPU memory utilization
    if estimated_memory < gpu_memory_gb * 0.5:
        gpu_memory_utilization = 0.9
    elif estimated_memory < gpu_memory_gb * 0.8:
        gpu_memory_utilization = 0.8
    else:
        gpu_memory_utilization = 0.7
    
    # Max model length
    max_model_len = min(info.get("max_length", 2048), 2048)  # Conservative default
    
    # Max num sequences
    if estimated_memory < 2:
        max_num_seqs = 256
    elif estimated_memory < 8:
        max_num_seqs = 128
    else:
        max_num_seqs = 64
    
    return {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "trust_remote_code": True,
        "estimated_memory_per_gpu_gb": estimated_memory / tensor_parallel_size,
        "recommendation": f"Use {tensor_parallel_size} GPU(s) with {gpu_memory_utilization*100:.0f}% memory utilization"
    }


def validate_model_compatibility(model_name: str) -> Dict[str, Any]:
    """Validate if a model is compatible with vLLM."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        
        # Known compatible architectures
        compatible_architectures = {
            'gpt2', 'gpt_neo', 'gpt_neox', 'llama', 'opt', 'bloom', 
            'falcon', 'mpt', 'chatglm', 'baichuan', 'qwen', 'mistral'
        }
        
        model_type = getattr(config, 'model_type', '').lower()
        
        is_compatible = model_type in compatible_architectures
        
        return {
            "model_name": model_name,
            "model_type": model_type,
            "is_compatible": is_compatible,
            "confidence": "high" if is_compatible else "low",
            "notes": f"Architecture '{model_type}' is {'supported' if is_compatible else 'not officially supported'}"
        }
        
    except Exception as e:
        return {
            "model_name": model_name,
            "is_compatible": False,
            "error": str(e),
            "confidence": "unknown"
        }


def save_model_config(model_name: str, config: Dict[str, Any], config_dir: str = "configs"):
    """Save model configuration to a file."""
    os.makedirs(config_dir, exist_ok=True)
    
    # Sanitize model name for filename
    safe_name = model_name.replace("/", "_").replace(":", "_")
    config_file = os.path.join(config_dir, f"{safe_name}_config.json")
    
    config_data = {
        "model_name": model_name,
        "config": config,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "created_by": "vLLM Learning Project"
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    return config_file


def load_model_config(model_name: str, config_dir: str = "configs") -> Optional[Dict[str, Any]]:
    """Load model configuration from a file."""
    safe_name = model_name.replace("/", "_").replace(":", "_")
    config_file = os.path.join(config_dir, f"{safe_name}_config.json")
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    
    return None


def list_recommended_models() -> Dict[str, List[str]]:
    """Get list of recommended models by category."""
    return {
        "small": [
            "gpt2",
            "distilgpt2",
            "microsoft/DialoGPT-small",
        ],
        "medium": [
            "gpt2-medium",
            "microsoft/DialoGPT-medium",
            "facebook/opt-350m",
        ],
        "large": [
            "gpt2-large",
            "microsoft/DialoGPT-large",
            "facebook/opt-1.3b",
            "EleutherAI/gpt-neo-1.3B",
        ],
        "xlarge": [
            "gpt2-xl",
            "facebook/opt-2.7b",
            "EleutherAI/gpt-neo-2.7B",
        ]
    }

