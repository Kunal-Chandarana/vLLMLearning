#!/usr/bin/env python3
"""
Model Download Utility

This script helps download and cache models for vLLM examples.
Features:
- Download popular models
- Verify model compatibility
- Cache management
- Progress tracking
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
from huggingface_hub import snapshot_download, list_repo_files
from transformers import AutoTokenizer, AutoConfig


class ModelDownloader:
    """Utility for downloading and managing models."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(Path.home(), ".cache", "huggingface", "transformers")
        self.recommended_models = {
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
                "microsoft/CodeGPT-small-py",
            ]
        }
    
    def list_recommended_models(self):
        """List recommended models by size category."""
        print("🤖 Recommended Models for vLLM Learning")
        print("=" * 60)
        
        for category, models in self.recommended_models.items():
            print(f"\n📦 {category.upper()} Models:")
            for i, model in enumerate(models, 1):
                print(f"   {i}. {model}")
        
        print("\n💡 Usage:")
        print("   python scripts/download_models.py --model gpt2")
        print("   python scripts/download_models.py --category small")
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a model."""
        try:
            config = AutoConfig.from_pretrained(model_name)
            
            # Estimate memory requirements (rough calculation)
            if hasattr(config, 'n_parameters'):
                params = config.n_parameters
            elif hasattr(config, 'num_parameters'):
                params = config.num_parameters  
            else:
                # Rough estimation based on hidden size and layers
                hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', 768))
                num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 12))
                vocab_size = getattr(config, 'vocab_size', 50257)
                
                # Rough parameter estimation
                params = (hidden_size * hidden_size * num_layers * 12) + (vocab_size * hidden_size * 2)
            
            # Memory estimation (parameters * 4 bytes for float32 + overhead)
            memory_gb = (params * 4) / (1024**3) * 1.5  # 1.5x for overhead
            
            return {
                "name": model_name,
                "parameters": params,
                "estimated_memory_gb": memory_gb,
                "architecture": config.model_type if hasattr(config, 'model_type') else "unknown",
                "vocab_size": getattr(config, 'vocab_size', 'unknown'),
                "max_position_embeddings": getattr(config, 'max_position_embeddings', 'unknown'),
            }
            
        except Exception as e:
            return {
                "name": model_name,
                "error": str(e),
                "parameters": "unknown",
                "estimated_memory_gb": "unknown",
            }
    
    def check_model_compatibility(self, model_name: str) -> bool:
        """Check if a model is compatible with vLLM."""
        try:
            # Try to load the config
            config = AutoConfig.from_pretrained(model_name)
            
            # Check for known compatible architectures
            compatible_architectures = {
                'gpt2', 'gpt_neo', 'gpt_neox', 'llama', 'opt', 
                'bloom', 'falcon', 'mpt', 'chatglm', 'baichuan'
            }
            
            model_type = getattr(config, 'model_type', '').lower()
            
            if model_type in compatible_architectures:
                return True
            else:
                print(f"⚠️  Warning: {model_type} architecture may not be fully supported")
                return True  # Still allow download, but warn user
                
        except Exception as e:
            print(f"❌ Error checking compatibility: {e}")
            return False
    
    def download_model(self, model_name: str, force: bool = False) -> bool:
        """Download a model and its tokenizer."""
        print(f"📥 Downloading model: {model_name}")
        
        # Check compatibility first
        if not self.check_model_compatibility(model_name):
            print("❌ Model compatibility check failed")
            return False
        
        # Show model info
        info = self.get_model_info(model_name)
        if "error" not in info:
            print(f"📊 Model info:")
            print(f"   Architecture: {info['architecture']}")
            print(f"   Parameters: {info['parameters']:,}" if isinstance(info['parameters'], int) else f"   Parameters: {info['parameters']}")
            print(f"   Estimated memory: {info['estimated_memory_gb']:.1f} GB" if isinstance(info['estimated_memory_gb'], float) else f"   Estimated memory: {info['estimated_memory_gb']}")
        
        try:
            # Download model files
            print("🔄 Downloading model files...")
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=self.cache_dir,
                force_download=force,
                resume_download=True,
            )
            
            # Verify tokenizer
            print("🔄 Verifying tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
            
            print(f"✅ Successfully downloaded {model_name}")
            print(f"📁 Cached at: {model_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {e}")
            return False
    
    def download_category(self, category: str, force: bool = False) -> List[str]:
        """Download all models in a category."""
        if category not in self.recommended_models:
            print(f"❌ Unknown category: {category}")
            print(f"Available categories: {list(self.recommended_models.keys())}")
            return []
        
        models = self.recommended_models[category]
        successful_downloads = []
        
        print(f"📦 Downloading {category} models ({len(models)} total)")
        print("=" * 50)
        
        for i, model in enumerate(models, 1):
            print(f"\n📥 [{i}/{len(models)}] {model}")
            if self.download_model(model, force):
                successful_downloads.append(model)
            else:
                print(f"❌ Failed to download {model}")
        
        print(f"\n📊 Download Summary:")
        print(f"   Successful: {len(successful_downloads)}/{len(models)}")
        print(f"   Failed: {len(models) - len(successful_downloads)}/{len(models)}")
        
        return successful_downloads
    
    def list_cached_models(self):
        """List models already cached locally."""
        print("💾 Cached Models")
        print("=" * 40)
        
        if not os.path.exists(self.cache_dir):
            print("No cache directory found")
            return
        
        cached_models = []
        for item in os.listdir(self.cache_dir):
            item_path = os.path.join(self.cache_dir, item)
            if os.path.isdir(item_path):
                # Check if it looks like a model directory
                if any(f.endswith('.json') for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))):
                    cached_models.append(item)
        
        if cached_models:
            for model in sorted(cached_models):
                model_path = os.path.join(self.cache_dir, model)
                size_mb = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_path)
                    for filename in filenames
                ) / (1024 * 1024)
                
                print(f"📁 {model} ({size_mb:.1f} MB)")
        else:
            print("No cached models found")
        
        print(f"\n📍 Cache directory: {self.cache_dir}")
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear model cache."""
        if model_name:
            model_path = os.path.join(self.cache_dir, model_name)
            if os.path.exists(model_path):
                import shutil
                shutil.rmtree(model_path)
                print(f"🗑️  Cleared cache for {model_name}")
            else:
                print(f"❌ Model {model_name} not found in cache")
        else:
            print("⚠️  This will clear ALL cached models. Type 'yes' to confirm:")
            confirmation = input("> ")
            if confirmation.lower() == 'yes':
                import shutil
                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                    print("🗑️  Cleared all model cache")
                else:
                    print("No cache directory found")
            else:
                print("Cache clear cancelled")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Download and manage models for vLLM")
    parser.add_argument("--model", type=str, help="Specific model to download")
    parser.add_argument("--category", type=str, choices=["small", "medium", "large", "xlarge"], 
                       help="Download all models in a category")
    parser.add_argument("--list", action="store_true", help="List recommended models")
    parser.add_argument("--cached", action="store_true", help="List cached models")
    parser.add_argument("--info", type=str, help="Show info about a specific model")
    parser.add_argument("--clear", type=str, nargs="?", const="", help="Clear cache (all or specific model)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--cache-dir", type=str, help="Custom cache directory")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ModelDownloader(cache_dir=args.cache_dir)
    
    if args.list:
        downloader.list_recommended_models()
    elif args.cached:
        downloader.list_cached_models()
    elif args.info:
        info = downloader.get_model_info(args.info)
        print(f"🤖 Model Information: {args.info}")
        print("=" * 50)
        for key, value in info.items():
            if key != "name":
                print(f"{key.replace('_', ' ').title()}: {value}")
    elif args.clear is not None:
        if args.clear:
            downloader.clear_cache(args.clear)
        else:
            downloader.clear_cache()
    elif args.model:
        downloader.download_model(args.model, force=args.force)
    elif args.category:
        downloader.download_category(args.category, force=args.force)
    else:
        print("🤖 vLLM Model Download Utility")
        print("=" * 40)
        print("Use --help for available options")
        print("\nQuick start:")
        print("  --list          Show recommended models")
        print("  --model gpt2    Download specific model")
        print("  --category small Download small models")
        print("  --cached        Show cached models")


if __name__ == "__main__":
    main()

