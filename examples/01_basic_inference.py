#!/usr/bin/env python3
"""
Basic vLLM Inference Example

This example demonstrates the fundamental usage of vLLM for text generation.
It covers:
- Loading a model with vLLM
- Basic text generation
- Different sampling parameters
- Error handling
"""

import os
import sys
from typing import List, Optional
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_model_parallel


def basic_inference_example():
    """Demonstrate basic vLLM inference with a small model."""
    
    print("🚀 vLLM Basic Inference Example")
    print("=" * 50)
    
    # Model configuration - using a smaller model for testing
    model_name = "microsoft/DialoGPT-medium"  # ~774M parameters
    
    # Alternative smaller models you can try:
    # model_name = "gpt2"  # ~124M parameters
    # model_name = "distilgpt2"  # ~82M parameters
    
    print(f"📦 Loading model: {model_name}")
    
    try:
        # Initialize vLLM engine
        # For larger models, you might need to adjust these parameters
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  # Number of GPUs to use
            max_model_len=512,       # Maximum sequence length
            gpu_memory_utilization=0.8,  # GPU memory utilization
            trust_remote_code=True,  # Trust remote code for some models
        )
        
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Tips:")
        print("- Ensure you have sufficient GPU memory")
        print("- Try a smaller model like 'gpt2' or 'distilgpt2'")
        print("- Check CUDA installation if using GPU")
        return
    
    # Sample prompts for testing
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The most important skill for the 21st century is",
        "Climate change can be addressed by",
    ]
    
    # Sampling parameters - experiment with these!
    sampling_params = SamplingParams(
        temperature=0.8,      # Controls randomness (0.0 = deterministic, 1.0+ = creative)
        top_p=0.95,          # Nucleus sampling (consider top 95% probability mass)
        top_k=50,            # Consider only top 50 tokens
        max_tokens=100,      # Maximum tokens to generate
        repetition_penalty=1.1,  # Penalize repetition
    )
    
    print(f"\n🎯 Generating responses for {len(prompts)} prompts...")
    print(f"📊 Sampling params: temp={sampling_params.temperature}, "
          f"top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}")
    print("-" * 80)
    
    # Generate responses
    try:
        outputs = llm.generate(prompts, sampling_params)
        
        # Display results
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            
            print(f"\n📝 Prompt {i+1}: {prompt}")
            print(f"🤖 Generated: {generated_text}")
            print(f"📏 Tokens generated: {len(output.outputs[0].token_ids)}")
            print("-" * 80)
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return
    
    # Demonstrate different sampling strategies
    print("\n🎨 Demonstrating Different Sampling Strategies")
    print("=" * 60)
    
    test_prompt = "The key to happiness is"
    
    strategies = [
        ("Conservative (low temp)", SamplingParams(temperature=0.2, max_tokens=50)),
        ("Balanced", SamplingParams(temperature=0.7, max_tokens=50)),
        ("Creative (high temp)", SamplingParams(temperature=1.2, max_tokens=50)),
        ("Deterministic", SamplingParams(temperature=0.0, max_tokens=50)),
    ]
    
    for strategy_name, params in strategies:
        try:
            output = llm.generate([test_prompt], params)[0]
            generated = output.outputs[0].text
            print(f"\n🎯 {strategy_name}:")
            print(f"   {test_prompt}{generated}")
        except Exception as e:
            print(f"❌ Error with {strategy_name}: {e}")
    
    print(f"\n✨ Basic inference example completed!")
    print("💡 Next steps:")
    print("   - Try different models")
    print("   - Experiment with sampling parameters")
    print("   - Run the API server example (02_api_server.py)")


def cleanup():
    """Clean up vLLM resources."""
    try:
        destroy_model_parallel()
    except:
        pass


if __name__ == "__main__":
    try:
        basic_inference_example()
    finally:
        cleanup()

