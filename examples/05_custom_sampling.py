#!/usr/bin/env python3
"""
Custom Sampling Parameters Example

This example demonstrates various sampling strategies and their effects on text generation.
Features:
- Temperature variations
- Top-p (nucleus) sampling
- Top-k sampling
- Repetition penalties
- Custom stopping criteria
- Comparative analysis
"""

import time
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel


class SamplingStrategiesExample:
    """Demonstrates different sampling strategies in vLLM."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.llm = None
    
    def initialize_model(self):
        """Initialize the vLLM model."""
        print(f"🚀 Initializing model: {self.model_name}")
        
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=1,
            max_model_len=512,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
        )
        
        print("✅ Model initialized successfully!")
    
    def demonstrate_temperature_effects(self):
        """Show how temperature affects generation creativity."""
        print("\n🌡️  Temperature Effects Demonstration")
        print("=" * 60)
        
        prompt = "The secret to innovation is"
        temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]
        
        print(f"📝 Prompt: '{prompt}'")
        print("-" * 60)
        
        for temp in temperatures:
            sampling_params = SamplingParams(
                temperature=temp,
                max_tokens=50,
                top_p=1.0,  # Disable nucleus sampling to isolate temperature effect
            )
            
            output = self.llm.generate([prompt], sampling_params)[0]
            generated_text = output.outputs[0].text
            
            print(f"\n🌡️  Temperature {temp}:")
            print(f"   {prompt}{generated_text}")
            
            # Analyze characteristics
            if temp == 0.0:
                print("   📊 Analysis: Deterministic, consistent output")
            elif temp < 0.5:
                print("   📊 Analysis: Conservative, focused responses")
            elif temp < 1.0:
                print("   📊 Analysis: Balanced creativity and coherence")
            else:
                print("   📊 Analysis: High creativity, potentially less coherent")
    
    def demonstrate_nucleus_sampling(self):
        """Demonstrate top-p (nucleus) sampling effects."""
        print("\n🎯 Nucleus Sampling (Top-p) Demonstration")
        print("=" * 60)
        
        prompt = "The future of artificial intelligence includes"
        top_p_values = [0.1, 0.5, 0.9, 0.95, 1.0]
        
        print(f"📝 Prompt: '{prompt}'")
        print("-" * 60)
        
        for top_p in top_p_values:
            sampling_params = SamplingParams(
                temperature=0.8,  # Fixed temperature
                top_p=top_p,
                max_tokens=50,
            )
            
            output = self.llm.generate([prompt], sampling_params)[0]
            generated_text = output.outputs[0].text
            
            print(f"\n🎯 Top-p {top_p}:")
            print(f"   {prompt}{generated_text}")
            
            # Explain the effect
            if top_p <= 0.1:
                print("   📊 Analysis: Very focused, uses only most likely tokens")
            elif top_p <= 0.5:
                print("   📊 Analysis: Moderately focused, good coherence")
            elif top_p <= 0.9:
                print("   📊 Analysis: Balanced diversity and quality")
            else:
                print("   📊 Analysis: High diversity, includes less likely tokens")
    
    def demonstrate_top_k_sampling(self):
        """Demonstrate top-k sampling effects."""
        print("\n🔢 Top-k Sampling Demonstration")
        print("=" * 60)
        
        prompt = "The most important lesson in life is"
        top_k_values = [1, 5, 20, 50, 100]
        
        print(f"📝 Prompt: '{prompt}'")
        print("-" * 60)
        
        for top_k in top_k_values:
            sampling_params = SamplingParams(
                temperature=0.8,
                top_k=top_k,
                top_p=1.0,  # Disable nucleus sampling
                max_tokens=50,
            )
            
            output = self.llm.generate([prompt], sampling_params)[0]
            generated_text = output.outputs[0].text
            
            print(f"\n🔢 Top-k {top_k}:")
            print(f"   {prompt}{generated_text}")
    
    def demonstrate_repetition_penalty(self):
        """Show how repetition penalty affects text generation."""
        print("\n🔄 Repetition Penalty Demonstration")
        print("=" * 60)
        
        # Use a prompt that might lead to repetition
        prompt = "The benefits of exercise are numerous. Exercise helps with"
        penalties = [1.0, 1.1, 1.2, 1.5, 2.0]
        
        print(f"📝 Prompt: '{prompt}'")
        print("-" * 60)
        
        for penalty in penalties:
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.9,
                max_tokens=60,
                repetition_penalty=penalty,
            )
            
            output = self.llm.generate([prompt], sampling_params)[0]
            generated_text = output.outputs[0].text
            
            print(f"\n🔄 Repetition Penalty {penalty}:")
            print(f"   {prompt}{generated_text}")
            
            # Count repeated words as a simple metric
            words = generated_text.lower().split()
            unique_words = len(set(words))
            total_words = len(words)
            diversity = unique_words / total_words if total_words > 0 else 0
            
            print(f"   📊 Word diversity: {diversity:.2f} ({unique_words}/{total_words} unique)")
    
    def demonstrate_stop_sequences(self):
        """Demonstrate custom stop sequences."""
        print("\n🛑 Custom Stop Sequences Demonstration")
        print("=" * 60)
        
        prompt = "Here's a step-by-step guide:"
        
        stop_configs = [
            {"name": "No stops", "stop": None},
            {"name": "Stop at period", "stop": ["."]},
            {"name": "Stop at newline", "stop": ["\\n"]},
            {"name": "Multiple stops", "stop": [".", "!", "?", "\\n"]},
            {"name": "Custom phrase", "stop": ["In conclusion", "Finally"]},
        ]
        
        print(f"📝 Prompt: '{prompt}'")
        print("-" * 60)
        
        for config in stop_configs:
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=100,
                stop=config["stop"],
            )
            
            output = self.llm.generate([prompt], sampling_params)[0]
            generated_text = output.outputs[0].text
            
            print(f"\n🛑 {config['name']}:")
            print(f"   {prompt}{generated_text}")
            print(f"   📏 Length: {len(generated_text)} characters")
    
    def comparative_analysis(self):
        """Compare different sampling strategies on the same prompt."""
        print("\n📊 Comparative Analysis")
        print("=" * 60)
        
        prompt = "The key to building successful AI systems is"
        
        strategies = [
            {
                "name": "Conservative",
                "params": SamplingParams(temperature=0.2, top_p=0.8, max_tokens=50),
                "description": "Low temperature, focused sampling"
            },
            {
                "name": "Balanced",
                "params": SamplingParams(temperature=0.7, top_p=0.9, max_tokens=50),
                "description": "Moderate creativity and coherence"
            },
            {
                "name": "Creative",
                "params": SamplingParams(temperature=1.2, top_p=0.95, max_tokens=50),
                "description": "High creativity, diverse outputs"
            },
            {
                "name": "Deterministic",
                "params": SamplingParams(temperature=0.0, max_tokens=50),
                "description": "Completely predictable output"
            },
            {
                "name": "Top-k Limited",
                "params": SamplingParams(temperature=0.8, top_k=10, max_tokens=50),
                "description": "Limited vocabulary, focused choices"
            },
        ]
        
        print(f"📝 Prompt: '{prompt}'")
        print("-" * 60)
        
        results = []
        for strategy in strategies:
            start_time = time.time()
            output = self.llm.generate([prompt], strategy["params"])[0]
            generation_time = time.time() - start_time
            
            generated_text = output.outputs[0].text
            
            print(f"\n🎯 {strategy['name']} Strategy:")
            print(f"   📝 {strategy['description']}")
            print(f"   💬 {prompt}{generated_text}")
            print(f"   ⏱️  Generation time: {generation_time:.3f}s")
            
            results.append({
                "strategy": strategy["name"],
                "text": generated_text,
                "time": generation_time,
                "length": len(generated_text),
            })
        
        # Summary analysis
        print(f"\n📈 Summary Analysis:")
        print("-" * 40)
        avg_time = sum(r["time"] for r in results) / len(results)
        avg_length = sum(r["length"] for r in results) / len(results)
        
        print(f"Average generation time: {avg_time:.3f}s")
        print(f"Average text length: {avg_length:.1f} characters")
        
        fastest = min(results, key=lambda x: x["time"])
        longest = max(results, key=lambda x: x["length"])
        
        print(f"Fastest strategy: {fastest['strategy']} ({fastest['time']:.3f}s)")
        print(f"Most verbose: {longest['strategy']} ({longest['length']} chars)")
    
    def advanced_sampling_techniques(self):
        """Demonstrate advanced sampling techniques."""
        print("\n🔬 Advanced Sampling Techniques")
        print("=" * 60)
        
        prompt = "The relationship between creativity and artificial intelligence"
        
        techniques = [
            {
                "name": "Typical Sampling",
                "params": SamplingParams(
                    temperature=0.8,
                    top_p=0.9,
                    typical_p=0.95,  # If supported
                    max_tokens=60,
                ),
                "description": "Uses typical sampling for more natural text"
            },
            {
                "name": "Length Penalty",
                "params": SamplingParams(
                    temperature=0.7,
                    max_tokens=60,
                    length_penalty=1.2,  # Encourage longer sequences
                ),
                "description": "Applies length penalty to encourage longer outputs"
            },
            {
                "name": "Frequency Penalty",
                "params": SamplingParams(
                    temperature=0.8,
                    max_tokens=60,
                    frequency_penalty=0.5,  # Reduce frequency of repeated tokens
                ),
                "description": "Reduces frequency of repeated tokens"
            },
            {
                "name": "Presence Penalty",
                "params": SamplingParams(
                    temperature=0.8,
                    max_tokens=60,
                    presence_penalty=0.3,  # Encourage new topics
                ),
                "description": "Encourages introduction of new topics"
            },
        ]
        
        print(f"📝 Prompt: '{prompt}'")
        print("-" * 60)
        
        for technique in techniques:
            try:
                output = self.llm.generate([prompt], technique["params"])[0]
                generated_text = output.outputs[0].text
                
                print(f"\n🔬 {technique['name']}:")
                print(f"   📝 {technique['description']}")
                print(f"   💬 {prompt} {generated_text}")
                
            except Exception as e:
                print(f"\n🔬 {technique['name']}:")
                print(f"   ❌ Not supported or error: {e}")


def main():
    """Main function to run sampling examples."""
    print("🎯 vLLM Custom Sampling Parameters Example")
    print("=" * 60)
    
    example = SamplingStrategiesExample()
    
    try:
        example.initialize_model()
        
        # Run all demonstrations
        example.demonstrate_temperature_effects()
        example.demonstrate_nucleus_sampling()
        example.demonstrate_top_k_sampling()
        example.demonstrate_repetition_penalty()
        example.demonstrate_stop_sequences()
        example.comparative_analysis()
        example.advanced_sampling_techniques()
        
        print("\n🎉 All sampling demonstrations completed!")
        print("\n💡 Key insights:")
        print("   🌡️  Temperature controls randomness (0.0 = deterministic, 1.0+ = creative)")
        print("   🎯 Top-p focuses on most probable tokens (0.9 is often optimal)")
        print("   🔢 Top-k limits vocabulary size (20-100 works well)")
        print("   🔄 Repetition penalty reduces repetitive text (1.1-1.2 recommended)")
        print("   🛑 Stop sequences provide precise control over generation length")
        print("   ⚖️  Balance creativity vs coherence based on your use case")
        
    except Exception as e:
        print(f"❌ Error during sampling demonstration: {e}")
    
    finally:
        try:
            destroy_model_parallel()
        except:
            pass


if __name__ == "__main__":
    main()

