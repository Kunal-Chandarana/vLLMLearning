#!/usr/bin/env python3
"""
Batch Inference Example

This example demonstrates efficient batch processing with vLLM.
Features:
- Processing multiple prompts simultaneously
- Batch size optimization
- Performance comparison with sequential processing
- Memory usage monitoring
"""

import time
import psutil
import os
from typing import List, Tuple
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_model_parallel


class BatchInferenceExample:
    """Demonstrates batch processing capabilities of vLLM."""
    
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
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def generate_test_prompts(self, count: int) -> List[str]:
        """Generate a list of test prompts."""
        base_prompts = [
            "The future of technology is",
            "Climate change solutions include",
            "The most important skill for success is",
            "Artificial intelligence will help us",
            "In the next decade, we will see",
            "The key to happiness is",
            "Education should focus on",
            "The biggest challenge facing humanity is",
            "Innovation comes from",
            "The best way to learn is",
        ]
        
        # Repeat and vary prompts to reach desired count
        prompts = []
        for i in range(count):
            base_prompt = base_prompts[i % len(base_prompts)]
            # Add variation to make each prompt unique
            variation = f" (scenario {i+1})"
            prompts.append(base_prompt + variation)
        
        return prompts
    
    def sequential_processing(self, prompts: List[str], sampling_params: SamplingParams) -> Tuple[List[str], float]:
        """Process prompts one by one (sequential)."""
        print(f"🐌 Sequential processing of {len(prompts)} prompts...")
        
        start_time = time.time()
        results = []
        
        for i, prompt in enumerate(prompts):
            if i % 5 == 0:  # Progress indicator
                print(f"   Processing prompt {i+1}/{len(prompts)}")
            
            output = self.llm.generate([prompt], sampling_params)[0]
            results.append(output.outputs[0].text)
        
        duration = time.time() - start_time
        print(f"✅ Sequential processing completed in {duration:.2f}s")
        
        return results, duration
    
    def batch_processing(self, prompts: List[str], sampling_params: SamplingParams) -> Tuple[List[str], float]:
        """Process all prompts in a single batch."""
        print(f"🚀 Batch processing of {len(prompts)} prompts...")
        
        start_time = time.time()
        
        # Process all prompts at once
        outputs = self.llm.generate(prompts, sampling_params)
        results = [output.outputs[0].text for output in outputs]
        
        duration = time.time() - start_time
        print(f"✅ Batch processing completed in {duration:.2f}s")
        
        return results, duration
    
    def chunked_batch_processing(self, prompts: List[str], sampling_params: SamplingParams, 
                                chunk_size: int = 5) -> Tuple[List[str], float]:
        """Process prompts in smaller batches (chunks)."""
        print(f"📦 Chunked batch processing ({chunk_size} prompts per chunk)...")
        
        start_time = time.time()
        results = []
        
        # Process in chunks
        for i in range(0, len(prompts), chunk_size):
            chunk = prompts[i:i + chunk_size]
            print(f"   Processing chunk {i//chunk_size + 1}/{(len(prompts) + chunk_size - 1)//chunk_size}")
            
            outputs = self.llm.generate(chunk, sampling_params)
            chunk_results = [output.outputs[0].text for output in outputs]
            results.extend(chunk_results)
        
        duration = time.time() - start_time
        print(f"✅ Chunked processing completed in {duration:.2f}s")
        
        return results, duration
    
    def compare_processing_methods(self, prompt_count: int = 20):
        """Compare different processing methods."""
        print(f"\n📊 Comparing Processing Methods ({prompt_count} prompts)")
        print("=" * 70)
        
        # Generate test prompts
        prompts = self.generate_test_prompts(prompt_count)
        
        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=50,
            top_p=0.9,
        )
        
        # Track memory usage
        initial_memory = self.get_memory_usage()
        print(f"📈 Initial memory usage: {initial_memory:.1f} MB")
        
        results = {}
        
        # Method 1: Sequential processing
        try:
            seq_results, seq_time = self.sequential_processing(prompts, sampling_params)
            seq_memory = self.get_memory_usage()
            results['sequential'] = {
                'time': seq_time,
                'memory': seq_memory - initial_memory,
                'throughput': len(prompts) / seq_time
            }
        except Exception as e:
            print(f"❌ Sequential processing failed: {e}")
            results['sequential'] = None
        
        # Method 2: Full batch processing
        try:
            batch_results, batch_time = self.batch_processing(prompts, sampling_params)
            batch_memory = self.get_memory_usage()
            results['batch'] = {
                'time': batch_time,
                'memory': batch_memory - initial_memory,
                'throughput': len(prompts) / batch_time
            }
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            results['batch'] = None
        
        # Method 3: Chunked batch processing
        try:
            chunk_results, chunk_time = self.chunked_batch_processing(prompts, sampling_params, chunk_size=5)
            chunk_memory = self.get_memory_usage()
            results['chunked'] = {
                'time': chunk_time,
                'memory': chunk_memory - initial_memory,
                'throughput': len(prompts) / chunk_time
            }
        except Exception as e:
            print(f"❌ Chunked processing failed: {e}")
            results['chunked'] = None
        
        # Display comparison results
        self.display_comparison_results(results, prompt_count)
    
    def display_comparison_results(self, results: dict, prompt_count: int):
        """Display performance comparison results."""
        print(f"\n📈 Performance Comparison Results")
        print("=" * 70)
        print(f"{'Method':<15} {'Time (s)':<10} {'Memory (MB)':<12} {'Throughput (req/s)':<18}")
        print("-" * 70)
        
        for method, data in results.items():
            if data is not None:
                print(f"{method.capitalize():<15} {data['time']:<10.2f} {data['memory']:<12.1f} {data['throughput']:<18.1f}")
            else:
                print(f"{method.capitalize():<15} {'FAILED':<10} {'N/A':<12} {'N/A':<18}")
        
        # Find best method
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_method = min(valid_results.keys(), key=lambda k: valid_results[k]['time'])
            best_time = valid_results[best_method]['time']
            
            print(f"\n🏆 Best performing method: {best_method.upper()}")
            print(f"   Time: {best_time:.2f}s")
            print(f"   Throughput: {valid_results[best_method]['throughput']:.1f} requests/second")
            
            # Calculate speedup compared to sequential
            if 'sequential' in valid_results and best_method != 'sequential':
                speedup = valid_results['sequential']['time'] / best_time
                print(f"   Speedup vs sequential: {speedup:.1f}x")
    
    def demonstrate_batch_sizes(self):
        """Demonstrate the impact of different batch sizes."""
        print(f"\n🔬 Batch Size Impact Analysis")
        print("=" * 50)
        
        batch_sizes = [1, 3, 5, 10, 15, 20]
        prompts = self.generate_test_prompts(20)
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=30,
        )
        
        print(f"{'Batch Size':<12} {'Time (s)':<10} {'Throughput':<12} {'Memory (MB)':<12}")
        print("-" * 50)
        
        for batch_size in batch_sizes:
            try:
                start_time = time.time()
                initial_memory = self.get_memory_usage()
                
                # Process in batches of specified size
                results = []
                for i in range(0, len(prompts), batch_size):
                    chunk = prompts[i:i + batch_size]
                    outputs = self.llm.generate(chunk, sampling_params)
                    results.extend([output.outputs[0].text for output in outputs])
                
                duration = time.time() - start_time
                final_memory = self.get_memory_usage()
                throughput = len(prompts) / duration
                memory_used = final_memory - initial_memory
                
                print(f"{batch_size:<12} {duration:<10.2f} {throughput:<12.1f} {memory_used:<12.1f}")
                
            except Exception as e:
                print(f"{batch_size:<12} {'ERROR':<10} {str(e)[:20]:<12}")


def main():
    """Main function to run batch inference examples."""
    print("📦 vLLM Batch Inference Example")
    print("=" * 50)
    
    # Initialize example
    example = BatchInferenceExample()
    
    try:
        example.initialize_model()
        
        # Run demonstrations
        example.compare_processing_methods(prompt_count=15)
        example.demonstrate_batch_sizes()
        
        print("\n🎉 Batch inference examples completed!")
        print("\n💡 Key insights:")
        print("   - Batch processing is typically much faster than sequential")
        print("   - Optimal batch size depends on your hardware and model")
        print("   - Memory usage scales with batch size")
        print("   - vLLM's continuous batching optimizes throughput automatically")
        
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   - Reduce batch size if running out of memory")
        print("   - Try a smaller model if resources are limited")
        print("   - Check GPU memory availability")
    
    finally:
        # Cleanup
        try:
            destroy_model_parallel()
        except:
            pass


if __name__ == "__main__":
    main()

