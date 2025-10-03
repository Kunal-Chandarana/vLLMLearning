#!/usr/bin/env python3
"""
Comprehensive Benchmarking Example

This example demonstrates how to benchmark vLLM performance across different scenarios.
Features:
- Multiple model comparisons
- Batch size optimization
- Sampling strategy analysis
- Memory usage tracking
- Performance reporting
"""

import os
import sys
import time
import json
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_model_parallel

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.benchmark_utils import (
    benchmark_inference, benchmark_batch_sizes, benchmark_sampling_strategies,
    compare_sequential_vs_batch, generate_benchmark_report, save_benchmark_results,
    BenchmarkResult
)
from utils.model_utils import get_model_info, estimate_memory_requirements


class ComprehensiveBenchmark:
    """Comprehensive benchmarking suite for vLLM."""
    
    def __init__(self):
        self.results = []
        self.models = {}
        
    def initialize_models(self, model_names: List[str]):
        """Initialize multiple models for comparison."""
        print("🚀 Initializing Models for Benchmarking")
        print("=" * 60)
        
        for model_name in model_names:
            try:
                print(f"\n📦 Loading {model_name}...")
                
                # Get model info first
                info = get_model_info(model_name)
                if "error" in info:
                    print(f"❌ Could not get info for {model_name}: {info['error']}")
                    continue
                
                print(f"   Parameters: {info.get('parameters', 'unknown'):,}" if isinstance(info.get('parameters'), int) else f"   Parameters: {info.get('parameters', 'unknown')}")
                print(f"   Estimated memory: {info.get('estimated_memory_gb', 'unknown'):.1f} GB" if isinstance(info.get('estimated_memory_gb'), float) else f"   Estimated memory: {info.get('estimated_memory_gb', 'unknown')}")
                
                # Initialize model with conservative settings for benchmarking
                llm = LLM(
                    model=model_name,
                    tensor_parallel_size=1,
                    max_model_len=512,  # Conservative for benchmarking
                    gpu_memory_utilization=0.7,  # Leave room for multiple models
                    trust_remote_code=True,
                )
                
                self.models[model_name] = {
                    'llm': llm,
                    'info': info
                }
                
                print(f"✅ {model_name} loaded successfully")
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
    
    def benchmark_basic_inference(self):
        """Benchmark basic inference across all models."""
        print("\n🎯 Basic Inference Benchmark")
        print("=" * 60)
        
        test_prompts = [
            "The future of artificial intelligence is",
            "Climate change solutions include",
            "The most important skill for success is",
        ]
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=50,
            top_p=0.9,
        )
        
        for model_name, model_data in self.models.items():
            print(f"\n🔄 Testing {model_name}...")
            
            result = benchmark_inference(
                model_data['llm'], 
                test_prompts, 
                sampling_params,
                name=f"{model_name}_basic"
            )
            
            self.results.append(result)
            
            if result.success:
                print(f"   ✅ Throughput: {result.throughput:.1f} req/s")
                print(f"   ⚡ Tokens/sec: {result.tokens_per_second:.1f}")
                print(f"   💾 Memory: {result.memory_used_mb:.1f} MB")
            else:
                print(f"   ❌ Failed: {result.error}")
    
    def benchmark_batch_performance(self):
        """Benchmark batch processing performance."""
        print("\n📦 Batch Size Performance Benchmark")
        print("=" * 60)
        
        base_prompts = [
            "The benefits of renewable energy include",
            "Artificial intelligence can help solve",
            "The key to effective communication is",
        ]
        
        batch_sizes = [1, 2, 4, 8]  # Conservative batch sizes for comparison
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=40,
        )
        
        for model_name, model_data in self.models.items():
            print(f"\n🔄 Testing batch sizes for {model_name}...")
            
            batch_results = benchmark_batch_sizes(
                model_data['llm'],
                base_prompts,
                sampling_params,
                batch_sizes
            )
            
            self.results.extend(batch_results)
            
            # Show summary
            successful_results = [r for r in batch_results if r.success]
            if successful_results:
                best_throughput = max(successful_results, key=lambda r: r.throughput)
                print(f"   🏆 Best batch size: {best_throughput.name.split('_')[-1]} "
                      f"({best_throughput.throughput:.1f} req/s)")
            else:
                print(f"   ❌ All batch tests failed for {model_name}")
    
    def benchmark_sampling_strategies(self):
        """Benchmark different sampling strategies."""
        print("\n🎨 Sampling Strategy Benchmark")
        print("=" * 60)
        
        test_prompts = [
            "The relationship between creativity and innovation is",
            "The most effective approach to problem-solving involves",
        ]
        
        sampling_configs = [
            {"name": "conservative", "temperature": 0.2, "max_tokens": 50},
            {"name": "balanced", "temperature": 0.7, "max_tokens": 50},
            {"name": "creative", "temperature": 1.2, "max_tokens": 50},
        ]
        
        for model_name, model_data in self.models.items():
            print(f"\n🔄 Testing sampling strategies for {model_name}...")
            
            sampling_results = benchmark_sampling_strategies(
                model_data['llm'],
                test_prompts,
                sampling_configs.copy()  # Copy to avoid modifying original
            )
            
            # Add model name to results
            for result in sampling_results:
                result.name = f"{model_name}_{result.name}"
            
            self.results.extend(sampling_results)
            
            # Show summary
            successful_results = [r for r in sampling_results if r.success]
            if successful_results:
                fastest = min(successful_results, key=lambda r: r.duration)
                print(f"   ⚡ Fastest strategy: {fastest.name.split('_')[-1]} "
                      f"({fastest.duration:.2f}s)")
    
    def benchmark_sequential_vs_batch(self):
        """Compare sequential vs batch processing."""
        print("\n⚖️  Sequential vs Batch Comparison")
        print("=" * 60)
        
        test_prompts = [
            "The evolution of technology has led to",
            "The importance of sustainable development includes",
            "The future of work will be characterized by",
            "The role of education in society is to",
        ]
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=40,
        )
        
        for model_name, model_data in self.models.items():
            print(f"\n🔄 Comparing processing methods for {model_name}...")
            
            comparison_results = compare_sequential_vs_batch(
                model_data['llm'],
                test_prompts,
                sampling_params
            )
            
            # Add model name to results
            for method, result in comparison_results.items():
                result.name = f"{model_name}_{method}"
                self.results.append(result)
            
            # Show comparison
            if comparison_results["sequential"].success and comparison_results["batch"].success:
                seq_throughput = comparison_results["sequential"].throughput
                batch_throughput = comparison_results["batch"].throughput
                speedup = batch_throughput / seq_throughput
                print(f"   📊 Batch speedup: {speedup:.1f}x faster than sequential")
            else:
                print(f"   ❌ Comparison failed for {model_name}")
    
    def memory_usage_analysis(self):
        """Analyze memory usage patterns."""
        print("\n💾 Memory Usage Analysis")
        print("=" * 60)
        
        for model_name, model_data in self.models.items():
            info = model_data['info']
            
            print(f"\n📊 {model_name}:")
            print(f"   Model size: {info.get('estimated_memory_gb', 'unknown'):.1f} GB" if isinstance(info.get('estimated_memory_gb'), float) else f"   Model size: {info.get('estimated_memory_gb', 'unknown')}")
            
            # Find memory usage from benchmark results
            model_results = [r for r in self.results if model_name in r.name and r.success]
            if model_results:
                avg_memory = sum(r.memory_used_mb for r in model_results) / len(model_results)
                max_memory = max(r.memory_used_mb for r in model_results)
                print(f"   Average runtime memory: {avg_memory:.1f} MB")
                print(f"   Peak runtime memory: {max_memory:.1f} MB")
            
            # Memory recommendations
            memory_reqs = estimate_memory_requirements(model_name)
            if "error" not in memory_reqs:
                print(f"   Recommended GPU memory: {memory_reqs.get('recommended_gpu_memory_gb', 'unknown'):.1f} GB" if isinstance(memory_reqs.get('recommended_gpu_memory_gb'), float) else f"   Recommended GPU memory: {memory_reqs.get('recommended_gpu_memory_gb', 'unknown')}")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive benchmark report."""
        print("\n📋 Generating Comprehensive Report")
        print("=" * 60)
        
        # Generate standard report
        report = generate_benchmark_report(self.results)
        
        # Add model comparison section
        model_comparison = self.generate_model_comparison()
        full_report = report + "\n\n" + model_comparison
        
        # Save report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"benchmark_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(full_report)
        
        print(f"📄 Report saved to: {report_file}")
        
        # Save raw results
        results_file = f"benchmark_results_{timestamp}.json"
        save_benchmark_results(self.results, results_file)
        print(f"💾 Raw results saved to: {results_file}")
        
        return full_report
    
    def generate_model_comparison(self) -> str:
        """Generate model comparison section."""
        lines = []
        lines.append("🏆 Model Comparison Summary")
        lines.append("=" * 60)
        
        # Group results by model
        model_results = {}
        for result in self.results:
            if result.success:
                model_name = result.name.split('_')[0]  # Extract model name
                if model_name not in model_results:
                    model_results[model_name] = []
                model_results[model_name].append(result)
        
        # Calculate averages for each model
        lines.append(f"{'Model':<20} {'Avg Throughput':<15} {'Avg Tokens/s':<12} {'Tests':<8}")
        lines.append("-" * 60)
        
        for model_name, results in model_results.items():
            avg_throughput = sum(r.throughput for r in results) / len(results)
            avg_tokens_per_sec = sum(r.tokens_per_second for r in results) / len(results)
            test_count = len(results)
            
            lines.append(f"{model_name:<20} {avg_throughput:<15.1f} {avg_tokens_per_sec:<12.1f} {test_count:<8}")
        
        # Best performers
        if self.results:
            successful_results = [r for r in self.results if r.success]
            if successful_results:
                best_throughput = max(successful_results, key=lambda r: r.throughput)
                best_tokens = max(successful_results, key=lambda r: r.tokens_per_second)
                
                lines.append(f"\n🥇 Best Performers:")
                lines.append(f"   Highest throughput: {best_throughput.name} ({best_throughput.throughput:.1f} req/s)")
                lines.append(f"   Fastest token generation: {best_tokens.name} ({best_tokens.tokens_per_second:.1f} tok/s)")
        
        return "\n".join(lines)
    
    def cleanup(self):
        """Clean up resources."""
        try:
            destroy_model_parallel()
        except:
            pass


def main():
    """Main function to run comprehensive benchmarks."""
    print("🏁 vLLM Comprehensive Benchmarking Suite")
    print("=" * 60)
    
    # Models to benchmark (start with smaller ones)
    models_to_test = [
        "gpt2",
        "distilgpt2",
        # Add more models as needed
        # "gpt2-medium",  # Uncomment if you have sufficient GPU memory
    ]
    
    benchmark = ComprehensiveBenchmark()
    
    try:
        # Initialize models
        benchmark.initialize_models(models_to_test)
        
        if not benchmark.models:
            print("❌ No models loaded successfully. Exiting.")
            return
        
        # Run benchmark suite
        benchmark.benchmark_basic_inference()
        benchmark.benchmark_batch_performance()
        benchmark.benchmark_sampling_strategies()
        benchmark.benchmark_sequential_vs_batch()
        benchmark.memory_usage_analysis()
        
        # Generate report
        report = benchmark.generate_comprehensive_report()
        
        print("\n🎉 Comprehensive benchmarking completed!")
        print("\n💡 Key insights:")
        print("   - Compare model performance for your use case")
        print("   - Optimize batch sizes for your hardware")
        print("   - Choose sampling strategies based on quality vs speed")
        print("   - Consider memory requirements for deployment")
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main()

