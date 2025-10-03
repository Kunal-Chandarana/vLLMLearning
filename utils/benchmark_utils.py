"""
Benchmarking Utilities

Tools for measuring and comparing vLLM performance.
"""

import time
import psutil
import os
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    duration: float
    throughput: float
    memory_used_mb: float
    tokens_generated: int
    tokens_per_second: float
    prompts_processed: int
    average_latency: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PerformanceMonitor:
    """Monitor system performance during benchmarks."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.process = psutil.Process(os.getpid())
    
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return metrics."""
        if self.start_time is None:
            raise RuntimeError("Monitor not started")
        
        duration = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_used = current_memory - self.start_memory
        
        return {
            "duration": duration,
            "memory_used_mb": memory_used,
            "peak_memory_mb": current_memory
        }


@contextmanager
def performance_monitor():
    """Context manager for performance monitoring."""
    monitor = PerformanceMonitor()
    monitor.start()
    try:
        yield monitor
    finally:
        pass


def benchmark_inference(llm, prompts: List[str], sampling_params, name: str = "inference") -> BenchmarkResult:
    """Benchmark basic inference performance."""
    try:
        with performance_monitor() as monitor:
            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params)
            duration = time.time() - start_time
        
        # Calculate metrics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = len(prompts) / duration
        tokens_per_second = total_tokens / duration
        average_latency = duration / len(prompts)
        
        perf_metrics = monitor.stop()
        
        return BenchmarkResult(
            name=name,
            duration=duration,
            throughput=throughput,
            memory_used_mb=perf_metrics["memory_used_mb"],
            tokens_generated=total_tokens,
            tokens_per_second=tokens_per_second,
            prompts_processed=len(prompts),
            average_latency=average_latency,
            success=True,
            metadata={
                "peak_memory_mb": perf_metrics["peak_memory_mb"],
                "prompts_count": len(prompts),
                "sampling_params": asdict(sampling_params) if hasattr(sampling_params, '__dict__') else str(sampling_params)
            }
        )
        
    except Exception as e:
        return BenchmarkResult(
            name=name,
            duration=0,
            throughput=0,
            memory_used_mb=0,
            tokens_generated=0,
            tokens_per_second=0,
            prompts_processed=0,
            average_latency=0,
            success=False,
            error=str(e)
        )


def benchmark_batch_sizes(llm, base_prompts: List[str], sampling_params, 
                         batch_sizes: List[int]) -> List[BenchmarkResult]:
    """Benchmark different batch sizes."""
    results = []
    
    for batch_size in batch_sizes:
        # Create prompts for this batch size
        prompts = base_prompts[:batch_size] if len(base_prompts) >= batch_size else base_prompts * ((batch_size // len(base_prompts)) + 1)
        prompts = prompts[:batch_size]
        
        result = benchmark_inference(
            llm, prompts, sampling_params, 
            name=f"batch_size_{batch_size}"
        )
        results.append(result)
    
    return results


def benchmark_sampling_strategies(llm, prompts: List[str], 
                                sampling_configs: List[Dict[str, Any]]) -> List[BenchmarkResult]:
    """Benchmark different sampling strategies."""
    from vllm import SamplingParams
    
    results = []
    
    for config in sampling_configs:
        name = config.pop("name", "unknown")
        sampling_params = SamplingParams(**config)
        
        result = benchmark_inference(
            llm, prompts, sampling_params, name=name
        )
        results.append(result)
    
    return results


def compare_sequential_vs_batch(llm, prompts: List[str], sampling_params) -> Dict[str, BenchmarkResult]:
    """Compare sequential vs batch processing."""
    results = {}
    
    # Sequential processing
    try:
        with performance_monitor() as monitor:
            start_time = time.time()
            sequential_outputs = []
            for prompt in prompts:
                output = llm.generate([prompt], sampling_params)[0]
                sequential_outputs.append(output)
            sequential_duration = time.time() - start_time
        
        sequential_tokens = sum(len(output.outputs[0].token_ids) for output in sequential_outputs)
        sequential_perf = monitor.stop()
        
        results["sequential"] = BenchmarkResult(
            name="sequential",
            duration=sequential_duration,
            throughput=len(prompts) / sequential_duration,
            memory_used_mb=sequential_perf["memory_used_mb"],
            tokens_generated=sequential_tokens,
            tokens_per_second=sequential_tokens / sequential_duration,
            prompts_processed=len(prompts),
            average_latency=sequential_duration / len(prompts),
            success=True
        )
        
    except Exception as e:
        results["sequential"] = BenchmarkResult(
            name="sequential",
            duration=0, throughput=0, memory_used_mb=0,
            tokens_generated=0, tokens_per_second=0,
            prompts_processed=0, average_latency=0,
            success=False, error=str(e)
        )
    
    # Batch processing
    results["batch"] = benchmark_inference(llm, prompts, sampling_params, "batch")
    
    return results


def generate_benchmark_report(results: List[BenchmarkResult], 
                            output_file: Optional[str] = None) -> str:
    """Generate a formatted benchmark report."""
    report_lines = []
    report_lines.append("🏁 vLLM Benchmark Report")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Summary table
    report_lines.append("📊 Performance Summary")
    report_lines.append("-" * 60)
    report_lines.append(f"{'Benchmark':<20} {'Success':<8} {'Duration(s)':<12} {'Throughput':<12} {'Tokens/s':<10}")
    report_lines.append("-" * 60)
    
    for result in results:
        status = "✅" if result.success else "❌"
        report_lines.append(
            f"{result.name:<20} {status:<8} {result.duration:<12.2f} "
            f"{result.throughput:<12.1f} {result.tokens_per_second:<10.1f}"
        )
    
    report_lines.append("")
    
    # Detailed results
    report_lines.append("📋 Detailed Results")
    report_lines.append("-" * 60)
    
    for result in results:
        report_lines.append(f"\n🎯 {result.name}")
        if result.success:
            report_lines.append(f"   Duration: {result.duration:.3f}s")
            report_lines.append(f"   Throughput: {result.throughput:.1f} requests/s")
            report_lines.append(f"   Tokens/second: {result.tokens_per_second:.1f}")
            report_lines.append(f"   Average latency: {result.average_latency:.3f}s")
            report_lines.append(f"   Memory used: {result.memory_used_mb:.1f} MB")
            report_lines.append(f"   Prompts processed: {result.prompts_processed}")
            report_lines.append(f"   Tokens generated: {result.tokens_generated}")
        else:
            report_lines.append(f"   ❌ Failed: {result.error}")
    
    # Find best performer
    successful_results = [r for r in results if r.success]
    if successful_results:
        best_throughput = max(successful_results, key=lambda r: r.throughput)
        best_tokens_per_sec = max(successful_results, key=lambda r: r.tokens_per_second)
        
        report_lines.append(f"\n🏆 Best Performance")
        report_lines.append(f"   Highest throughput: {best_throughput.name} ({best_throughput.throughput:.1f} req/s)")
        report_lines.append(f"   Fastest token generation: {best_tokens_per_sec.name} ({best_tokens_per_sec.tokens_per_second:.1f} tok/s)")
    
    report_text = "\n".join(report_lines)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        report_lines.append(f"\n💾 Report saved to: {output_file}")
    
    return report_text


def save_benchmark_results(results: List[BenchmarkResult], filename: str):
    """Save benchmark results to JSON file."""
    data = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "results": [asdict(result) for result in results]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_benchmark_results(filename: str) -> List[BenchmarkResult]:
    """Load benchmark results from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    results = []
    for result_data in data["results"]:
        results.append(BenchmarkResult(**result_data))
    
    return results


def create_performance_test_suite(model_name: str) -> Dict[str, Any]:
    """Create a comprehensive performance test suite for a model."""
    return {
        "model_name": model_name,
        "test_prompts": [
            "The future of artificial intelligence is",
            "Climate change solutions include",
            "The most important skill for success is",
            "Innovation in technology will lead to",
            "The key to effective communication is",
        ],
        "batch_sizes": [1, 2, 4, 8, 16],
        "sampling_configs": [
            {"name": "conservative", "temperature": 0.2, "max_tokens": 50},
            {"name": "balanced", "temperature": 0.7, "max_tokens": 50},
            {"name": "creative", "temperature": 1.2, "max_tokens": 50},
            {"name": "deterministic", "temperature": 0.0, "max_tokens": 50},
        ],
        "test_types": [
            "basic_inference",
            "batch_comparison",
            "sampling_strategies",
            "sequential_vs_batch"
        ]
    }

