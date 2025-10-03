#!/usr/bin/env python3
"""
Installation Test Script

This script verifies that vLLM and all dependencies are properly installed.
It performs comprehensive checks including:
- Python version compatibility
- Package installations
- CUDA availability
- Basic functionality tests
"""

import sys
import subprocess
import importlib
import platform
from typing import List, Tuple, Dict, Any


class InstallationTester:
    """Tests vLLM installation and dependencies."""
    
    def __init__(self):
        self.results = []
        self.errors = []
    
    def log_result(self, test_name: str, status: str, details: str = ""):
        """Log a test result."""
        self.results.append({
            "test": test_name,
            "status": status,
            "details": details
        })
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   {details}")
    
    def test_python_version(self):
        """Test Python version compatibility."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major == 3 and version.minor >= 9:
            self.log_result("Python Version", "PASS", f"Python {version_str} (compatible)")
        else:
            self.log_result("Python Version", "FAIL", f"Python {version_str} (requires 3.9+)")
    
    def test_package_installation(self, package_name: str, import_name: str = None):
        """Test if a package is installed and importable."""
        import_name = import_name or package_name
        
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            self.log_result(f"Package: {package_name}", "PASS", f"Version {version}")
            return True
        except ImportError as e:
            self.log_result(f"Package: {package_name}", "FAIL", f"Import error: {e}")
            return False
    
    def test_cuda_availability(self):
        """Test CUDA availability."""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                cuda_version = torch.version.cuda
                self.log_result("CUDA", "PASS", f"{device_count} GPU(s), {device_name}, CUDA {cuda_version}")
                return True
            else:
                self.log_result("CUDA", "WARN", "CUDA not available (CPU-only mode)")
                return False
        except Exception as e:
            self.log_result("CUDA", "FAIL", f"Error checking CUDA: {e}")
            return False
    
    def test_vllm_basic_functionality(self):
        """Test basic vLLM functionality."""
        try:
            from vllm import LLM, SamplingParams
            
            # Try to create a minimal LLM instance (this might fail on systems without GPU)
            try:
                # Use a very small model for testing
                llm = LLM(
                    model="gpt2",
                    tensor_parallel_size=1,
                    max_model_len=128,
                    gpu_memory_utilization=0.3,
                    trust_remote_code=True,
                )
                
                # Test basic generation
                sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
                outputs = llm.generate(["Hello"], sampling_params)
                
                if outputs and len(outputs) > 0:
                    self.log_result("vLLM Basic Functionality", "PASS", "Successfully generated text")
                    return True
                else:
                    self.log_result("vLLM Basic Functionality", "FAIL", "No output generated")
                    return False
                    
            except Exception as e:
                # If GPU test fails, try CPU mode or smaller configuration
                error_msg = str(e)
                if "CUDA" in error_msg or "GPU" in error_msg:
                    self.log_result("vLLM Basic Functionality", "WARN", 
                                  f"GPU test failed ({error_msg[:50]}...), but vLLM is installed")
                    return True
                else:
                    self.log_result("vLLM Basic Functionality", "FAIL", f"Error: {error_msg[:100]}...")
                    return False
                    
        except ImportError as e:
            self.log_result("vLLM Basic Functionality", "FAIL", f"Import error: {e}")
            return False
    
    def test_system_requirements(self):
        """Test system requirements."""
        # Check available memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if memory_gb >= 8:
                self.log_result("System Memory", "PASS", f"{memory_gb:.1f} GB available")
            else:
                self.log_result("System Memory", "WARN", f"{memory_gb:.1f} GB (8GB+ recommended)")
        except ImportError:
            self.log_result("System Memory", "WARN", "psutil not available, cannot check memory")
        
        # Check platform
        system = platform.system()
        self.log_result("Operating System", "INFO", f"{system} {platform.release()}")
    
    def test_optional_dependencies(self):
        """Test optional dependencies."""
        optional_packages = [
            ("fastapi", "fastapi"),
            ("uvicorn", "uvicorn"),
            ("requests", "requests"),
            ("jupyter", "jupyter"),
            ("matplotlib", "matplotlib"),
            ("gradio", "gradio"),
            ("streamlit", "streamlit"),
        ]
        
        for package_name, import_name in optional_packages:
            self.test_package_installation(package_name, import_name)
    
    def run_comprehensive_test(self):
        """Run all tests."""
        print("🧪 vLLM Installation Test")
        print("=" * 50)
        
        print("\n📋 Basic Requirements:")
        print("-" * 30)
        self.test_python_version()
        self.test_system_requirements()
        
        print("\n📦 Core Dependencies:")
        print("-" * 30)
        core_packages = [
            ("vllm", "vllm"),
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("tokenizers", "tokenizers"),
        ]
        
        for package_name, import_name in core_packages:
            self.test_package_installation(package_name, import_name)
        
        print("\n🎮 Hardware Support:")
        print("-" * 30)
        self.test_cuda_availability()
        
        print("\n🚀 Functionality Test:")
        print("-" * 30)
        self.test_vllm_basic_functionality()
        
        print("\n📚 Optional Dependencies:")
        print("-" * 30)
        self.test_optional_dependencies()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print("=" * 50)
        
        pass_count = sum(1 for r in self.results if r["status"] == "PASS")
        fail_count = sum(1 for r in self.results if r["status"] == "FAIL")
        warn_count = sum(1 for r in self.results if r["status"] == "WARN")
        total_count = len(self.results)
        
        print(f"✅ Passed: {pass_count}")
        print(f"❌ Failed: {fail_count}")
        print(f"⚠️  Warnings: {warn_count}")
        print(f"📊 Total: {total_count}")
        
        if fail_count == 0:
            print("\n🎉 Installation appears to be working correctly!")
            print("💡 You can now run the examples:")
            print("   python examples/01_basic_inference.py")
            print("   python examples/02_api_server.py")
        else:
            print("\n⚠️  Some issues were found. Please address the failures above.")
            print("💡 Common solutions:")
            print("   - Install missing packages: pip install -r requirements.txt")
            print("   - Check CUDA installation if using GPU")
            print("   - Ensure sufficient system memory (8GB+ recommended)")
        
        if warn_count > 0:
            print(f"\n⚠️  {warn_count} warning(s) found - these may not prevent basic functionality")


def main():
    """Main function to run installation tests."""
    tester = InstallationTester()
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()

