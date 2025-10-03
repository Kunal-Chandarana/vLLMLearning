# vLLM Learning Project

A comprehensive sample project to learn and experiment with vLLM (Very Large Language Model) inference and serving.

## What is vLLM?

vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It features:
- **Fast inference**: Up to 24x higher throughput than HuggingFace Transformers
- **Efficient memory management**: PagedAttention for dynamic attention key-value memory management
- **Continuous batching**: Process requests as they come without waiting for batches to fill
- **OpenAI-compatible API**: Drop-in replacement for OpenAI API
- **Multi-GPU support**: Distributed inference across multiple GPUs

## Project Structure

```
vLLMLearning/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                 # Setup script for easy installation
├── examples/                # Learning examples
│   ├── 01_basic_inference.py      # Basic vLLM usage
│   ├── 02_api_server.py           # OpenAI-compatible API server
│   ├── 03_streaming_example.py    # Streaming responses
│   ├── 04_batch_inference.py      # Batch processing
│   ├── 05_custom_sampling.py      # Custom sampling parameters
│   ├── 06_multi_model.py          # Multiple model serving
│   └── 07_benchmarking.py         # Performance benchmarking
├── notebooks/               # Jupyter notebooks for interactive learning
│   ├── vLLM_Basics.ipynb         # Interactive basics tutorial
│   └── Performance_Analysis.ipynb # Performance comparison
├── scripts/                 # Utility scripts
│   ├── setup_environment.sh      # Environment setup
│   ├── download_models.py        # Model downloading utility
│   └── test_installation.py      # Installation verification
├── configs/                 # Configuration files
│   ├── model_configs.yaml        # Model configurations
│   └── server_configs.yaml       # Server configurations
└── utils/                   # Helper utilities
    ├── __init__.py
    ├── model_utils.py            # Model loading utilities
    └── benchmark_utils.py        # Benchmarking utilities
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv vllm_env
source vllm_env/bin/activate  # On Windows: vllm_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python scripts/test_installation.py
```

### 3. Download a Model (Optional)

```bash
python scripts/download_models.py --model microsoft/DialoGPT-medium
```

### 4. Run Basic Example

```bash
python examples/01_basic_inference.py
```

### 5. Start API Server

```bash
python examples/02_api_server.py
```

## Learning Path

1. **Start with basics**: `examples/01_basic_inference.py`
2. **API serving**: `examples/02_api_server.py`
3. **Streaming**: `examples/03_streaming_example.py`
4. **Batch processing**: `examples/04_batch_inference.py`
5. **Advanced features**: Explore remaining examples
6. **Interactive learning**: Use Jupyter notebooks in `notebooks/`

## Hardware Requirements

- **Minimum**: 8GB GPU memory (for small models like GPT-2)
- **Recommended**: 16GB+ GPU memory (for larger models)
- **For production**: Multiple GPUs with 24GB+ each

## Supported Models

vLLM supports many popular model architectures:
- GPT-2, GPT-3.5, GPT-4
- LLaMA, LLaMA 2, Code Llama
- Mistral, Mixtral
- Falcon
- And many more!

## Common Issues & Solutions

### Installation Issues
- Ensure CUDA is properly installed
- Use Python 3.9+ 
- Install PyTorch with CUDA support first

### Memory Issues
- Reduce `max_model_len` parameter
- Use smaller models for testing
- Enable tensor parallelism for multi-GPU setups

### Performance Issues
- Adjust `max_num_seqs` for your hardware
- Use appropriate `tensor_parallel_size`
- Monitor GPU utilization

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Paper: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

## Contributing

Feel free to add more examples, fix bugs, or improve documentation!

## License

This project is for educational purposes. Please respect the licenses of the models you use.

