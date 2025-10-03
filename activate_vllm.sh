#!/bin/bash
# vLLM Environment Activation Script

echo "🚀 Activating vLLM Learning Environment"

# Activate virtual environment
if [ -f "vllm_env/bin/activate" ]; then
    source vllm_env/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run setup_environment.sh first."
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export VLLM_LEARNING_PATH="$(pwd)"

echo "📚 vLLM Learning Environment Ready!"
echo "💡 Try these commands:"
echo "   python examples/01_basic_inference.py"
echo "   python examples/02_api_server.py"
echo "   python scripts/test_installation.py"

# Start bash with custom prompt
export PS1="(vLLM) \u@\h:\w$ "
bash --rcfile <(echo "PS1='(vLLM) \u@\h:\w$ '")
