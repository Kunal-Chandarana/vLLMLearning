#!/bin/bash

# vLLM Environment Setup Script
# This script sets up the complete environment for the vLLM learning project

set -e  # Exit on any error

echo "🚀 vLLM Learning Environment Setup"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
            print_success "Python $PYTHON_VERSION found (compatible)"
            PYTHON_CMD="python3"
        else
            print_error "Python $PYTHON_VERSION found but requires 3.9+"
            exit 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
            print_success "Python $PYTHON_VERSION found (compatible)"
            PYTHON_CMD="python"
        else
            print_error "Python $PYTHON_VERSION found but requires 3.9+"
            exit 1
        fi
    else
        print_error "Python not found. Please install Python 3.9+"
        exit 1
    fi
}

# Check if we're in a virtual environment
check_virtual_env() {
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_success "Virtual environment detected: $VIRTUAL_ENV"
        return 0
    else
        print_warning "No virtual environment detected"
        return 1
    fi
}

# Create virtual environment
create_virtual_env() {
    print_status "Creating virtual environment..."
    
    if [ -d "vllm_env" ]; then
        print_warning "Virtual environment 'vllm_env' already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf vllm_env
        else
            print_status "Using existing virtual environment"
            source vllm_env/bin/activate
            return 0
        fi
    fi
    
    $PYTHON_CMD -m venv vllm_env
    source vllm_env/bin/activate
    print_success "Virtual environment created and activated"
}

# Upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    $PYTHON_CMD -m pip install --upgrade pip
    print_success "Pip upgraded"
}

# Install PyTorch with CUDA support
install_pytorch() {
    print_status "Installing PyTorch with CUDA support..."
    
    # Check if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
        print_status "CUDA $CUDA_VERSION detected"
        
        # Install PyTorch with CUDA support
        if [[ "$CUDA_VERSION" == "12."* ]]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        elif [[ "$CUDA_VERSION" == "11."* ]]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            print_warning "Unsupported CUDA version, installing CPU version"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        print_warning "CUDA not detected, installing CPU version"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    print_success "PyTorch installed"
}

# Install requirements
install_requirements() {
    print_status "Installing project requirements..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Requirements installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    if [ -f "scripts/test_installation.py" ]; then
        $PYTHON_CMD scripts/test_installation.py
    else
        print_warning "Test script not found, skipping tests"
    fi
}

# Download a small model for testing
download_test_model() {
    print_status "Downloading test model..."
    
    read -p "Do you want to download a small test model (gpt2)? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_status "Skipping model download"
        return 0
    fi
    
    if [ -f "scripts/download_models.py" ]; then
        $PYTHON_CMD scripts/download_models.py --model gpt2
        print_success "Test model downloaded"
    else
        print_warning "Download script not found, skipping model download"
    fi
}

# Create activation script
create_activation_script() {
    print_status "Creating activation script..."
    
    cat > activate_vllm.sh << 'EOF'
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
EOF

    chmod +x activate_vllm.sh
    print_success "Activation script created (activate_vllm.sh)"
}

# Main setup function
main() {
    echo
    print_status "Starting vLLM environment setup..."
    echo
    
    # Check prerequisites
    check_python
    
    # Handle virtual environment
    if ! check_virtual_env; then
        create_virtual_env
    fi
    
    # Install dependencies
    upgrade_pip
    install_pytorch
    install_requirements
    
    # Test and finalize
    test_installation
    download_test_model
    create_activation_script
    
    echo
    print_success "🎉 Setup completed successfully!"
    echo
    echo "📋 Next steps:"
    echo "1. Activate the environment: source vllm_env/bin/activate"
    echo "2. Or use the convenience script: ./activate_vllm.sh"
    echo "3. Test the installation: python scripts/test_installation.py"
    echo "4. Run your first example: python examples/01_basic_inference.py"
    echo
    echo "📚 Check the README.md for detailed usage instructions"
    echo
}

# Run main function
main "$@"

