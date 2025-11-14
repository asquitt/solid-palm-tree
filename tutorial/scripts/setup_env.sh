#!/bin/bash

# Setup script for the LLM Fine-tuning Tutorial

echo "=========================================="
echo "LLM Fine-tuning Tutorial - Setup"
echo "=========================================="

# Check Python version
echo ""
echo "[1/5] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python $python_version found (>= 3.9)"
else
    echo "✗ Python >= 3.9 required. Found: $python_version"
    exit 1
fi

# Create virtual environment
echo ""
echo "[2/5] Creating virtual environment..."
if [ ! -d "tutorial_env" ]; then
    python -m venv tutorial_env
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[3/5] Activating virtual environment..."
source tutorial_env/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "[4/5] Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"

# Install dependencies
echo ""
echo "[5/5] Installing dependencies..."
echo "This may take 5-10 minutes..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import torch; print('✓ PyTorch:', torch.__version__)"
python -c "import transformers; print('✓ Transformers:', transformers.__version__)"
python -c "import datasets; print('✓ Datasets:', datasets.__version__)"
python -c "import ray; print('✓ Ray:', ray.__version__)"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future:"
echo "  source tutorial_env/bin/activate"
echo ""
echo "To get started:"
echo "  cd week1"
echo "  cat README.md"
echo ""
echo "Happy learning! 🚀"
