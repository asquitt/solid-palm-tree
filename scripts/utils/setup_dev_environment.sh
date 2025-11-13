#!/bin/bash
# Setup development environment

set -e

echo "Setting up development environment..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Create directories
mkdir -p outputs data checkpoints logs

# Download sample dataset
echo "Downloading sample dataset..."
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-2-raw-v1')"

echo "✓ Development environment ready!"
echo ""
echo "Activate environment: source venv/bin/activate"
echo "Run tests: pytest tests/"
echo "Start training: python scripts/examples/quick_start.py"
