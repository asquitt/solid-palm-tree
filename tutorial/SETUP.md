# Environment Setup Guide

This guide will help you set up your development environment for the LLM Fine-tuning Platform tutorial.

## Quick Setup (Recommended)

```bash
cd tutorial
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

The script will handle everything automatically. If you prefer manual setup, continue reading.

---

## Manual Setup

### 1. System Requirements

**Minimum Requirements**:
- Python 3.9 or higher
- 8GB RAM
- 10GB free disk space
- Internet connection

**Recommended**:
- Python 3.10 or 3.11
- 16GB RAM
- GPU with CUDA support (for faster training)
- 20GB free disk space

### 2. Check Python Version

```bash
python --version
# Should show Python 3.9.x or higher

# If not, install Python 3.9+
# macOS: brew install python@3.11
# Ubuntu: sudo apt install python3.11
# Windows: Download from python.org
```

### 3. Create Virtual Environment

```bash
# Navigate to tutorial directory
cd solid-palm-tree/tutorial

# Create virtual environment
python -m venv tutorial_env

# Activate it
# macOS/Linux:
source tutorial_env/bin/activate

# Windows:
tutorial_env\Scripts\activate

# You should see (tutorial_env) in your prompt
```

### 4. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install tutorial dependencies
pip install -r requirements.txt

# This installs:
# - PyTorch (CPU version for compatibility)
# - Transformers (for pre-trained models)
# - Datasets (for data loading)
# - Ray (for distributed computing)
# - FastAPI (for serving)
# - Pytest (for testing)
# - And more...
```

### 5. Verify Installation

```bash
# Run the verification script
python scripts/verify_setup.py

# Expected output:
# ✅ Python version: 3.11.5
# ✅ PyTorch installed: 2.1.0
# ✅ Transformers installed: 4.35.0
# ✅ Ray installed: 2.8.0
# ✅ All dependencies OK!
```

---

## GPU Setup (Optional but Recommended)

### Check GPU Availability

```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

### Install PyTorch with CUDA (if you have NVIDIA GPU)

```bash
# Uninstall CPU version
pip uninstall torch

# Install CUDA version (example for CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Download Sample Data

```bash
# Download small datasets for exercises
chmod +x scripts/download_data.sh
./scripts/download_data.sh

# This downloads:
# - Small text datasets for training
# - Pre-tokenized samples
# - Evaluation datasets
# Total size: ~500MB
```

---

## IDE Setup (Recommended: VS Code)

### Install VS Code Extensions

1. **Python** (by Microsoft)
2. **Pylance** (for better code intelligence)
3. **Python Test Explorer** (for running tests)
4. **GitHub Copilot** (optional, helpful for learning)

### Configure VS Code

Create `.vscode/settings.json` in the tutorial folder:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/tutorial_env/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["-v"],
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true
}
```

---

## Troubleshooting

### Issue: "Python not found"

**Solution**: Make sure Python 3.9+ is installed and in your PATH

```bash
# macOS/Linux
which python3

# Windows
where python
```

### Issue: "Permission denied" when running scripts

**Solution**: Make scripts executable

```bash
chmod +x scripts/*.sh
```

### Issue: "pip install fails with SSL error"

**Solution**: Upgrade pip and try again

```bash
python -m pip install --upgrade pip
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Issue: "Out of memory" during training

**Solution**: Use smaller batch sizes in exercises

```python
# In training code, reduce batch_size
batch_size = 4  # Instead of 32
```

### Issue: "CUDA out of memory"

**Solution**: Reduce model size or batch size

```python
# Use smaller model
model_name = "gpt2"  # Instead of "gpt2-large"

# Or reduce batch size
batch_size = 2
```

---

## Dependency Details

### Core Dependencies

```
torch>=2.0.0              # Deep learning framework
transformers>=4.30.0      # Pre-trained models
datasets>=2.14.0          # Dataset loading
ray[default]>=2.6.0       # Distributed computing
fastapi>=0.100.0          # Web framework
```

### Testing Dependencies

```
pytest>=7.4.0             # Testing framework
pytest-asyncio>=0.21.0    # Async test support
pytest-cov>=4.1.0         # Coverage reporting
pytest-mock>=3.11.0       # Mocking utilities
```

### Development Dependencies

```
black>=23.7.0             # Code formatter
isort>=5.12.0            # Import sorter
flake8>=6.1.0            # Linter
mypy>=1.5.0              # Type checker
```

---

## Alternative: Docker Setup (Advanced)

If you prefer Docker:

```bash
# Build the tutorial image
docker build -t llm-tutorial -f tutorial/Dockerfile .

# Run interactive container
docker run -it --rm \
  -v $(pwd):/workspace \
  -p 8000:8000 \
  llm-tutorial bash

# Inside container, activate environment
source tutorial_env/bin/activate
```

---

## Next Steps

Once setup is complete:

1. ✅ Verify all dependencies are installed
2. ✅ Download sample data
3. ✅ Configure your IDE
4. 🚀 Start Week 1: `cd week1 && cat README.md`

---

## Getting Help

- Check `resources/troubleshooting.md` for common issues
- Verify your setup: `python scripts/verify_setup.py`
- Test your environment: `pytest tutorial/common/tests/ -v`

---

**Ready to start learning? Head to Week 1!**

```bash
cd week1
cat README.md
```
