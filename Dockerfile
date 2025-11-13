# Dockerfile for Distributed LLM Fine-tuning Platform
# Multi-stage build for optimized image size

# Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY setup.py .
COPY README.md .

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p /app/outputs /app/data /app/checkpoints

# Expose ports
EXPOSE 8265  # Ray Dashboard
EXPOSE 5000  # MLflow
EXPOSE 8000  # Model Serving

# Default command
CMD ["bash"]

# ============================================================================
# Development image with additional tools
# ============================================================================
FROM base as dev

# Install development tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipython \
    black \
    flake8 \
    mypy \
    pytest \
    pytest-cov

EXPOSE 8888  # Jupyter

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

# ============================================================================
# Production image (minimal)
# ============================================================================
FROM base as prod

# Run as non-root user for security
RUN useradd -m -u 1000 llmuser && \
    chown -R llmuser:llmuser /app

USER llmuser

CMD ["llm-train", "--help"]
