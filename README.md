# 🚀 Distributed LLM Fine-tuning & Evaluation Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Ray 2.0+](https://img.shields.io/badge/ray-2.0+-orange.svg)](https://www.ray.io/)

A production-ready, cost-optimized platform for distributed fine-tuning and evaluation of Large Language Models (LLMs) using Ray, Kubernetes, and modern MLOps tools.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Cost Optimization](#cost-optimization)
- [Documentation](#documentation)
- [Testing](#testing)
- [Contributing](#contributing)
- [Learning Resources](#learning-resources)

## 🎯 Overview

This platform demonstrates production-grade distributed machine learning for LLM fine-tuning, showcasing:

- **Distributed Training**: Data-parallel fine-tuning across multiple GPU nodes using Ray Train
- **Hyperparameter Optimization**: Automated search at scale with Ray Tune
- **Cost Efficiency**: Spot instance management and auto-scaling to minimize cloud costs
- **Production Deployment**: Ray Serve with A/B testing for model serving
- **Experiment Tracking**: Comprehensive MLflow integration with custom LLM metrics
- **Data Management**: Feast feature store and DVC for reproducible pipelines

## ✨ Key Features

### Training & Optimization
- 🔥 **Ray Train Integration**: Data-parallel fine-tuning of Llama-3, GPT-2, or custom models
- 🎛️ **Ray Tune**: Distributed hyperparameter optimization with state-of-the-art algorithms (ASHA, PBT, Bayesian)
- 🔄 **Horovod Support**: Multi-node synchronization for efficient large-scale training
- 💾 **Smart Checkpointing**: Automatic model checkpointing to S3/GCS with DVC versioning

### Infrastructure & Cost Management
- ☁️ **Kubernetes Native**: Auto-scaling GPU node pools with spot instance support
- 💰 **Cost Optimizer**: Real-time cost tracking and automatic spot instance fallback
- 📊 **Resource Monitoring**: Grafana dashboards for GPU utilization and cost metrics
- 🔧 **One-Click Deployment**: Automated setup scripts for AWS, GCP, and Azure

### Evaluation & Serving
- 📈 **Comprehensive Metrics**: Perplexity, ROUGE, BLEU, BERTScore, human preference simulation
- 🧪 **A/B Testing**: Built-in framework for comparing model versions in production
- 🚀 **Ray Serve**: Low-latency model serving with automatic batching
- 📉 **Performance Benchmarks**: Automated comparison against baseline models

### Data & Experiments
- 🗄️ **Feast Integration**: Feature store for managing training datasets and embeddings
- 📊 **MLflow Tracking**: Custom LLM evaluation metrics and experiment comparison
- 🔍 **Data Versioning**: DVC integration for reproducible data pipelines
- 📦 **Dataset Utilities**: Pre-built loaders for common LLM fine-tuning datasets

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│              (CLI, Jupyter Notebooks, REST API)                  │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer (Ray)                     │
├─────────────────┬─────────────────┬────────────────────────────┤
│   Ray Train     │    Ray Tune     │      Ray Serve             │
│ (Fine-tuning)   │ (Hyperparam)    │   (Model Serving)          │
└─────────────────┴─────────────────┴────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                    MLOps & Data Layer                            │
├─────────────┬────────────┬────────────┬────────────────────────┤
│   MLflow    │   Feast    │    DVC     │   Cost Optimizer       │
│ (Tracking)  │ (Features) │ (Versions) │ (Spot Instances)       │
└─────────────┴────────────┴────────────┴────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│            Infrastructure Layer (Kubernetes)                     │
├─────────────────────────────────────────────────────────────────┤
│  Auto-scaling GPU Node Pools (Spot + On-Demand Instances)       │
│  Persistent Storage (S3/GCS), Monitoring (Prometheus/Grafana)   │
└─────────────────────────────────────────────────────────────────┘
```

See [Architecture Documentation](docs/ARCHITECTURE.md) for detailed diagrams.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker & Kubernetes (kubectl configured)
- Cloud provider account (AWS/GCP/Azure)
- NVIDIA GPU drivers (for local development)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/distributed-llm-platform.git
cd distributed-llm-platform

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Local Development (Single GPU)

```bash
# Fine-tune GPT-2 on a small dataset
python scripts/examples/quick_start.py \
  --model gpt2 \
  --dataset wikitext \
  --epochs 3 \
  --local

# Results will be logged to MLflow (localhost:5000)
```

### Distributed Training (Kubernetes)

```bash
# Deploy infrastructure (creates Kubernetes cluster with GPU nodes)
./scripts/deployment/deploy_infrastructure.sh \
  --provider aws \
  --gpu-type g4dn.xlarge \
  --num-nodes 4 \
  --spot-instances

# Launch distributed fine-tuning
python scripts/examples/distributed_training.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset alpaca \
  --num-workers 4 \
  --use-spot-instances

# Monitor at the Ray dashboard
kubectl port-forward svc/ray-dashboard 8265:8265
```

### Hyperparameter Optimization

```bash
# Run distributed hyperparameter search
python scripts/examples/hyperparameter_search.py \
  --model gpt2 \
  --dataset wikitext \
  --num-trials 20 \
  --scheduler ASHA

# View results in MLflow
mlflow ui --host 0.0.0.0 --port 5000
```

## 📁 Project Structure

```
distributed-llm-platform/
├── src/
│   ├── training/          # Ray Train integration & training loops
│   ├── tuning/            # Ray Tune hyperparameter optimization
│   ├── serving/           # Ray Serve deployment & A/B testing
│   ├── evaluation/        # Custom LLM evaluation metrics
│   ├── infrastructure/    # Kubernetes & cloud resource management
│   ├── data/              # Feast integration & data pipelines
│   ├── checkpointing/     # DVC & cloud storage checkpointing
│   └── optimization/      # Cost optimization & spot instance management
├── tests/
│   ├── unit/              # Unit tests for each module
│   ├── integration/       # End-to-end integration tests
│   └── performance/       # Performance benchmarks & regression tests
├── configs/
│   ├── kubernetes/        # K8s manifests for deployment
│   ├── ray/               # Ray cluster configurations
│   ├── mlflow/            # MLflow server setup
│   └── feast/             # Feature store definitions
├── scripts/
│   ├── deployment/        # One-click deployment scripts
│   ├── utils/             # Helper utilities
│   └── examples/          # Example usage scripts
├── notebooks/             # Jupyter notebooks for learning & exploration
├── docs/
│   ├── guides/            # Step-by-step learning guides
│   ├── api/               # API documentation
│   └── diagrams/          # Architecture diagrams
└── requirements.txt       # Python dependencies
```

## 💰 Cost Optimization

This platform includes several cost-saving strategies:

1. **Spot Instance Management**: Automatic fallback to on-demand instances
2. **Auto-scaling**: Scale down to zero when idle
3. **Checkpointing**: Resume training after spot instance interruptions
4. **Cost Tracking**: Real-time cost monitoring and budgets
5. **Resource Optimization**: Automatic batch size tuning for GPU utilization

**Average Cost Savings**: 60-80% compared to on-demand instances

See [Cost Optimization Guide](docs/guides/COST_OPTIMIZATION.md) for details.

## 📚 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Learning Guide](docs/guides/LEARNING_GUIDE.md) - Start here if you're new!
- [API Reference](docs/api/API_REFERENCE.md)
- [Deployment Guide](docs/guides/DEPLOYMENT.md)
- [Troubleshooting](docs/guides/TROUBLESHOOTING.md)
- [FAQ](docs/guides/FAQ.md)

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/performance/ -v             # Performance benchmarks

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🎓 Learning Resources

### Key Concepts to Understand
- **Distributed Training**: Data parallelism, model parallelism, gradient synchronization
- **Ray Framework**: Actors, tasks, distributed computing primitives
- **LLM Fine-tuning**: Transfer learning, LoRA, adapters
- **Kubernetes**: Pods, services, deployments, auto-scaling
- **MLOps**: Experiment tracking, model versioning, feature stores

### Recommended Reading Order
1. [Learning Guide](docs/guides/LEARNING_GUIDE.md) - Foundational concepts
2. [Architecture Overview](docs/ARCHITECTURE.md) - System design
3. [Quick Start Examples](scripts/examples/) - Hands-on practice
4. [Jupyter Notebooks](notebooks/) - Interactive tutorials

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Ray Project](https://www.ray.io/) for distributed computing framework
- [HuggingFace](https://huggingface.co/) for transformers library
- [MLflow](https://mlflow.org/) for experiment tracking
- [Feast](https://feast.dev/) for feature store

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our community](https://discord.gg/example)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/distributed-llm-platform/issues)

---

**Built with ❤️ for the ML community**

*Star ⭐ this repository if you find it helpful!*
