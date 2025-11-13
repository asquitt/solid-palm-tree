# Learning Guide: Distributed LLM Fine-tuning

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Core Concepts](#core-concepts)
4. [Step-by-Step Tutorial](#step-by-step-tutorial)
5. [Key Terms Glossary](#key-terms-glossary)
6. [Learning Path](#learning-path)
7. [Common Pitfalls](#common-pitfalls)
8. [Additional Resources](#additional-resources)

## Introduction

Welcome! This guide will help you understand and use the Distributed LLM Fine-tuning Platform, even if you're new to distributed machine learning. We'll start with fundamentals and build up to advanced concepts.

**Who is this for?**
- ML engineers learning distributed training
- Data scientists fine-tuning models for the first time
- Students learning about LLMs and scalable ML
- Anyone curious about production ML systems

**What you'll learn**:
- How LLM fine-tuning works
- Distributed training fundamentals
- Hyperparameter optimization strategies
- Cost optimization techniques
- Production deployment patterns

## Prerequisites

### Required Knowledge

**Must Have** ✅:
- Basic Python programming
- Understanding of machine learning (train/test, loss, gradients)
- Familiarity with deep learning concepts (neural networks, backpropagation)

**Nice to Have** 🎯:
- Experience with PyTorch or TensorFlow
- Understanding of transformers and attention
- Basic command-line skills
- Cloud computing fundamentals

### Software Requirements

```bash
# Python 3.8+
python --version

# GPU drivers (if using GPU)
nvidia-smi

# Docker (for Kubernetes)
docker --version

# kubectl (for Kubernetes)
kubectl version
```

### Hardware Requirements

**Minimum (Learning)**:
- CPU: 4 cores
- RAM: 16 GB
- GPU: Optional (CPU training works but is slow)

**Recommended (Development)**:
- CPU: 8+ cores
- RAM: 32 GB
- GPU: NVIDIA T4 (16GB) or better

**Production**:
- Multiple GPUs (A10, A100)
- High-speed interconnect (NVLink)
- Cloud cluster with auto-scaling

## Core Concepts

### 1. What is Fine-tuning?

**Simple Explanation**:
Fine-tuning is like teaching a smart student (pre-trained model) a specific subject. The student already knows language and general knowledge, but you're teaching them domain-specific skills.

**Technical Definition**:
Fine-tuning is transfer learning where we:
1. Start with a pre-trained model (trained on billions of tokens)
2. Continue training on task-specific data
3. Update weights to adapt to new domain/task

**Example**:
```python
# Pre-trained GPT-2 knows general language
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Fine-tune on medical data → Medical GPT-2
# Fine-tune on code → Code GPT-2
# Fine-tune on your data → Your Custom GPT-2
```

**Why Fine-tune?**
- ✅ Better performance than prompting alone
- ✅ Cheaper than training from scratch (1000x less compute)
- ✅ Works with less data than training from scratch
- ✅ Preserves general knowledge while adding specific skills

### 2. Distributed Training

**Simple Explanation**:
Distributed training is like having multiple workers building a house together. Each worker does part of the work, they coordinate, and together they finish faster.

**Technical Definition**:
Training a model across multiple GPUs/machines by:
1. **Data Parallelism**: Each GPU processes different data batches
2. **Gradient Synchronization**: GPUs share gradients and update together
3. **Scaling**: More GPUs = faster training (with some overhead)

**Visual**:
```
Single GPU Training:
┌────────┐
│  GPU 1 │ → Processes 8 samples/batch → 100 samples/sec
└────────┘

Distributed Training (4 GPUs):
┌────────┐
│  GPU 1 │ → 8 samples
├────────┤
│  GPU 2 │ → 8 samples    → Sync gradients → 350 samples/sec
├────────┤                    (3.5x speedup, not perfect 4x)
│  GPU 3 │ → 8 samples
├────────┤
│  GPU 4 │ → 8 samples
└────────┘
```

**Why not perfect 4x speedup?**
- Communication overhead (synchronizing gradients)
- Network bandwidth limits
- Stragglers (slowest GPU limits speed)

### 3. Hyperparameter Optimization

**Simple Explanation**:
Hyperparameters are the knobs you turn before training. HPO is systematically trying different knob settings to find the best configuration.

**Key Hyperparameters for LLMs**:
1. **Learning Rate**: How big are weight updates?
   - Too high: Training unstable, doesn't converge
   - Too low: Training too slow, may not converge
   - Typical range: 1e-5 to 5e-5 for fine-tuning

2. **Batch Size**: How many samples per gradient update?
   - Larger: More stable gradients, better GPU utilization
   - Smaller: Less memory, noisier gradients
   - Typical: 8-32 for LLMs

3. **Number of Epochs**: How many passes through data?
   - Too few: Underfitting (model hasn't learned enough)
   - Too many: Overfitting (model memorizes training data)
   - Typical: 3-5 epochs for fine-tuning

**Search Strategies**:

1. **Grid Search**: Try all combinations
   ```python
   learning_rates = [1e-5, 5e-5, 1e-4]
   batch_sizes = [8, 16, 32]
   # Total trials: 3 × 3 = 9
   ```
   - Pros: Thorough
   - Cons: Expensive (exponential growth)

2. **Random Search**: Sample randomly
   ```python
   learning_rate = uniform(1e-6, 1e-3)
   batch_size = choice([8, 16, 32])
   # Try 20 random samples
   ```
   - Pros: More efficient than grid
   - Cons: May miss optimal region

3. **Bayesian Optimization**: Learn from past trials
   ```python
   # Build probabilistic model of metric vs. hyperparams
   # Sample from regions likely to be good
   # Balance exploration vs. exploitation
   ```
   - Pros: Most efficient
   - Cons: More complex

4. **ASHA (Aggressive Early Stopping)**: Stop bad trials early
   ```python
   # Start all trials
   # After 1 epoch: keep top 50%
   # After 2 epochs: keep top 50% of remaining
   # After 4 epochs: keep top 50% of remaining
   # Final: best 1-2 trials run to completion
   ```
   - Pros: 60-90% cost savings!
   - Cons: May stop promising slow starters

### 4. Ray Framework

**Simple Explanation**:
Ray is like a universal remote control for distributed computing. Instead of manually managing multiple machines, Ray handles it automatically.

**Key Ray Components**:

1. **Ray Core**: Distributed computing primitives
   ```python
   @ray.remote
   def train_model(config):
       # This function can run on any machine in the cluster
       return model
   ```

2. **Ray Train**: Distributed training
   ```python
   # Automatically handles:
   # - Launching workers on multiple GPUs
   # - Setting up distributed PyTorch
   # - Synchronizing gradients
   # - Fault tolerance
   ```

3. **Ray Tune**: Hyperparameter optimization
   ```python
   # Automatically handles:
   # - Parallel trial execution
   # - Early stopping
   # - Checkpointing
   # - Result tracking
   ```

4. **Ray Serve**: Model serving
   ```python
   # Automatically handles:
   # - Load balancing
   # - Autoscaling
   # - Batching
   ```

### 5. Cost Optimization

**Spot Instances**:

Cloud providers sell unused capacity at 60-80% discount:
- **AWS**: Spot Instances
- **GCP**: Preemptible VMs
- **Azure**: Spot Virtual Machines

**The Catch**: Can be interrupted with 2-minute warning

**How to Use Safely**:
1. **Checkpoint frequently**: Save every 5-10 minutes
2. **Use DVC**: Version checkpoints to cloud storage
3. **Automatic recovery**: Resume from last checkpoint
4. **Fallback option**: Switch to on-demand if spots unavailable

**Cost Comparison**:
```
Model: GPT-2 (124M params)
Data: 100K samples
Training time: 4 hours

On-Demand (p3.2xlarge): $3.06/hour × 4 = $12.24
Spot (p3.2xlarge):      $0.92/hour × 4 = $3.68

Savings: $8.56 (70%)

With 5 interruptions + recovery overhead:
Spot cost: $0.92 × 4.5 = $4.14
Still 66% cheaper!
```

## Step-by-Step Tutorial

### Tutorial 1: Your First Fine-tuning (10 minutes)

**Goal**: Fine-tune GPT-2 on WikiText dataset

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run quick start script
python scripts/examples/quick_start.py \
  --model gpt2 \
  --dataset wikitext \
  --epochs 1 \
  --batch-size 8

# 3. Monitor progress
# Open http://localhost:5000 in browser (MLflow UI)

# 4. Find your model
ls outputs/quick_start/best_model/
```

**What you learned**:
- ✅ How to configure training
- ✅ How to run training
- ✅ How to monitor with MLflow
- ✅ Where models are saved

### Tutorial 2: Hyperparameter Optimization (30 minutes)

**Goal**: Find best learning rate and batch size

```python
# 1. Create base configuration
from src.training.config import TrainingConfig

base_config = TrainingConfig(
    model_name="gpt2",
    dataset_name="wikitext",
    num_epochs=3,
)

# 2. Define search space
from src.tuning.search_spaces import get_search_space

search_space = get_search_space("quick")
# Searches: learning_rate, batch_size, warmup_ratio

# 3. Configure tuning
from src.tuning.tuner import TuningConfig, HyperparameterTuner

tuning_config = TuningConfig(
    num_samples=10,  # Try 10 configurations
    scheduler="asha",  # Use early stopping
    max_concurrent_trials=2,  # Run 2 at a time
)

# 4. Run tuning
tuner = HyperparameterTuner(base_config, search_space, tuning_config)
results = tuner.tune()

# 5. Get best config
best_config = tuner.get_best_config(results)
print(f"Best learning rate: {best_config.learning_rate}")
print(f"Best batch size: {best_config.batch_size}")
```

**What you learned**:
- ✅ How to define search spaces
- ✅ How to run parallel trials
- ✅ How ASHA saves compute
- ✅ How to find best hyperparameters

### Tutorial 3: Distributed Training (45 minutes)

**Goal**: Train on 4 GPUs simultaneously

```python
from src.training.config import TrainingConfig
from src.training.trainer import DistributedTrainer

# 1. Configure for distributed training
config = TrainingConfig(
    model_name="gpt2",
    dataset_name="wikitext",
    num_epochs=3,
    batch_size=8,
    num_workers=4,  # 4 GPUs
)

# 2. Create distributed trainer
trainer = DistributedTrainer(config)

# 3. Launch training
# Ray automatically:
# - Launches 4 workers
# - Shards data across workers
# - Synchronizes gradients
# - Aggregates metrics
results = trainer.train_distributed()

# 4. Check speedup
# Expect: ~3.5x faster than single GPU
# (Not perfect 4x due to communication overhead)
```

**What you learned**:
- ✅ How to configure multi-GPU training
- ✅ How Ray Train works
- ✅ Why speedup isn't linear
- ✅ How to monitor distributed jobs

### Tutorial 4: Cost-Optimized Training (60 minutes)

**Goal**: Use spot instances to save 70%

```python
config = TrainingConfig(
    model_name="gpt2",
    dataset_name="wikitext",
    num_epochs=5,

    # Cost optimization settings
    use_spot_instances=True,
    spot_fallback_enabled=True,
    checkpoint_frequency=100,  # Checkpoint every 100 steps
    checkpoint_to_cloud=True,
    cloud_storage_path="s3://my-bucket/checkpoints",
    use_dvc=True,
)

# Training will:
# 1. Use spot instances (70% cheaper)
# 2. Checkpoint frequently to S3
# 3. Handle interruptions gracefully
# 4. Resume from checkpoints
# 5. Fallback to on-demand if needed

trainer = Trainer(config)
results = trainer.train()
```

**What you learned**:
- ✅ How spot instances work
- ✅ How to handle interruptions
- ✅ How to use cloud storage
- ✅ How to calculate savings

## Key Terms Glossary

### Training Terms

**Fine-tuning**: Adapting a pre-trained model to a specific task by continued training

**Epoch**: One complete pass through the entire training dataset

**Batch**: Subset of data processed together (e.g., 8 samples)

**Gradient**: Direction and magnitude to update weights

**Learning Rate**: Step size for weight updates

**Overfitting**: Model memorizes training data, poor generalization

**Underfitting**: Model hasn't learned enough, poor performance

### Model Terms

**LLM (Large Language Model)**: Neural network trained on massive text data (e.g., GPT, Llama)

**Parameters**: Learnable weights in the model (e.g., GPT-2 has 124M parameters)

**Checkpoint**: Saved model state (weights, optimizer, etc.)

**Inference**: Using trained model for predictions

### Distributed Training Terms

**Data Parallelism**: Different GPUs process different data

**Model Parallelism**: Different GPUs hold different parts of model

**Gradient Synchronization**: Averaging gradients across GPUs

**AllReduce**: Efficient algorithm for gradient synchronization

**Worker**: Process/GPU participating in distributed training

**Rank**: Identifier for each worker (0, 1, 2, ...)

### Optimization Terms

**Hyperparameter**: Configuration set before training (not learned)

**Hyperparameter Optimization (HPO)**: Searching for best hyperparameters

**Search Space**: Range of hyperparameter values to explore

**Trial**: Single training run with specific hyperparameters

**Early Stopping**: Stopping training when validation stops improving

**ASHA**: Asynchronous Successive Halving Algorithm (aggressive early stopping)

**PBT**: Population Based Training (evolutionary approach)

### Infrastructure Terms

**Ray**: Distributed computing framework

**Kubernetes (K8s)**: Container orchestration system

**Spot Instance**: Discounted cloud instance that can be interrupted

**Node**: Physical or virtual machine in cluster

**Pod**: Kubernetes unit of deployment (one or more containers)

**Auto-scaling**: Automatic adjustment of resources based on load

### MLOps Terms

**MLflow**: Experiment tracking platform

**DVC**: Data Version Control for ML artifacts

**Feast**: Feature store for ML data

**Checkpoint**: Saved state of training (for recovery)

**Artifact**: Output of training (model, metrics, etc.)

## Learning Path

### Beginner (Week 1-2)

**Goals**:
- Understand LLM basics
- Run first fine-tuning job
- Use MLflow for tracking

**Tasks**:
1. Read Architecture Documentation
2. Complete Tutorial 1 (Quick Start)
3. Modify hyperparameters and observe effects
4. Browse MLflow UI

**Resources**:
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [HuggingFace Course](https://huggingface.co/course)

### Intermediate (Week 3-4)

**Goals**:
- Understand distributed training
- Run hyperparameter optimization
- Use multiple GPUs

**Tasks**:
1. Complete Tutorial 2 (HPO)
2. Complete Tutorial 3 (Distributed)
3. Compare different search spaces
4. Monitor Ray Dashboard

**Resources**:
- [Ray Documentation](https://docs.ray.io)
- [Distributed Training Guide](https://pytorch.org/tutorials/beginner/dist_overview.html)

### Advanced (Week 5-6)

**Goals**:
- Deploy to Kubernetes
- Use spot instances
- Optimize costs
- Deploy models

**Tasks**:
1. Complete Tutorial 4 (Cost Optimization)
2. Deploy to Kubernetes cluster
3. Set up monitoring dashboards
4. Implement custom evaluation metrics

**Resources**:
- [Kubernetes Basics](https://kubernetes.io/docs/tutorials/)
- [Cost Optimization Strategies](docs/guides/COST_OPTIMIZATION.md)

### Expert (Ongoing)

**Goals**:
- Contribute improvements
- Handle edge cases
- Optimize for specific use cases

**Tasks**:
1. Implement custom callbacks
2. Add new model architectures
3. Optimize for your workload
4. Contribute to repository

## Common Pitfalls

### 1. Out of Memory (OOM) Errors

**Symptoms**: Training crashes with "CUDA out of memory"

**Solutions**:
```python
# 1. Reduce batch size
config.batch_size = 4  # or 2, or 1

# 2. Enable gradient checkpointing
config.gradient_checkpointing = True

# 3. Use mixed precision
config.precision = "fp16"

# 4. Use LoRA (train fewer parameters)
config.use_lora = True
config.lora_r = 8

# 5. Use gradient accumulation
config.gradient_accumulation_steps = 4
# Effective batch size = 4 × 4 = 16
```

### 2. Training Doesn't Converge

**Symptoms**: Loss doesn't decrease or is unstable

**Solutions**:
```python
# 1. Lower learning rate
config.learning_rate = 1e-5  # Instead of 5e-5

# 2. Add warmup
config.warmup_ratio = 0.1

# 3. Increase batch size (more stable gradients)
config.batch_size = 16

# 4. Check data quality
# - Ensure proper tokenization
# - Remove corrupted samples
# - Balance dataset
```

### 3. Training Too Slow

**Symptoms**: Training takes forever

**Solutions**:
```python
# 1. Use mixed precision
config.precision = "fp16"  # 2x speedup

# 2. Increase batch size
config.batch_size = 32

# 3. Use multiple GPUs
config.num_workers = 4

# 4. Profile bottlenecks
# - Check GPU utilization (nvidia-smi)
# - Check data loading (increase num_workers)
# - Check disk I/O
```

### 4. Spot Instances Keep Interrupting

**Symptoms**: Training keeps stopping and restarting

**Solutions**:
```python
# 1. Checkpoint more frequently
config.checkpoint_frequency = 100  # Every 100 steps

# 2. Use multiple instance types
# Diversification reduces interruption rate

# 3. Switch to on-demand if interruptions > threshold
config.spot_fallback_enabled = True
config.max_interruptions = 3

# 4. Try different availability zones
```

### 5. Models Don't Improve on Validation

**Symptoms**: Training loss decreases but validation loss doesn't

**Solutions**:
```python
# 1. Regularization
config.weight_decay = 0.1

# 2. Dropout
config.lora_dropout = 0.1  # If using LoRA

# 3. Early stopping
# Use EarlyStoppingCallback

# 4. More data
# Collect more training samples

# 5. Simpler model
# Use smaller model or LoRA
```

## Additional Resources

### Documentation
- [Architecture Overview](../ARCHITECTURE.md)
- [API Reference](../api/API_REFERENCE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Troubleshooting](TROUBLESHOOTING.md)

### External Resources

**Transformers & LLMs**:
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Original paper)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

**Distributed Training**:
- [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Ray Train Documentation](https://docs.ray.io/en/latest/train/train.html)
- [Horovod Documentation](https://horovod.readthedocs.io/)

**Hyperparameter Optimization**:
- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
- [Hyperparameter Optimization Book](https://www.automl.org/book/)

**MLOps**:
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Feast Documentation](https://docs.feast.dev/)

**Cloud & Kubernetes**:
- [Kubernetes Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
- [AWS Training](https://aws.amazon.com/training/)
- [GCP Training](https://cloud.google.com/training)

### Community
- GitHub Issues: [Report bugs or ask questions](https://github.com/yourusername/distributed-llm-platform/issues)
- Discord: [Join discussions](https://discord.gg/example)
- Stack Overflow: Tag `distributed-llm-platform`

### Papers to Read

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original transformer paper

2. **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
   - Foundation of modern NLP

3. **GPT-3: Language Models are Few-Shot Learners** (Brown et al., 2020)
   - Demonstrates scale and capabilities

4. **LoRA: Low-Rank Adaptation** (Hu et al., 2021)
   - Parameter-efficient fine-tuning

5. **FlashAttention** (Dao et al., 2022)
   - Fast and memory-efficient attention

6. **ASHA: A System for Massively Parallel Hyperparameter Tuning** (Li et al., 2020)
   - Efficient hyperparameter search

## Next Steps

Now that you've completed this guide:

1. ✅ **Practice**: Run all tutorials with different datasets
2. ✅ **Experiment**: Try different hyperparameters
3. ✅ **Deploy**: Set up production deployment
4. ✅ **Optimize**: Fine-tune for your specific use case
5. ✅ **Contribute**: Share improvements with community

**Happy Learning! 🚀**

Got questions? Open an issue or join our Discord!
