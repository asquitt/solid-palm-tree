# Frequently Asked Questions (FAQ)

## General Questions

### Q: What is this platform for?
A: Distributed fine-tuning and deployment of large language models using Ray, with cost optimization and production-ready tooling.

### Q: What models are supported?
A: Any HuggingFace transformer model: GPT-2, Llama-2, GPT-Neo, OPT, BLOOM, etc.

### Q: Do I need multiple GPUs?
A: No! Works on single GPU, single CPU, or distributed clusters. Start small, scale as needed.

### Q: What cloud providers are supported?
A: AWS, GCP, and Azure. The platform is cloud-agnostic.

## Training Questions

### Q: How long does training take?
A: Depends on model size and data:
- GPT-2 (124M) on WikiText: 15-30 minutes (single GPU)
- Llama-2-7B on 100K samples: 4-8 hours (8x A100)

### Q: How much does training cost?
A: Examples with spot instances (70% discount):
- GPT-2: $0.50-1.00
- GPT-Neo-2.7B: $5-10
- Llama-2-7B: $20-50

### Q: What if I run out of GPU memory?
A: Try these solutions:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use LoRA (train fewer parameters)
4. Use 8-bit quantization

### Q: Can I use CPU only?
A: Yes, but it's very slow (100x slower than GPU). Use for development/testing only.

## Distributed Training Questions

### Q: How many GPUs do I need?
A: Start with 1. Scale to 2-4 for faster training. 8+ for large models (7B+ parameters).

### Q: What speedup can I expect?
A: Rule of thumb: 80-90% of linear speedup
- 2 GPUs: 1.8x faster
- 4 GPUs: 3.5x faster
- 8 GPUs: 7x faster

### Q: Why not perfect linear speedup?
A: Communication overhead for gradient synchronization. More GPUs = more communication.

## Cost Optimization Questions

### Q: How much can spot instances save?
A: 60-80% compared to on-demand instances.

### Q: What happens if spot instance is interrupted?
A: Training pauses, checkpoint is saved, and automatically resumes when new instance starts.

### Q: How often are interruptions?
A: Varies by instance type and region. Typically 5-20% interruption rate.

### Q: Should I always use spot instances?
A: For training: Yes (with checkpointing)
For inference: No (use on-demand for reliability)

## Hyperparameter Tuning Questions

### Q: How many trials should I run?
A: 
- Quick search: 5-10 trials
- Default search: 20-30 trials
- Extensive search: 50-100 trials

### Q: How long does HPO take?
A: With ASHA early stopping: 2-4x longer than single training run.
Without: num_trials × training_time

### Q: Which search algorithm should I use?
A: 
- Start with Random or Optuna
- Use ASHA scheduler for cost savings
- Use PBT for learning rate schedules

## Deployment Questions

### Q: How do I deploy a fine-tuned model?
A: Use Ray Serve (included) or HuggingFace Inference API.

### Q: What's the latency?
A: Depends on model size:
- GPT-2: 20-50ms per request
- GPT-Neo-2.7B: 100-200ms
- Llama-2-7B: 200-500ms

### Q: How many requests can it handle?
A: With autoscaling: thousands of requests per second.

## Troubleshooting

### Q: Training is very slow, what should I check?
A: 
1. GPU utilization (should be >80%)
2. Data loading (increase num_workers)
3. Batch size (larger = faster)
4. Mixed precision (FP16 = 2x speedup)

### Q: Loss is NaN, what's wrong?
A: 
1. Lower learning rate
2. Add gradient clipping
3. Check for corrupted data
4. Use mixed precision carefully (may need loss scaling)

### Q: Model doesn't improve on validation, why?
A:
1. Overfitting: Add regularization
2. Too few epochs: Train longer
3. Learning rate too high: Reduce
4. Bad hyperparameters: Run HPO

### Q: Where can I get help?
A: 
1. Check documentation
2. Search GitHub issues
3. Open new issue
4. Join Discord community
