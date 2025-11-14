# Week 2: Distributed Training with Ray

Learn to scale training across multiple GPUs and machines.

## Goals
- ✅ Understand Ray fundamentals
- ✅ Implement data-parallel training
- ✅ Use Ray Train for distributed fine-tuning
- ✅ Handle multi-GPU checkpointing

## Files
- `starter/ray_trainer.py` - Distributed trainer implementation
- `starter/scaling_config.py` - Configure distributed resources
- `solution/` - Reference implementations

## Key Concepts (see CONCEPTS.md)
- Data parallelism vs model parallelism
- AllReduce for gradient synchronization
- Ray Train API
- Distributed checkpointing

## Getting Started
```bash
cd starter
# Implement ray_trainer.py
pytest tests/ -v
```

## Time Estimate: 10-12 hours

---

**Prerequisites**: Complete Week 1

**Next**: Week 3 - Model Serving
