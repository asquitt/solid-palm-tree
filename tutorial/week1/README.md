# Week 1: Basic LLM Fine-tuning (Single-Node)

Welcome to Week 1! This week you'll learn the fundamentals of fine-tuning large language models on a single machine.

## 🎯 Learning Goals

By the end of this week, you will:

- ✅ Understand how LLMs work at a high level
- ✅ Load pre-trained models from Hugging Face
- ✅ Prepare and tokenize datasets
- ✅ Implement a training loop from scratch
- ✅ Calculate and optimize loss
- ✅ Evaluate model performance
- ✅ Save and load checkpoints

## 📚 Prerequisites

- Basic Python knowledge
- Understanding of neural networks (high level)
- Completed environment setup (`tutorial/SETUP.md`)

## 📖 This Week's Structure

### 1. **Read the Concepts** (`CONCEPTS.md`)
Start here to understand the theory behind what you'll build.

### 2. **Implement Starter Code** (`starter/`)
Fill in the blanks in these files:
- `model_loader.py` - Load pre-trained models
- `data_prep.py` - Prepare and tokenize datasets
- `trainer.py` - Implement training loop
- `evaluator.py` - Evaluate model performance

### 3. **Run Tests** (`starter/tests/`)
```bash
cd starter
pytest tests/ -v
```

Tests will guide your implementation.

### 4. **Complete Exercises** (`exercises/`)
Additional practice problems to reinforce learning.

### 5. **Compare with Solution** (`solution/`)
Reference implementation (check after attempting yourself!)

## 🚀 Getting Started

### Step 1: Read Concepts

```bash
cat CONCEPTS.md
```

Spend 30-60 minutes understanding the theory.

### Step 2: Navigate to Starter Code

```bash
cd starter
ls -la
```

You should see:
```
model_loader.py      # TODO: Load pre-trained models
data_prep.py         # TODO: Prepare datasets
trainer.py           # TODO: Implement training
evaluator.py         # TODO: Evaluate models
tests/               # Tests to guide you
```

### Step 3: Start with Model Loading

```bash
# Open in your editor
code model_loader.py

# Or use vim/nano
vim model_loader.py
```

Look for sections marked:
```python
# TODO: Your task here
# HINT: Helpful guidance
# YOUR CODE HERE
```

### Step 4: Run Tests Frequently

```bash
# Run all tests
pytest tests/ -v

# Run specific file
pytest tests/test_model_loader.py -v

# Run specific test
pytest tests/test_model_loader.py::test_load_model -v
```

### Step 5: Implement Each Component

Suggested order:
1. `model_loader.py` (easiest, ~30 minutes)
2. `data_prep.py` (medium, ~1 hour)
3. `trainer.py` (hardest, ~2-3 hours)
4. `evaluator.py` (medium, ~1 hour)

## 📝 Implementation Checklist

Use this to track your progress:

### Model Loading (`model_loader.py`)
- [ ] Load tokenizer from Hugging Face
- [ ] Load pre-trained model
- [ ] Move model to correct device (CPU/GPU)
- [ ] Implement model info function
- [ ] All tests passing

### Data Preparation (`data_prep.py`)
- [ ] Load dataset from Hugging Face
- [ ] Implement tokenization function
- [ ] Create data collator for batching
- [ ] Create DataLoader
- [ ] Handle padding and truncation
- [ ] All tests passing

### Training (`trainer.py`)
- [ ] Implement forward pass
- [ ] Calculate loss
- [ ] Implement backward pass
- [ ] Update optimizer
- [ ] Handle learning rate scheduling
- [ ] Implement training loop for one epoch
- [ ] Add progress tracking
- [ ] All tests passing

### Evaluation (`evaluator.py`)
- [ ] Implement evaluation loop
- [ ] Calculate perplexity
- [ ] Calculate accuracy
- [ ] Generate sample predictions
- [ ] All tests passing

## 🧪 Testing Your Implementation

### Run All Tests

```bash
cd starter
pytest tests/ -v
```

Expected output when complete:
```
tests/test_model_loader.py::test_load_tokenizer PASSED
tests/test_model_loader.py::test_load_model PASSED
tests/test_data_prep.py::test_tokenization PASSED
tests/test_data_prep.py::test_dataloader PASSED
tests/test_trainer.py::test_forward_pass PASSED
tests/test_trainer.py::test_loss_calculation PASSED
tests/test_trainer.py::test_training_step PASSED
tests/test_evaluator.py::test_evaluation PASSED
tests/test_evaluator.py::test_perplexity PASSED

================= 9 passed in 12.34s =================
```

### Run with Coverage

```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

## 💡 Tips and Hints

### General Tips

1. **Read the tests first** - They show you exactly what's expected
2. **Use print statements** - Debug by printing intermediate values
3. **Start simple** - Get basic functionality working before optimizing
4. **Check shapes** - Most bugs are tensor shape mismatches
5. **Use the debugger** - Set breakpoints with `import pdb; pdb.set_trace()`

### Model Loading Tips

```python
# Load from Hugging Face Hub
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### Tokenization Tips

```python
# Tokenize with padding and truncation
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# Access token IDs and attention mask
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]
```

### Training Loop Tips

```python
# Basic training step
model.train()
optimizer.zero_grad()

outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
loss = outputs.loss

loss.backward()
optimizer.step()
```

### Common Errors and Solutions

**Error**: `CUDA out of memory`
```python
# Solution: Reduce batch size
batch_size = 4  # Instead of 32
```

**Error**: `RuntimeError: Expected all tensors to be on the same device`
```python
# Solution: Move data to same device as model
inputs = inputs.to(device)
```

**Error**: `TypeError: forward() got an unexpected keyword argument 'labels'`
```python
# Solution: Use the correct model type
# For causal LM: AutoModelForCausalLM
# For masked LM: AutoModelForMaskedLM
```

## 📚 Key Concepts to Understand

### Before You Code
- What is fine-tuning?
- Difference between pre-training and fine-tuning
- What are tokens and tokenization?
- What is a loss function?

### While You Code
- How does the forward pass work?
- What does backward() do?
- Why do we need optimizer.zero_grad()?
- What is perplexity?

### After You Code
- Why does fine-tuning work better than training from scratch?
- What are the trade-offs in batch size?
- How do learning rates affect training?

Check `CONCEPTS.md` for detailed explanations!

## 🎓 Exercises

After completing the starter code, try these exercises in `exercises/`:

1. **Exercise 1**: Fine-tune GPT-2 on a custom dataset
2. **Exercise 2**: Implement learning rate warmup
3. **Exercise 3**: Add gradient clipping
4. **Exercise 4**: Implement early stopping
5. **Exercise 5**: Generate text from your fine-tuned model

## 🔍 Solution Code

Once you've attempted the implementation, compare with the reference solution:

```bash
cd ../solution
cat trainer.py
```

Don't look too early - struggling is part of learning!

## ⏭️ Next Steps

Once you've completed Week 1:

1. ✅ All tests passing
2. ✅ Understand each component
3. ✅ Completed at least 3 exercises
4. 🚀 Move to Week 2: Distributed Training

Read `NEXT_STEPS.md` for Week 2 preparation.

## 📊 Time Estimate

- Reading concepts: 1 hour
- Implementing model_loader.py: 0.5 hours
- Implementing data_prep.py: 1 hour
- Implementing trainer.py: 2-3 hours
- Implementing evaluator.py: 1 hour
- Exercises: 2-3 hours

**Total: 8-10 hours**

## 🆘 Getting Stuck?

1. **Re-read the concepts** - `CONCEPTS.md`
2. **Check the tests** - They show expected behavior
3. **Use print debugging** - Print shapes and values
4. **Review error messages** - They usually tell you what's wrong
5. **Check the solution** - But try yourself first!

## 🎉 You've Got This!

Remember:
- Everyone struggles at first
- Debugging is learning
- Small steps lead to big progress
- Ask questions (even to yourself)

Ready? Let's start coding! 🚀

```bash
cd starter
code model_loader.py
```
