# Week 1 Concepts: LLM Fine-tuning Fundamentals

This document explains the core concepts you'll encounter in Week 1.

## Table of Contents

1. [What is an LLM?](#what-is-an-llm)
2. [Pre-training vs Fine-tuning](#pre-training-vs-fine-tuning)
3. [Tokenization](#tokenization)
4. [The Training Loop](#the-training-loop)
5. [Loss Functions](#loss-functions)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Checkpointing](#checkpointing)

---

## What is an LLM?

**LLM = Large Language Model**

### Definition
A neural network trained on massive amounts of text data to understand and generate human language.

### How They Work (Simplified)

```
Input Text → Tokenization → Model → Predictions → Output Text
```

**Example**:
```
Input:  "The cat sat on the"
Process: Model predicts next word
Output: "mat" (predicted)
```

### Architecture

Most modern LLMs use the **Transformer architecture**:

```
┌─────────────────────────────┐
│   Input: "Hello world"      │
└──────────┬──────────────────┘
           ↓
   ┌───────────────┐
   │ Tokenization  │  Convert text → numbers
   └───────┬───────┘
           ↓
   ┌───────────────┐
   │   Embedding   │  Numbers → vectors
   └───────┬───────┘
           ↓
   ┌───────────────┐
   │  Transformer  │  Process relationships
   │    Layers     │  (attention, feedforward)
   └───────┬───────┘
           ↓
   ┌───────────────┐
   │  Output Head  │  Predict next token
   └───────┬───────┘
           ↓
   "Hello world _" → "!"
```

### Key Components

1. **Embeddings**: Convert tokens to vectors
2. **Attention**: Learn relationships between words
3. **Feedforward**: Transform representations
4. **Output**: Predict next token

---

## Pre-training vs Fine-tuning

### Pre-training

**What**: Training on massive generic datasets (books, websites, etc.)

**Goal**: Learn general language understanding

**Data**: Billions of tokens

**Time**: Weeks/months on hundreds of GPUs

**Cost**: $$$$ (millions of dollars)

**Example**: GPT-3 trained on 300B+ tokens

### Fine-tuning

**What**: Training on specific task data

**Goal**: Adapt model to your use case

**Data**: Thousands to millions of tokens

**Time**: Hours to days on 1-8 GPUs

**Cost**: $ to $$ (affordable)

**Example**: Fine-tune GPT-2 on customer support conversations

### Why Fine-tuning Works

```
Pre-trained Model:  [General Language Understanding]
        +
Fine-tuning:       [Specific Task Knowledge]
        =
Final Model:       [General + Specific = Powerful]
```

**Analogy**:
- Pre-training = Learning to read and write
- Fine-tuning = Becoming a doctor/lawyer/programmer

### Comparison

| Aspect | Pre-training | Fine-tuning |
|--------|--------------|-------------|
| Data | Billions of tokens | Thousands-millions |
| Time | Weeks-months | Hours-days |
| Cost | Millions | Hundreds-thousands |
| Hardware | 100s of GPUs | 1-8 GPUs |
| Goal | General language | Specific task |

---

## Tokenization

### What is a Token?

A **token** is a piece of text (word, subword, or character) that the model processes.

### Examples

**Word-level** (naive):
```
"Hello world" → ["Hello", "world"]
```

**Subword-level** (most LLMs):
```
"unhappiness" → ["un", "happiness"]
"ChatGPT" → ["Chat", "G", "PT"]
```

**Character-level**:
```
"Hi" → ["H", "i"]
```

### Why Subwords?

**Advantages**:
- Handle unknown words (split into known subwords)
- Smaller vocabulary size
- Better generalization

**Example**:
```
Model knows: ["run", "ning", "er"]
Can understand: "runner" = ["run", "##ner"]
Even if "runner" wasn't in training data!
```

### Tokenization Process

```python
# Text to tokens
text = "Hello, world!"
tokens = tokenizer.tokenize(text)
# Output: ["Hello", ",", "world", "!"]

# Tokens to IDs (numbers the model can process)
ids = tokenizer.convert_tokens_to_ids(tokens)
# Output: [15496, 11, 995, 0]

# All in one step
encoded = tokenizer("Hello, world!")
# Output: {"input_ids": [15496, 11, 995, 0], "attention_mask": [1, 1, 1, 1]}
```

### Special Tokens

- `[CLS]` / `<|endoftext|>`: Start/end of sequence
- `[SEP]`: Separator between sequences
- `[PAD]`: Padding for batching
- `[UNK]`: Unknown token
- `[MASK]`: Masked token (for BERT-style models)

### Padding and Truncation

**Problem**: Models need fixed-length inputs, but text varies in length.

**Solution**:

```python
# Different length texts
texts = ["Hi", "Hello world", "This is a longer sentence"]

# Tokenize with padding and truncation
encoded = tokenizer(
    texts,
    padding=True,        # Pad to longest in batch
    truncation=True,     # Cut if too long
    max_length=10,       # Maximum length
    return_tensors="pt"  # Return PyTorch tensors
)

# Result:
# input_ids shape: [3, 10] (3 texts, max 10 tokens)
# attention_mask: tells model which tokens are real vs padding
```

---

## The Training Loop

### Overview

Training is the process of improving the model by showing it examples and adjusting its parameters.

### Basic Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward pass - predict
        outputs = model(batch["input_ids"])

        # 2. Calculate loss - how wrong were we?
        loss = loss_function(outputs, batch["labels"])

        # 3. Backward pass - compute gradients
        loss.backward()

        # 4. Update parameters - improve the model
        optimizer.step()

        # 5. Reset gradients for next iteration
        optimizer.zero_grad()
```

### Detailed Steps

#### 1. Forward Pass

**What happens**: Input flows through the model to produce predictions.

```
Input: "The cat sat on the"
       ↓
Model processes (matrix multiplications, attention, etc.)
       ↓
Output: Probabilities for next word
       [0.01, 0.15, 0.40, ...] (40% chance of "mat")
```

#### 2. Loss Calculation

**What is loss?**: A number measuring how wrong the model's predictions are.

**Goal**: Minimize loss (lower = better)

**Example**:
```
Correct answer: "mat" (token ID: 1245)
Model prediction: 40% probability for "mat"
Loss: 0.916 (high = bad)

After training:
Model prediction: 95% probability for "mat"
Loss: 0.051 (low = good!)
```

#### 3. Backward Pass

**What happens**: Calculate how much each parameter contributed to the loss.

**Technical**: Computes gradients via backpropagation.

**Intuition**: "If I change this parameter by X, loss changes by Y"

#### 4. Parameter Update

**What happens**: Adjust parameters to reduce loss.

**Formula**:
```
new_param = old_param - learning_rate * gradient
```

**Analogy**:
- Gradient = "which direction to go"
- Learning rate = "how big a step to take"

#### 5. Zero Gradients

**Why?**: Gradients accumulate by default. We need to reset them each step.

```python
# Without zero_grad()
step 1: gradients = [1.2, 0.5, ...]
step 2: gradients = [1.2 + 2.3, 0.5 + 1.1, ...] # WRONG! Accumulated

# With zero_grad()
step 1: gradients = [1.2, 0.5, ...]
optimizer.zero_grad()
step 2: gradients = [2.3, 1.1, ...]  # Correct!
```

### Hyperparameters

**Learning Rate**: How big are parameter updates?
- Too high: Model doesn't converge
- Too low: Training takes forever
- Typical: 1e-5 to 5e-5 for fine-tuning

**Batch Size**: How many examples to process together?
- Larger: Faster, more memory, more stable gradients
- Smaller: Slower, less memory, noisier gradients
- Typical: 8-32 for fine-tuning

**Epochs**: How many times to see the entire dataset?
- Too few: Underfitting (not learning enough)
- Too many: Overfitting (memorizing training data)
- Typical: 3-10 for fine-tuning

---

## Loss Functions

### Cross-Entropy Loss (Most Common)

**Used for**: Predicting discrete tokens (words)

**Intuition**: Penalizes wrong predictions, rewards correct ones.

**Formula**:
```
Loss = -log(probability of correct token)
```

**Example**:
```
Correct token: "cat"

Prediction 1: {"cat": 0.8, "dog": 0.1, "bird": 0.1}
Loss: -log(0.8) = 0.22 (good!)

Prediction 2: {"cat": 0.1, "dog": 0.8, "bird": 0.1}
Loss: -log(0.1) = 2.30 (bad!)
```

### In PyTorch

```python
# Automatic in most transformers
outputs = model(input_ids, labels=labels)
loss = outputs.loss  # Cross-entropy computed automatically

# Manual calculation
from torch.nn import CrossEntropyLoss

criterion = CrossEntropyLoss()
loss = criterion(logits, labels)
```

---

## Evaluation Metrics

### Perplexity

**Definition**: Measure of how "surprised" the model is by the text.

**Formula**:
```
Perplexity = exp(average_loss)
```

**Interpretation**:
- Lower = better
- Perplexity of 10 = model is as confused as if choosing from 10 random words
- Perplexity of 2 = very confident

**Example**:
```
Model A: Perplexity = 15 (okay)
Model B: Perplexity = 8  (better!)
Model C: Perplexity = 3  (excellent!)
```

### Accuracy

**Definition**: Percentage of correct predictions.

**Formula**:
```
Accuracy = (# correct predictions) / (# total predictions)
```

**Example**:
```
Predictions: ["cat", "dog", "bird", "mat", "fish"]
Actual:      ["cat", "dog", "cat",  "mat", "fish"]
Correct:     [  ✓  ,   ✓  ,   ✗  ,    ✓  ,   ✓  ]
Accuracy: 4/5 = 80%
```

### Generation Quality

For text generation tasks, use:
- **BLEU**: Compares generated text to reference
- **ROUGE**: Measures overlap with reference
- **Human evaluation**: Ultimate judge

---

## Checkpointing

### What is a Checkpoint?

A **checkpoint** is a saved snapshot of your model at a point in time.

### Why Checkpoint?

1. **Resume training** if it crashes
2. **Save best model** (not just latest)
3. **Experiment** with different settings
4. **Deploy** to production

### What to Save

```python
checkpoint = {
    "model_state_dict": model.state_dict(),      # Model parameters
    "optimizer_state_dict": optimizer.state_dict(),  # Optimizer state
    "epoch": epoch,                              # Training progress
    "loss": loss,                                # Performance metric
    "config": config,                            # Hyperparameters
}

torch.save(checkpoint, "checkpoint.pt")
```

### When to Checkpoint

**Strategies**:
1. **Every epoch**: Simple, can resume easily
2. **Every N steps**: More frequent, better for long training
3. **Best performance**: Keep only the best model

**Example**:
```python
# Save best model
if val_loss < best_loss:
    best_loss = val_loss
    torch.save(model.state_dict(), "best_model.pt")

# Save every epoch
torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")
```

### Loading a Checkpoint

```python
# Load checkpoint
checkpoint = torch.load("checkpoint.pt")

# Restore model
model.load_state_dict(checkpoint["model_state_dict"])

# Restore optimizer (to resume training)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Resume from saved epoch
start_epoch = checkpoint["epoch"] + 1
```

---

## Putting It All Together

### Complete Training Flow

```
1. Load pre-trained model
        ↓
2. Prepare dataset (tokenize, batch)
        ↓
3. Training loop:
   For each epoch:
     For each batch:
       - Forward pass
       - Calculate loss
       - Backward pass
       - Update parameters
     - Evaluate on validation set
     - Save checkpoint if best
        ↓
4. Load best checkpoint
        ↓
5. Evaluate on test set
        ↓
6. Deploy to production!
```

### Code Example

```python
# 1. Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. Prepare data
dataset = load_dataset("my_data")
dataset = dataset.map(lambda x: tokenizer(x["text"]))
dataloader = DataLoader(dataset, batch_size=8)

# 3. Training loop
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()

for epoch in range(3):
    for batch in dataloader:
        # Forward
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss

        # Backward
        loss.backward()

        # Update
        optimizer.step()
        optimizer.zero_grad()

    # Evaluate
    val_loss = evaluate(model, val_dataloader)

    # Checkpoint
    if val_loss < best_loss:
        torch.save(model.state_dict(), "best_model.pt")

# 4. Load best model
model.load_state_dict(torch.load("best_model.pt"))

# 5. Test
test_loss = evaluate(model, test_dataloader)
print(f"Test perplexity: {math.exp(test_loss)}")
```

---

## Key Takeaways

✅ **LLMs** are neural networks trained on text to predict the next word

✅ **Fine-tuning** adapts pre-trained models to specific tasks (cheap and effective)

✅ **Tokenization** converts text to numbers the model can process

✅ **Training loop**: forward → loss → backward → update → repeat

✅ **Loss** measures how wrong the model is (lower = better)

✅ **Perplexity** is the main evaluation metric (lower = better)

✅ **Checkpoints** save your progress and best models

---

## Further Reading

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [PyTorch Tutorial: Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

**Ready to code?** Head back to `README.md` and start implementing! 🚀
