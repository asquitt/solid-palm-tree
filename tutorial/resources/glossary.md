# Technical Glossary

Quick reference for terms used throughout the tutorial.

## A

**API (Application Programming Interface)**: Interface for programs to communicate. Example: REST API for model inference.

**Attention Mechanism**: Neural network component that learns relationships between tokens.

**Authentication**: Verifying identity (who are you?). Example: API keys, JWT tokens.

**Authorization**: Verifying permissions (what can you do?). Example: RBAC roles.

## B

**Backpropagation**: Algorithm for computing gradients in neural networks.

**Batch**: Group of examples processed together. Example: batch_size=32 means 32 examples at once.

**BERT**: Bidirectional Encoder Representations from Transformers. Masked language model.

## C

**Causal LM**: Language model that predicts next token (GPT-style).

**Checkpoint**: Saved model state that can be restored later.

**Circuit Breaker**: Pattern to prevent cascading failures by stopping requests to failing services.

**Cross-Entropy Loss**: Loss function for classification tasks.

## D

**Data Parallelism**: Distribute data across GPUs, same model on each.

**DataLoader**: PyTorch utility for batching and shuffling data.

**Distributed Training**: Training across multiple GPUs or machines.

## E

**Embedding**: Converting tokens to dense vectors.

**Epoch**: One complete pass through the training dataset.

**Evaluation**: Testing model performance on held-out data.

**Exponential Backoff**: Retry strategy with increasing delays.

## F

**FastAPI**: Modern Python web framework for building APIs.

**Fine-tuning**: Adapting pre-trained model to specific task.

**Forward Pass**: Input → Model → Output

## G

**GPT (Generative Pre-trained Transformer)**: Autoregressive language model.

**Gradient**: Direction and magnitude of parameter update.

**GPU (Graphics Processing Unit)**: Hardware accelerator for deep learning.

## H

**Health Check**: Endpoint to verify service is working.

**Hidden Size**: Dimension of model's internal representations.

**Hugging Face**: Company/platform for sharing models and datasets.

**Hyperparameter**: Configuration value (learning rate, batch size, etc.).

## I

**Inference**: Using trained model to make predictions.

**Input IDs**: Numerical representation of tokens.

## J

**JWT (JSON Web Token)**: Secure method for transmitting information between parties.

## K

**Kubernetes**: Container orchestration platform.

## L

**Learning Rate**: Step size for parameter updates.

**Liveness Probe**: Kubernetes check if container is alive.

**Loss**: Metric of how wrong model predictions are.

**LLM (Large Language Model)**: Neural network trained on massive text data.

## M

**Masked LM**: Model that predicts masked tokens (BERT-style).

**Model Parallelism**: Distribute model layers across GPUs.

**Multi-tenancy**: Supporting multiple independent users/organizations.

## O

**Optimizer**: Algorithm for updating parameters (AdamW, SGD, etc.).

**Overfitting**: Model memorizes training data, poor generalization.

## P

**Padding**: Adding special tokens to make sequences same length.

**Parameter**: Learnable weight in neural network.

**Perplexity**: exp(loss), measures model uncertainty.

**Pre-training**: Training on large generic dataset.

## R

**Ray**: Framework for distributed Python applications.

**Ray Serve**: Ray's model serving library.

**Ray Train**: Ray's distributed training library.

**RBAC (Role-Based Access Control)**: Permission system based on user roles.

**Readiness Probe**: Kubernetes check if container can handle traffic.

**Retry Logic**: Automatically retrying failed operations.

## S

**Sequence Length**: Number of tokens in input.

**Startup Probe**: Kubernetes check if container has started.

**Subword**: Piece of word used in tokenization.

## T

**Token**: Unit of text (word, subword, or character).

**Tokenization**: Converting text to tokens.

**Tokenizer**: Tool that performs tokenization.

**Training Loop**: Iterative process of improving model.

**Transformer**: Neural network architecture based on attention.

**Truncation**: Cutting sequences that are too long.

## V

**Validation**: Evaluating model on validation set during training.

**Vocabulary**: Set of all tokens model knows.

## W

**Warmup**: Gradually increasing learning rate at start of training.

---

**Not finding a term?** Check the week's CONCEPTS.md or search online!
