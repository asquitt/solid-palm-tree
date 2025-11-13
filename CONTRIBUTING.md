# Contributing to Distributed LLM Platform

Thank you for your interest in contributing! This document provides guidelines for contributions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/distributed-llm-platform.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `make test`
6. Submit a pull request

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
make format

# Lint code
make lint
```

## Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all functions
- Keep functions focused and small
- Write comprehensive comments for learning

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from maintainers

## Areas for Contribution

- Bug fixes
- New features
- Documentation improvements
- Performance optimizations
- Test coverage
- Example notebooks

## Questions?

Open an issue or join our Discord!
