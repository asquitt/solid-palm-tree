"""
Setup configuration for Distributed LLM Fine-tuning Platform.

This setup.py allows the project to be installed as a Python package,
making imports cleaner and enabling editable installs for development.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README for the long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="distributed-llm-platform",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready distributed LLM fine-tuning and evaluation platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/distributed-llm-platform",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/distributed-llm-platform/issues",
        "Documentation": "https://github.com/yourusername/distributed-llm-platform/docs",
        "Source Code": "https://github.com/yourusername/distributed-llm-platform",
    },
    packages=find_packages(where=".", exclude=["tests*", "docs*", "scripts*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.1.0",
            "mkdocstrings[python]>=0.22.0",
        ],
        "test": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "locust>=2.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-train=src.training.cli:main",
            "llm-tune=src.tuning.cli:main",
            "llm-serve=src.serving.cli:main",
            "llm-eval=src.evaluation.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
