"""
E-commerce LLM Fine-tuning Project
Setup script for package installation
"""
from setuptools import setup, find_packages

setup(
    name="ecommerce-llm",
    version="0.1.0",
    description="E-commerce LLM system for product classification, attribute extraction, and Q&A",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
        "bitsandbytes>=0.41.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "evaluate>=0.4.0",
        "rouge-score>=0.1.2",
        "bert-score>=0.3.13",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort"],
        "inference": ["vllm>=0.2.0"],
        "quantization": ["autoawq>=0.1.8"],
    },
    entry_points={
        "console_scripts": [
            "train-model=training.trainer:main",
            "evaluate-model=evaluation.evaluate:main",
        ],
    },
)
