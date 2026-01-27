"""
Configuration for the Streamlit demo app.

This file contains all configurable settings for the demo,
making it easy to add new models and adjust behavior.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import os

# API Configuration
VLLM_FINETUNED_URL = os.getenv("VLLM_FINETUNED_URL", "http://localhost:8000/v1")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")

# For single-model setup (uses same port for both)
VLLM_SINGLE_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    model_id: str
    api_url: str
    description: str
    is_finetuned: bool = False


# Available models
MODELS: Dict[str, ModelConfig] = {
    "mistral-base": ModelConfig(
        name="Mistral-7B (Base)",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        api_url=VLLM_BASE_URL,
        description="Original Mistral-7B without fine-tuning",
        is_finetuned=False,
    ),
    "mistral-finetuned": ModelConfig(
        name="Mistral-7B (Fine-tuned)",
        model_id="ciq-model",
        api_url=VLLM_FINETUNED_URL,
        description="Fine-tuned on 50K e-commerce examples with QLoRA",
        is_finetuned=True,
    ),
    # Add more models here as they become available
    # "llama-finetuned": ModelConfig(
    #     name="LLaMA-3-8B (Fine-tuned)",
    #     model_id="llama-ciq-model",
    #     api_url=VLLM_FINETUNED_URL,
    #     description="Fine-tuned LLaMA-3 for e-commerce",
    #     is_finetuned=True,
    # ),
}


# Task configurations
@dataclass
class TaskConfig:
    """Configuration for a task type."""
    prefix: str
    instruction: str
    suffix: str
    examples: List[Dict[str, str]]


TASKS: Dict[str, TaskConfig] = {
    "classify": TaskConfig(
        prefix="[CLASSIFY]",
        instruction="Classify the following product into the Google Product Taxonomy.",
        suffix="\nCategory:",
        examples=[
            {
                "title": "Electronics - Headphones",
                "prompt": "Product: Sony WH-1000XM5 Wireless Noise Canceling Headphones",
            },
            {
                "title": "Apparel - Shoes",
                "prompt": "Product: Nike Air Max 90 Men's Running Shoes - White/Black",
            },
            {
                "title": "Home & Kitchen",
                "prompt": "Product: Instant Pot Duo 7-in-1 Electric Pressure Cooker 6Qt",
            },
        ],
    ),
    "extract": TaskConfig(
        prefix="[EXTRACT]",
        instruction="Extract product attributes as JSON from the following product.",
        suffix="\nAttributes:",
        examples=[
            {
                "title": "Smartphone",
                "prompt": "Product: Apple iPhone 15 Pro Max 256GB Natural Titanium",
            },
            {
                "title": "Laptop",
                "prompt": "Product: Dell XPS 15 - Intel i7, 16GB RAM, 512GB SSD, RTX 4050",
            },
            {
                "title": "Clothing",
                "prompt": "Product: Levi's 501 Original Fit Men's Jeans - Dark Wash 32x30",
            },
        ],
    ),
    "qa": TaskConfig(
        prefix="[QA]",
        instruction="Answer the question about the following product.",
        suffix="\nAnswer:",
        examples=[
            {
                "title": "Battery Life",
                "prompt": "Product: MacBook Air M3 with 18-hour battery\nQuestion: How long does the battery last?",
            },
            {
                "title": "Compatibility",
                "prompt": "Product: Logitech MX Master 3S - Works with macOS, Windows, Linux\nQuestion: Does it work with Mac?",
            },
            {
                "title": "Size",
                "prompt": "Product: Samsung 65-inch OLED 4K TV\nQuestion: What is the screen size?",
            },
        ],
    ),
}


# Generation defaults
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.95


# UI Configuration
APP_TITLE = "E-Commerce LLM Comparison"
APP_ICON = "🛒"
GITHUB_URL = "https://github.com/nikhilsanghi/ciq"
