"""
Inference utilities for model loading, quantization, and e-commerce tasks.

This module provides:
- Model loading with LoRA adapter support
- AWQ/GPTQ post-training quantization
- High-level inference classes for classification, extraction, and Q&A
- vLLM integration for high-throughput serving

Example usage:
    >>> from src.inference import EcommerceInference
    >>> inference = EcommerceInference("mistralai/Mistral-7B-Instruct-v0.3")
    >>> category = inference.classify("Apple iPhone 15 Pro 256GB")
"""

from .model import (
    load_model,
    merge_lora_weights,
    generate_response,
    batch_generate,
    EcommerceInference,
    VLLMInference,
)
from .quantization import (
    quantize_awq,
    quantize_gptq,
    quantize_bnb_nf4,
    compare_quantization_methods,
    estimate_model_size,
    validate_quantized_model,
)

__all__ = [
    # Model loading
    "load_model",
    "merge_lora_weights",
    "generate_response",
    "batch_generate",
    # High-level inference
    "EcommerceInference",
    "VLLMInference",
    # Quantization
    "quantize_awq",
    "quantize_gptq",
    "quantize_bnb_nf4",
    "compare_quantization_methods",
    "estimate_model_size",
    "validate_quantized_model",
]
