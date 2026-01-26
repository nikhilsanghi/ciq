"""
Post-training quantization utilities for inference optimization.

This module provides utilities for quantizing models using AWQ and GPTQ methods.
Quantization reduces model size and increases inference speed with minimal
quality loss, making it essential for production deployment.

Quantization Method Comparison:
-----------------------------
| Method | Bits | Speed  | Quality | Use Case                          |
|--------|------|--------|---------|-----------------------------------|
| FP16   | 16   | 1x     | Best    | Training, high-quality inference  |
| INT8   | 8    | 1.5x   | Good    | Balanced speed/quality            |
| AWQ    | 4    | 2-3x   | Good    | Instruction-tuned models          |
| GPTQ   | 4    | 2-3x   | Good    | General models, wider support     |

Note: This module is OPTIONAL for the learning project. The main inference
pipeline works with bitsandbytes on-the-fly quantization. AWQ/GPTQ are
for production optimization when you need pre-quantized model weights.

Example usage:
    >>> from src.inference.quantization import quantize_awq
    >>> quantize_awq(
    ...     model_path="./outputs/merged-model",
    ...     output_path="./outputs/model-awq",
    ...     w_bit=4
    ... )
"""

from typing import Optional, Dict, Any, List
import logging
import os

logger = logging.getLogger(__name__)


def quantize_awq(
    model_path: str,
    output_path: str,
    w_bit: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    version: str = "GEMM",
    calib_data: Optional[str] = None,
    calib_samples: int = 128,
    calib_seq_len: int = 512,
) -> None:
    """
    Quantize a model using AWQ (Activation-aware Weight Quantization).

    AWQ is particularly effective for instruction-tuned models because it
    preserves the weights that are most important for following instructions.
    It analyzes activation patterns to identify salient weights and protects
    them during quantization.

    How AWQ Works:
    1. Runs calibration data through the model to measure activations
    2. Identifies weights with high activation magnitude (salient weights)
    3. Scales salient weights UP before quantization (making them larger)
    4. Quantizes all weights to target bit-width
    5. Scales are stored and applied during inference to recover precision

    Args:
        model_path: Path to the model to quantize (HuggingFace format).
            Should be a merged model (not LoRA adapter).
        output_path: Path to save the quantized model.
        w_bit: Bit-width for weight quantization (typically 4).
        group_size: Number of weights that share quantization parameters.
            Smaller = better quality, larger model. Default 128 is balanced.
        zero_point: Whether to use asymmetric quantization (with zero point).
            True = better for weights with non-symmetric distributions.
        version: AWQ kernel version.
            - "GEMM": Optimized for batch size >= 8
            - "GEMV": Optimized for batch size = 1
            Default "GEMM" is better for most inference scenarios.
        calib_data: Dataset for calibration. Options:
            - None: Uses default "pileval" dataset
            - "c4": C4 dataset (general text)
            - "wikitext2": Wikipedia text
            - Path to custom JSONL file
        calib_samples: Number of calibration samples to use.
        calib_seq_len: Sequence length for calibration samples.

    Returns:
        None. Saves quantized model to output_path.

    Example:
        >>> # Quantize a merged fine-tuned model
        >>> quantize_awq(
        ...     model_path="./outputs/ecommerce-merged",
        ...     output_path="./outputs/ecommerce-awq",
        ...     w_bit=4,
        ...     group_size=128,
        ... )

        >>> # Load quantized model for inference
        >>> from awq import AutoAWQForCausalLM
        >>> model = AutoAWQForCausalLM.from_quantized("./outputs/ecommerce-awq")

    When to Use AWQ:
        - Instruction-tuned models (Mistral-Instruct, Llama-Instruct, etc.)
        - Models where you need best quality at 4-bit
        - When you have time to run calibration (few minutes)
        - vLLM deployment (native AWQ support)

    When NOT to Use AWQ:
        - Base models (GPTQ may be equally good)
        - When you need instant quantization (use bitsandbytes instead)
        - When target platform doesn't support AWQ kernels

    Notes:
        - Requires autoawq package: pip install autoawq
        - Calibration takes 5-15 minutes depending on model size
        - Output model is ~2-3GB for 7B model (vs ~14GB FP16)
        - Compatible with vLLM, TGI, and HuggingFace transformers
    """
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "AWQ quantization requires autoawq package. "
            "Install with: pip install autoawq"
        )

    logger.info(f"Loading model from {model_path} for AWQ quantization")

    # Load model and tokenizer
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Quantization configuration
    quant_config = {
        "w_bit": w_bit,
        "q_group_size": group_size,
        "zero_point": zero_point,
        "version": version,
    }

    logger.info(f"Starting AWQ quantization with config: {quant_config}")
    logger.info(f"Using {calib_samples} calibration samples of length {calib_seq_len}")

    # Run quantization with calibration
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data or "pileval",
        split="train",
        text_column="text",
        duo_scaling=True,  # Improves quality
        export_compatible=True,  # For vLLM compatibility
    )

    # Save quantized model
    logger.info(f"Saving quantized model to {output_path}")
    os.makedirs(output_path, exist_ok=True)

    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info("AWQ quantization complete!")

    # Log size comparison
    _log_size_comparison(model_path, output_path)


def quantize_gptq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    dataset: str = "c4",
    num_samples: int = 128,
    seq_len: int = 2048,
    use_exllama: bool = True,
    use_triton: bool = False,
    desc_act: bool = False,
    sym: bool = True,
    true_sequential: bool = True,
    batch_size: int = 1,
) -> None:
    """
    Quantize a model using GPTQ (Generative Post-Training Quantization).

    GPTQ uses a calibration dataset to minimize the quantization error
    through layer-wise optimization. It's the most widely supported
    quantization method with excellent ecosystem support.

    How GPTQ Works:
    1. Processes model layer by layer
    2. For each layer, uses calibration data to compute Hessian matrix
    3. Quantizes weights while minimizing squared error (using Hessian)
    4. Uses "Optimal Brain Quantization" algorithm for weight selection
    5. Compensates remaining weights to reduce accumulated error

    Args:
        model_path: Path to the model to quantize (HuggingFace format).
        output_path: Path to save the quantized model.
        bits: Bit-width for quantization (2, 3, 4, or 8).
        group_size: Number of weights per quantization group.
            - 128: Balanced (default, recommended)
            - 64: Better quality, slightly larger
            - 32: Best quality, largest size
            - -1: Per-column quantization (smallest, lowest quality)
        dataset: Calibration dataset name or path.
            - "c4": C4 dataset (recommended for general use)
            - "wikitext2": Wikipedia text
            - "ptb": Penn Treebank
            - Path to custom dataset
        num_samples: Number of calibration samples.
        seq_len: Sequence length for calibration.
        use_exllama: Use ExLlama kernels for faster inference.
            Only works on NVIDIA GPUs with compute capability >= 8.0.
        use_triton: Use Triton kernels (alternative to ExLlama).
        desc_act: Use descending activation order (can improve quality).
            Warning: Significantly slower inference.
        sym: Use symmetric quantization (recommended for most models).
        true_sequential: Process layers in true sequential order.
            Improves quality but increases quantization time.
        batch_size: Batch size for calibration (reduce if OOM).

    Returns:
        None. Saves quantized model to output_path.

    Example:
        >>> # Standard GPTQ quantization
        >>> quantize_gptq(
        ...     model_path="./outputs/ecommerce-merged",
        ...     output_path="./outputs/ecommerce-gptq",
        ...     bits=4,
        ...     group_size=128,
        ...     dataset="c4",
        ... )

        >>> # Load for inference
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "./outputs/ecommerce-gptq",
        ...     device_map="auto",
        ... )

    When to Use GPTQ:
        - Need widest compatibility (most frameworks support GPTQ)
        - Base models (not instruction-tuned)
        - ExLlama/ExLlamaV2 deployment (fastest GPTQ inference)
        - llama.cpp or GGML conversion target

    When NOT to Use GPTQ:
        - Instruction-tuned models (AWQ may be better)
        - Need fastest quantization (GPTQ is slower than AWQ)
        - vLLM deployment (AWQ has better vLLM integration)

    Notes:
        - Requires auto-gptq: pip install auto-gptq
        - Quantization takes 15-45 minutes for 7B model
        - ExLlama kernels provide fastest inference on supported GPUs
        - Compatible with most inference frameworks
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
    except ImportError:
        raise ImportError(
            "GPTQ quantization requires transformers>=4.32.0 and auto-gptq. "
            "Install with: pip install transformers auto-gptq optimum"
        )

    logger.info(f"Loading model from {model_path} for GPTQ quantization")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure GPTQ
    gptq_config = GPTQConfig(
        bits=bits,
        group_size=group_size,
        dataset=dataset,
        desc_act=desc_act,
        sym=sym,
        true_sequential=true_sequential,
        use_exllama=use_exllama,
        model_seqlen=seq_len,
    )

    logger.info(f"Starting GPTQ quantization: {bits}-bit, group_size={group_size}")
    logger.info(f"Using {num_samples} calibration samples from '{dataset}'")

    # Load model with quantization config
    # This triggers quantization during loading
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=gptq_config,
    )

    # Save quantized model
    logger.info(f"Saving quantized model to {output_path}")
    os.makedirs(output_path, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info("GPTQ quantization complete!")

    # Log size comparison
    _log_size_comparison(model_path, output_path)


def quantize_bnb_nf4(
    model_path: str,
    output_path: str,
    double_quant: bool = True,
    compute_dtype: str = "float16",
) -> None:
    """
    Apply bitsandbytes NF4 quantization and save.

    Unlike AWQ/GPTQ, bitsandbytes quantization is typically applied at load
    time. This function loads a model with NF4 quantization and saves it
    in a format that preserves the quantization for faster loading.

    Note: This is less common than AWQ/GPTQ for pre-quantized models.
    Usually you just pass quantization="4bit" to load_model() at runtime.

    Args:
        model_path: Path to the model to quantize.
        output_path: Path to save the quantized model.
        double_quant: Use double quantization for extra memory savings.
        compute_dtype: Data type for computations ("float16" or "bfloat16").

    Example:
        >>> quantize_bnb_nf4(
        ...     model_path="./outputs/merged-model",
        ...     output_path="./outputs/model-bnb4",
        ... )

    Notes:
        - bitsandbytes quantization is dynamic (computed at load time)
        - For truly pre-quantized models, use AWQ or GPTQ instead
        - This is mainly useful for saving a pre-configured model state
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
    except ImportError:
        raise ImportError("Requires transformers and bitsandbytes packages")

    logger.info(f"Loading model with NF4 quantization from {model_path}")

    # Configure quantization
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype_map.get(compute_dtype, torch.float16),
        bnb_4bit_use_double_quant=double_quant,
    )

    # Load with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Save
    logger.info(f"Saving to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info("NF4 quantization complete!")


def compare_quantization_methods() -> Dict[str, Any]:
    """
    Returns a comprehensive comparison of quantization methods.

    This function provides a detailed comparison table useful for:
    - Interview preparation
    - Choosing the right method for your use case
    - Understanding tradeoffs

    Returns:
        Dictionary containing comparison data and recommendations.

    Example:
        >>> comparison = compare_quantization_methods()
        >>> print(comparison["summary_table"])
        >>> print(comparison["recommendations"])
    """
    comparison = {
        "summary_table": """
        +-----------+------+---------+----------+---------------+------------------+
        | Method    | Bits | Quality | Speed    | Memory (7B)   | Best For         |
        +-----------+------+---------+----------+---------------+------------------+
        | FP32      | 32   | 100%    | 1.0x     | ~28 GB        | Training only    |
        | FP16      | 16   | ~100%   | 1.0x     | ~14 GB        | High-quality inf |
        | BF16      | 16   | ~100%   | 1.0x     | ~14 GB        | Training + inf   |
        | INT8      | 8    | ~99%    | 1.5x     | ~7 GB         | Balanced         |
        | FP8       | 8    | ~99%    | 2.0x     | ~7 GB         | Hopper GPUs      |
        | NF4 (bnb) | 4    | ~97%    | 0.8x     | ~4 GB         | Dev/fine-tuning  |
        | AWQ       | 4    | ~98%    | 2-3x     | ~4 GB         | Instruct models  |
        | GPTQ      | 4    | ~97%    | 2-3x     | ~4 GB         | General models   |
        | GGUF Q4   | 4    | ~95%    | Varies   | ~4 GB         | CPU inference    |
        +-----------+------+---------+----------+---------------+------------------+

        Notes:
        - Quality is approximate perplexity retention on common benchmarks
        - Speed is relative throughput compared to FP16 baseline
        - Memory is approximate for 7B parameter model
        """,

        "methods": {
            "fp16": {
                "bits": 16,
                "quality_retention": "~100%",
                "speed_multiplier": 1.0,
                "memory_7b_gb": 14,
                "pros": [
                    "No quality loss",
                    "Universal support",
                    "No calibration needed",
                ],
                "cons": [
                    "Highest memory usage",
                    "May not fit on consumer GPUs",
                ],
                "best_for": "High-quality inference where memory is available",
                "frameworks": ["All"],
            },

            "int8_smoothquant": {
                "bits": 8,
                "quality_retention": "~99%",
                "speed_multiplier": 1.5,
                "memory_7b_gb": 7,
                "pros": [
                    "Good quality retention",
                    "Faster than FP16",
                    "Wide framework support",
                ],
                "cons": [
                    "Some quality loss on edge cases",
                    "Requires SmoothQuant for best results",
                ],
                "best_for": "Production serving with quality priority",
                "frameworks": ["vLLM", "TensorRT-LLM", "TGI"],
            },

            "fp8": {
                "bits": 8,
                "quality_retention": "~99%",
                "speed_multiplier": 2.0,
                "memory_7b_gb": 7,
                "pros": [
                    "Best speed/quality ratio",
                    "Native hardware support on Hopper",
                    "Simple to apply",
                ],
                "cons": [
                    "Requires H100/H200 or Ada GPUs",
                    "Limited framework support",
                ],
                "best_for": "Production on Hopper/Ada GPUs",
                "frameworks": ["TensorRT-LLM", "vLLM (experimental)"],
            },

            "nf4_bitsandbytes": {
                "bits": 4,
                "quality_retention": "~97%",
                "speed_multiplier": 0.8,  # Can be slower due to dequantization
                "memory_7b_gb": 4,
                "pros": [
                    "Lowest memory usage",
                    "Easy to use (just add config)",
                    "Great for fine-tuning (QLoRA)",
                    "No calibration needed",
                ],
                "cons": [
                    "Slower inference than AWQ/GPTQ",
                    "Quality loss noticeable",
                    "Not ideal for serving",
                ],
                "best_for": "Development, fine-tuning, memory-constrained experimentation",
                "frameworks": ["HuggingFace Transformers"],
            },

            "awq": {
                "bits": 4,
                "quality_retention": "~98%",
                "speed_multiplier": 2.5,
                "memory_7b_gb": 4,
                "pros": [
                    "Best quality at 4-bit",
                    "Preserves instruction-following",
                    "Fast inference with custom kernels",
                    "Good vLLM integration",
                ],
                "cons": [
                    "Requires calibration",
                    "Smaller ecosystem than GPTQ",
                    "Some models not supported",
                ],
                "best_for": "Instruction-tuned models in production",
                "frameworks": ["vLLM", "TGI", "AutoAWQ", "Transformers"],
            },

            "gptq": {
                "bits": 4,
                "quality_retention": "~97%",
                "speed_multiplier": 2.5,
                "memory_7b_gb": 4,
                "pros": [
                    "Widest ecosystem support",
                    "ExLlama provides fastest inference",
                    "Easy to convert to GGUF",
                    "Many pre-quantized models available",
                ],
                "cons": [
                    "Calibration takes longer than AWQ",
                    "May lose more quality on instruct models",
                ],
                "best_for": "General models, wide deployment compatibility",
                "frameworks": ["ExLlama", "vLLM", "TGI", "llama.cpp", "Transformers"],
            },

            "gguf": {
                "bits": "2-8 (configurable)",
                "quality_retention": "Varies (95-99%)",
                "speed_multiplier": "Varies",
                "memory_7b_gb": "2-7 GB",
                "pros": [
                    "CPU inference support",
                    "Flexible bit-width per layer",
                    "Works on consumer hardware",
                    "Active community",
                ],
                "cons": [
                    "Slower than GPU quantization",
                    "Quality varies with quantization level",
                ],
                "best_for": "CPU inference, edge devices, consumer hardware",
                "frameworks": ["llama.cpp", "Ollama", "LM Studio"],
            },
        },

        "recommendations": {
            "production_high_throughput": {
                "first_choice": "FP8 (if H100/H200 available)",
                "second_choice": "INT8 SmoothQuant",
                "third_choice": "AWQ 4-bit",
                "reason": "FP8 provides best speed with minimal quality loss on modern GPUs",
            },

            "production_memory_constrained": {
                "first_choice": "AWQ 4-bit (for instruct models)",
                "second_choice": "GPTQ 4-bit (for base models)",
                "reason": "4-bit quantization fits 7B models in 8GB VRAM while maintaining quality",
            },

            "development_experimentation": {
                "first_choice": "bitsandbytes NF4",
                "reason": "No calibration needed, easy to use, good for iteration",
            },

            "fine_tuning": {
                "first_choice": "QLoRA with NF4",
                "reason": "Only method that supports training with 4-bit base model",
            },

            "edge_deployment": {
                "first_choice": "GGUF Q4_K_M",
                "reason": "Best balance of quality and size for CPU inference",
            },

            "interview_answer": """
            For production LLM serving, I would recommend a tiered approach:

            1. **If using H100/H200 GPUs**: FP8 quantization provides ~2x throughput
               improvement with <1% quality loss. This is the new standard for
               high-performance serving.

            2. **If using A100 or older GPUs**: Start with INT8 SmoothQuant for
               balanced speed/quality. If memory is constrained, use AWQ 4-bit
               for instruction-tuned models or GPTQ for base models.

            3. **For development/fine-tuning**: Use bitsandbytes NF4 with QLoRA.
               It's the only method supporting efficient fine-tuning at 4-bit.

            4. **For serving frameworks**: vLLM with AWQ provides excellent
               throughput through continuous batching + PagedAttention +
               efficient kernels.

            Key insight: The best quantization method depends on your model type
            (instruct vs base), hardware (GPU generation), and use case
            (training vs inference vs serving).
            """,
        },

        "quality_benchmarks": {
            "note": "Quality retention varies by task. Numbers below are approximate.",
            "classification_f1_retention": {
                "fp16": "100%",
                "int8": "99.5%",
                "awq_4bit": "98.5%",
                "gptq_4bit": "97.5%",
                "nf4": "96.5%",
            },
            "generation_perplexity_increase": {
                "fp16": "+0%",
                "int8": "+0.5%",
                "awq_4bit": "+2%",
                "gptq_4bit": "+3%",
                "nf4": "+4%",
            },
        },
    }

    return comparison


def estimate_model_size(
    num_parameters: int,
    quantization: Optional[str] = None,
) -> Dict[str, float]:
    """
    Estimate model size in GB for different quantization methods.

    Args:
        num_parameters: Number of model parameters (e.g., 7_000_000_000 for 7B).
        quantization: Specific method to calculate, or None for all methods.

    Returns:
        Dictionary of method -> size in GB.

    Example:
        >>> sizes = estimate_model_size(7_000_000_000)
        >>> print(sizes)
        {'fp32': 28.0, 'fp16': 14.0, 'int8': 7.0, '4bit': 3.5}
    """
    # Bytes per parameter for each method
    bytes_per_param = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "int8": 1.0,
        "fp8": 1.0,
        "4bit": 0.5,  # AWQ, GPTQ, NF4
    }

    # Additional overhead (quantization parameters, etc.)
    overhead_factor = {
        "fp32": 1.0,
        "fp16": 1.0,
        "bf16": 1.0,
        "int8": 1.05,  # Scales and zeros
        "fp8": 1.02,
        "4bit": 1.1,   # Group quantization parameters
    }

    sizes = {}
    for method, bytes_p in bytes_per_param.items():
        if quantization is None or quantization == method:
            size_gb = (num_parameters * bytes_p * overhead_factor[method]) / (1024**3)
            sizes[method] = round(size_gb, 2)

    return sizes


def _log_size_comparison(original_path: str, quantized_path: str) -> None:
    """Log size comparison between original and quantized models."""
    import glob

    def get_dir_size(path):
        total = 0
        for f in glob.glob(os.path.join(path, "**", "*"), recursive=True):
            if os.path.isfile(f):
                total += os.path.getsize(f)
        return total

    try:
        original_size = get_dir_size(original_path)
        quantized_size = get_dir_size(quantized_path)

        reduction = (1 - quantized_size / original_size) * 100

        logger.info(f"Size comparison:")
        logger.info(f"  Original:  {original_size / (1024**3):.2f} GB")
        logger.info(f"  Quantized: {quantized_size / (1024**3):.2f} GB")
        logger.info(f"  Reduction: {reduction:.1f}%")
    except Exception as e:
        logger.warning(f"Could not calculate size comparison: {e}")


def validate_quantized_model(
    model_path: str,
    test_prompt: str = "Hello, how are you?",
    expected_output_contains: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate that a quantized model loads and generates correctly.

    Args:
        model_path: Path to the quantized model.
        test_prompt: Simple prompt to test generation.
        expected_output_contains: Optional string that output should contain.

    Returns:
        Dictionary with validation results.

    Example:
        >>> result = validate_quantized_model("./outputs/model-awq")
        >>> print(result["success"])  # True if model works
        >>> print(result["output"])   # Generated text
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    result = {
        "success": False,
        "model_path": model_path,
        "error": None,
        "output": None,
        "load_time_seconds": None,
        "generation_time_seconds": None,
    }

    try:
        import time

        # Test loading
        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        result["load_time_seconds"] = round(time.time() - start, 2)

        # Test generation
        start = time.time()
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=32)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result["generation_time_seconds"] = round(time.time() - start, 2)
        result["output"] = output_text

        # Validate output if expected string provided
        if expected_output_contains:
            if expected_output_contains.lower() not in output_text.lower():
                result["error"] = f"Output did not contain '{expected_output_contains}'"
                return result

        result["success"] = True
        logger.info(f"Quantized model validation successful: {model_path}")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Quantized model validation failed: {e}")

    return result
