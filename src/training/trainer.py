"""
QLoRA training script for e-commerce LLM.

This module provides a complete training pipeline for fine-tuning large language
models on e-commerce tasks using QLoRA (Quantized Low-Rank Adaptation).

Supported Models:
- Mistral-7B-Instruct-v0.3 (recommended)
- Meta-Llama-3-8B-Instruct
- Any HuggingFace causal LM model

Key Features:
1. QLoRA fine-tuning for memory efficiency
2. Gradient checkpointing for VRAM optimization
3. Checkpoint saving for spot instance recovery
4. Wandb/TensorBoard logging integration
5. Multi-task support (classification, extraction, Q&A)

Training Pipeline Overview:
--------------------------
1. Load and quantize base model (4-bit NF4)
2. Configure LoRA adapters on target modules
3. Prepare tokenizer with proper padding
4. Format dataset with task-specific prompts
5. Train with SFTTrainer (Supervised Fine-Tuning)
6. Save adapters and optionally merge with base model

Usage:
------
CLI:
    python -m src.training.trainer \\
        --model_name mistralai/Mistral-7B-Instruct-v0.3 \\
        --train_data ./data/train.jsonl \\
        --output_dir ./outputs/my-model

Python:
    from src.training.trainer import train_model
    from src.training.config import TrainingConfig

    config = TrainingConfig(model_name="mistralai/Mistral-7B-Instruct-v0.3")
    trainer = train_model(config, train_dataset, eval_dataset)
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime

import torch
from datasets import Dataset, load_dataset

from .config import (
    TrainingConfig,
    get_qlora_config,
    get_lora_config,
    load_config_from_yaml,
    save_config_to_yaml,
    get_preset_config,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and return device information.

    WHY THIS MATTERS:
    -----------------
    QLoRA training requires a CUDA-capable GPU. This function verifies
    GPU availability and reports useful diagnostics before training starts.

    Returns:
        Dict containing:
        - cuda_available: Whether CUDA is available
        - device_count: Number of GPUs
        - device_name: Name of the first GPU
        - vram_total: Total VRAM in GB
        - vram_free: Free VRAM in GB (approximate)
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "device_name": None,
        "vram_total_gb": 0,
        "vram_free_gb": 0,
        "bf16_supported": False,
    }

    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)

        # Get memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory
        info["vram_total_gb"] = total_memory / (1024**3)

        # Check free memory
        torch.cuda.empty_cache()
        free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        info["vram_free_gb"] = free_memory / (1024**3)

        # Check bfloat16 support (Ampere and newer)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()

    return info


def load_base_model(
    model_name: str,
    quantization_config,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    attn_implementation: Optional[str] = None,
):
    """
    Load and quantize the base language model.

    WHY THESE SETTINGS:
    -------------------
    1. quantization_config: Enables 4-bit loading, reducing VRAM from ~14GB to ~4GB
    2. device_map="auto": Automatically places model layers across available devices
    3. trust_remote_code: Required for some models (Mistral, LLaMA) that have
       custom code in their repositories
    4. attn_implementation: "flash_attention_2" speeds up training 2-3x on
       supported GPUs (Ampere+). Falls back to standard attention if unavailable.

    Args:
        model_name: HuggingFace model ID or local path
        quantization_config: BitsAndBytesConfig for 4-bit quantization
        device_map: Device placement strategy ("auto", "cuda:0", etc.)
        trust_remote_code: Allow executing model's custom code
        attn_implementation: "flash_attention_2" for faster attention

    Returns:
        The loaded and quantized model.

    Raises:
        RuntimeError: If CUDA is not available
        ValueError: If model cannot be loaded
    """
    from transformers import AutoModelForCausalLM

    # Verify CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for QLoRA training. "
            "Please run on a machine with a CUDA-capable GPU."
        )

    logger.info(f"Loading base model: {model_name}")
    logger.info(f"Quantization: 4-bit NF4 with double quantization")

    # Determine attention implementation
    # Flash Attention 2 provides significant speedup but requires:
    # 1. Ampere or newer GPU (A10, A100, RTX 3090+)
    # 2. flash-attn package installed
    if attn_implementation is None:
        try:
            import flash_attn
            attn_implementation = "flash_attention_2"
            logger.info("Flash Attention 2 detected - enabling for faster training")
        except ImportError:
            attn_implementation = "eager"
            logger.info(
                "Flash Attention 2 not installed. "
                "Consider installing for 2-3x speedup: pip install flash-attn"
            )

    # Load model with quantization
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            # torch_dtype is handled by quantization_config
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise ValueError(f"Could not load model '{model_name}': {e}")

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded successfully")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} (before LoRA)")

    return model


def setup_tokenizer(
    model_name: str,
    max_seq_length: int = 2048,
    padding_side: str = "right",
    trust_remote_code: bool = True,
):
    """
    Initialize and configure the tokenizer.

    WHY THESE SETTINGS:
    -------------------
    1. padding_side="right": For causal LM training, we pad on the right
       so that the model sees the actual content first. Left padding is
       used for generation/inference.

    2. pad_token = eos_token: Many models (LLaMA, Mistral) don't have a
       dedicated pad token. Using EOS as pad works because:
       - Padded positions are masked in attention anyway
       - EOS is a natural "nothing here" signal

    3. model_max_length: Prevents accidentally creating sequences longer
       than the model can handle. Truncation happens automatically.

    Args:
        model_name: HuggingFace model ID or local path
        max_seq_length: Maximum sequence length for training
        padding_side: "right" for training, "left" for generation
        trust_remote_code: Allow executing model's custom code

    Returns:
        Configured tokenizer.
    """
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer for: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        model_max_length=max_seq_length,
    )

    # Configure padding
    # WHY: Causal LM training needs right padding so the model sees actual
    # content first. The padding tokens at the end are masked during loss computation.
    tokenizer.padding_side = padding_side

    # Handle missing pad token
    # WHY: Many modern LLMs (LLaMA, Mistral) don't define a pad token.
    # Using EOS as pad is safe because padding positions are masked in attention.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set pad_token to eos_token: '{tokenizer.eos_token}'")

    logger.info(f"Tokenizer configured:")
    logger.info(f"  Vocab size: {tokenizer.vocab_size:,}")
    logger.info(f"  Max length: {tokenizer.model_max_length}")
    logger.info(f"  Padding side: {tokenizer.padding_side}")

    return tokenizer


def prepare_model_for_training(
    model,
    lora_config,
    gradient_checkpointing: bool = True,
):
    """
    Prepare the quantized model for LoRA training.

    WHY THESE STEPS:
    ----------------
    1. prepare_model_for_kbit_training(): Enables gradient computation for
       the frozen quantized weights. Without this, gradients don't flow
       through the 4-bit layers properly.

    2. gradient_checkpointing: Trades compute for memory by not storing
       intermediate activations. Instead, they're recomputed during
       backward pass. Saves ~50% VRAM at cost of ~30% slower training.

    3. get_peft_model(): Wraps the base model with LoRA adapters. Only
       the adapter weights are trainable; base weights stay frozen.

    Args:
        model: The quantized base model
        lora_config: PEFT LoraConfig specifying adapter configuration
        gradient_checkpointing: Enable gradient checkpointing for memory savings

    Returns:
        Model with LoRA adapters ready for training.
    """
    from peft import prepare_model_for_kbit_training, get_peft_model

    logger.info("Preparing model for LoRA training...")

    # Step 1: Prepare for k-bit training
    # This enables gradient computation through the quantized layers
    # Note: use_gradient_checkpointing parameter was removed in peft 0.13+
    model = prepare_model_for_kbit_training(model)

    # Step 2: Enable gradient checkpointing if requested
    # WHY: Essential for fitting 7B models on 12GB GPUs
    # Note: enable_input_require_grads() is deprecated and no longer needed
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled (saves ~50% VRAM)")

    # Step 3: Add LoRA adapters
    model = get_peft_model(model, lora_config)

    # Log trainable parameters
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    logger.info(f"LoRA adapters added:")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable %: {trainable_pct:.2f}%")

    return model


def format_prompt(
    example: Dict[str, Any],
    task_type: str = "classify",
    tokenizer=None,
) -> str:
    """
    Format a training example into the expected prompt format.

    WHY THIS FORMAT:
    ----------------
    We use task-specific prefixes ([CLASSIFY], [EXTRACT], [QA]) to help
    the model distinguish between different task types. This multi-task
    approach allows a single model to handle all e-commerce tasks.

    The format follows the chat template of the base model (Mistral/LLaMA)
    to maintain compatibility with the model's pre-training.

    Args:
        example: Dictionary containing the training example
        task_type: One of "classify", "extract", "qa"
        tokenizer: Tokenizer for applying chat template

    Returns:
        Formatted prompt string.
    """
    task_prefixes = {
        "classify": "[CLASSIFY]",
        "extract": "[EXTRACT]",
        "qa": "[QA]",
    }

    prefix = task_prefixes.get(task_type, "")

    # Build the prompt based on task type
    if task_type == "classify":
        # Product classification task
        user_content = f"{prefix} Classify the following product into the Google Product Taxonomy.\n\nProduct: {example.get('product_title', example.get('text', ''))}"
        if "product_description" in example:
            user_content += f"\n\nDescription: {example['product_description']}"
        assistant_content = example.get("category", example.get("label", ""))

    elif task_type == "extract":
        # Attribute extraction task
        user_content = f"{prefix} Extract product attributes as JSON.\n\nProduct: {example.get('product_title', example.get('text', ''))}"
        if "product_description" in example:
            user_content += f"\n\nDescription: {example['product_description']}"
        assistant_content = example.get("attributes", example.get("output", "{}"))

    elif task_type == "qa":
        # Question answering task
        product_context = example.get("product_context", example.get("context", ""))
        question = example.get("question", "")
        user_content = f"{prefix} Answer the question about this product.\n\nProduct: {product_context}\n\nQuestion: {question}"
        assistant_content = example.get("answer", "")

    else:
        # Generic format
        user_content = example.get("input", example.get("text", ""))
        assistant_content = example.get("output", example.get("label", ""))

    # Format as chat messages
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    # Apply chat template if tokenizer is provided
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        # Fallback format
        return f"<|user|>\n{user_content}\n<|assistant|>\n{assistant_content}"


def prepare_dataset(
    data_path: str,
    tokenizer,
    task_type: str = "classify",
    max_samples: Optional[int] = None,
    text_column: str = "text",
) -> Dataset:
    """
    Load and prepare the training dataset.

    WHY THIS APPROACH:
    ------------------
    We support multiple data formats (JSONL, CSV, Parquet, HuggingFace datasets)
    to accommodate different data sources. The formatting function converts
    each example into the expected prompt format for SFT training.

    Args:
        data_path: Path to data file or HuggingFace dataset ID
        tokenizer: Tokenizer for formatting prompts
        task_type: Task type for prompt formatting
        max_samples: Optional limit on number of samples
        text_column: Column name for the formatted text (for SFTTrainer)

    Returns:
        Prepared Dataset ready for training.
    """
    logger.info(f"Loading dataset from: {data_path}")

    # Determine data format and load
    if data_path.endswith(".jsonl") or data_path.endswith(".json"):
        dataset = load_dataset("json", data_files=data_path, split="train")
    elif data_path.endswith(".csv"):
        dataset = load_dataset("csv", data_files=data_path, split="train")
    elif data_path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=data_path, split="train")
    else:
        # Assume it's a HuggingFace dataset ID
        dataset = load_dataset(data_path, split="train")

    logger.info(f"Loaded {len(dataset)} examples")

    # Limit samples if specified
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        logger.info(f"Limited to {max_samples} examples")

    # Format each example
    def formatting_function(example):
        return {text_column: format_prompt(example, task_type, tokenizer)}

    dataset = dataset.map(
        formatting_function,
        remove_columns=dataset.column_names,
        desc="Formatting dataset",
    )

    logger.info(f"Dataset prepared with {len(dataset)} examples")
    return dataset


def train_model(
    config: TrainingConfig,
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Dataset] = None,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Execute the full training pipeline.

    This is the main training function that orchestrates:
    1. Model loading and quantization
    2. LoRA adapter setup
    3. Dataset preparation
    4. Training with SFTTrainer
    5. Checkpoint saving

    WHY SFTTrainer:
    ---------------
    SFTTrainer (Supervised Fine-Tuning Trainer) from TRL library is
    specifically designed for instruction fine-tuning. It handles:
    - Proper loss masking (only compute loss on assistant responses)
    - Efficient packing of short examples
    - Integration with PEFT/LoRA
    - Automatic handling of chat templates

    Args:
        config: TrainingConfig with all hyperparameters
        train_dataset: Pre-loaded training dataset (optional)
        eval_dataset: Pre-loaded evaluation dataset (optional)
        resume_from_checkpoint: Path to checkpoint for resuming training

    Returns:
        The trained SFTTrainer object.

    Example:
        >>> config = TrainingConfig(
        ...     model_name="mistralai/Mistral-7B-Instruct-v0.3",
        ...     output_dir="./outputs/my-model",
        ... )
        >>> trainer = train_model(config)
        >>> trainer.save_model("./final-model")
    """
    from trl import SFTTrainer, SFTConfig

    # Check GPU availability
    gpu_info = check_gpu_availability()
    if not gpu_info["cuda_available"]:
        raise RuntimeError("CUDA is required for QLoRA training")

    logger.info("=" * 60)
    logger.info("Starting QLoRA Training")
    logger.info("=" * 60)
    logger.info(f"GPU: {gpu_info['device_name']}")
    logger.info(f"VRAM: {gpu_info['vram_total_gb']:.1f} GB")
    logger.info(f"BF16 Supported: {gpu_info['bf16_supported']}")
    logger.info("=" * 60)

    # Estimate VRAM usage
    vram_estimate = config.estimate_vram_usage()
    logger.info("Estimated VRAM Usage:")
    for key, value in vram_estimate.items():
        logger.info(f"  {key}: {value:.2f} GB")

    if vram_estimate["total_estimated"] > gpu_info["vram_total_gb"]:
        logger.warning(
            f"Estimated VRAM ({vram_estimate['total_estimated']:.1f}GB) exceeds "
            f"available ({gpu_info['vram_total_gb']:.1f}GB). Training may fail or be slow."
        )

    # Initialize Wandb if configured
    if config.report_to == "wandb":
        try:
            import wandb

            run_name = config.wandb_run_name or f"ecommerce-llm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project=config.wandb_project,
                name=run_name,
                config={
                    "model_name": config.model_name,
                    "lora_r": config.lora.lora_r,
                    "lora_alpha": config.lora.lora_alpha,
                    "learning_rate": config.learning_rate,
                    "epochs": config.num_train_epochs,
                    "batch_size": config.get_effective_batch_size(),
                },
            )
            logger.info(f"Wandb initialized: {config.wandb_project}/{run_name}")
        except ImportError:
            logger.warning("Wandb not installed. Falling back to tensorboard.")
            config.report_to = "tensorboard"

    # Step 1: Load quantization config
    logger.info("Step 1/6: Loading quantization configuration...")
    quant_config = config.quantization.to_bits_and_bytes_config()

    # Step 2: Load base model
    logger.info("Step 2/6: Loading and quantizing base model...")
    model = load_base_model(
        config.model_name,
        quant_config,
    )

    # Step 3: Setup tokenizer
    logger.info("Step 3/6: Setting up tokenizer...")
    tokenizer = setup_tokenizer(
        config.model_name,
        max_seq_length=config.max_seq_length,
    )

    # Step 4: Prepare model with LoRA
    logger.info("Step 4/6: Adding LoRA adapters...")
    lora_config = config.lora.to_peft_config()
    model = prepare_model_for_training(
        model,
        lora_config,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    # Step 5: Load datasets if not provided
    logger.info("Step 5/6: Preparing datasets...")
    if train_dataset is None and config.train_data_path:
        train_dataset = prepare_dataset(
            config.train_data_path,
            tokenizer,
        )

    if eval_dataset is None and config.eval_data_path:
        eval_dataset = prepare_dataset(
            config.eval_data_path,
            tokenizer,
        )

    if train_dataset is None:
        raise ValueError(
            "No training data provided. Either pass train_dataset or set train_data_path in config."
        )

    # Step 6: Create trainer and train
    logger.info("Step 6/6: Initializing SFTTrainer...")

    # Get SFTConfig (TRL 0.12+ API - combines TrainingArguments with SFT-specific params)
    sft_config = config.to_sft_config(
        dataset_text_field="text",  # Column containing formatted prompts
        packing=False,  # Disable packing for better loss computation
    )

    # Handle checkpoint resumption
    if resume_from_checkpoint:
        sft_config.resume_from_checkpoint = resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

    # Create SFTTrainer with SFTConfig (TRL 0.12+ API)
    # Note: max_seq_length, dataset_text_field, and packing are now in SFTConfig
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Save config for reproducibility
    config_save_path = Path(config.output_dir) / "training_config.yaml"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    save_config_to_yaml(config, str(config_save_path))
    logger.info(f"Saved training config to: {config_save_path}")

    # Start training
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"  Epochs: {config.num_train_epochs}")
    logger.info(f"  Effective batch size: {config.get_effective_batch_size()}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Total steps: {len(train_dataset) * config.num_train_epochs // config.get_effective_batch_size()}")
    logger.info("=" * 60)

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving checkpoint...")
        trainer.save_model(f"{config.output_dir}/interrupted-checkpoint")
        raise

    # Save final model
    logger.info("Training complete! Saving model...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    logger.info(f"Model saved to: {config.output_dir}")

    # Clean up Wandb
    if config.report_to == "wandb":
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    return trainer


def merge_and_save(
    adapter_path: str,
    output_path: str,
    base_model_name: Optional[str] = None,
):
    """
    Merge LoRA adapters with base model and save.

    WHY MERGE:
    ----------
    During training, we only save the LoRA adapter weights (small).
    For deployment, we can either:
    1. Load base model + adapters separately (more flexible)
    2. Merge adapters into base model (faster inference, simpler deployment)

    This function performs option 2, creating a standalone model
    with the fine-tuned weights baked in.

    Args:
        adapter_path: Path to the saved LoRA adapters
        output_path: Path to save the merged model
        base_model_name: Base model name (auto-detected if not provided)
    """
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading adapters from: {adapter_path}")

    # Load adapter config to get base model name
    if base_model_name is None:
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model_name = peft_config.base_model_name_or_path

    logger.info(f"Base model: {base_model_name}")

    # Load base model in full precision for merging
    logger.info("Loading base model in FP16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load adapters
    logger.info("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge adapters into base model
    logger.info("Merging adapters...")
    model = model.merge_and_unload()

    # Save merged model
    logger.info(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    logger.info("Merge complete!")


def parse_args():
    """
    Parse command-line arguments.

    This provides a comprehensive CLI for training with all configurable options.
    """
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for e-commerce LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python -m src.training.trainer --train_data ./data/train.jsonl

  # Train with custom model and hyperparameters
  python -m src.training.trainer \\
      --model_name meta-llama/Meta-Llama-3-8B-Instruct \\
      --train_data ./data/train.jsonl \\
      --eval_data ./data/eval.jsonl \\
      --output_dir ./outputs/llama3-ecommerce \\
      --lora_r 64 \\
      --learning_rate 1e-4 \\
      --epochs 3

  # Use a preset configuration
  python -m src.training.trainer --preset balanced --train_data ./data/train.jsonl

  # Resume from checkpoint
  python -m src.training.trainer \\
      --train_data ./data/train.jsonl \\
      --resume_from_checkpoint ./outputs/checkpoint-1000

  # Load config from YAML
  python -m src.training.trainer --config ./configs/training.yaml
        """,
    )

    # Configuration options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    config_group.add_argument(
        "--preset",
        type=str,
        choices=["minimal", "balanced", "high_quality"],
        help="Use a preset configuration (minimal, balanced, high_quality)",
    )

    # Model settings
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model name or path (default: Mistral-7B-Instruct-v0.3)",
    )
    model_group.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )

    # LoRA settings
    lora_group = parser.add_argument_group("LoRA Settings")
    lora_group.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="LoRA rank (default: 32). Higher = more capacity but more VRAM",
    )
    lora_group.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha/scaling factor (default: 64). Typically 2x rank",
    )
    lora_group.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)",
    )

    # Training hyperparameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2)",
    )
    train_group.add_argument(
        "--gradient_accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)",
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    train_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio (default: 0.03)",
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )

    # Data settings
    data_group = parser.add_argument_group("Data Settings")
    data_group.add_argument(
        "--train_data",
        type=str,
        required=False,
        help="Path to training data (JSONL, CSV, Parquet, or HuggingFace dataset)",
    )
    data_group.add_argument(
        "--eval_data",
        type=str,
        help="Path to evaluation data",
    )
    data_group.add_argument(
        "--task_type",
        type=str,
        default="classify",
        choices=["classify", "extract", "qa"],
        help="Task type for prompt formatting (default: classify)",
    )
    data_group.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of training samples",
    )

    # Output settings
    output_group = parser.add_argument_group("Output Settings")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/ecommerce-llm",
        help="Output directory for checkpoints and model",
    )
    output_group.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps (default: 100)",
    )
    output_group.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log metrics every N steps (default: 10)",
    )

    # Logging settings
    logging_group = parser.add_argument_group("Logging Settings")
    logging_group.add_argument(
        "--wandb_project",
        type=str,
        default="ecommerce-llm",
        help="Wandb project name",
    )
    logging_group.add_argument(
        "--wandb_run_name",
        type=str,
        help="Wandb run name",
    )
    logging_group.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard", "none"],
        help="Logging backend (default: wandb)",
    )

    # Checkpointing
    checkpoint_group = parser.add_argument_group("Checkpointing")
    checkpoint_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to checkpoint to resume from",
    )

    # Memory optimization
    memory_group = parser.add_argument_group("Memory Optimization")
    memory_group.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (uses more VRAM but faster)",
    )
    memory_group.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bfloat16 (use float16 instead)",
    )

    # Utility commands
    util_group = parser.add_argument_group("Utility Commands")
    util_group.add_argument(
        "--merge_adapters",
        type=str,
        metavar="ADAPTER_PATH",
        help="Merge LoRA adapters from this path with base model",
    )
    util_group.add_argument(
        "--merge_output",
        type=str,
        help="Output path for merged model (required with --merge_adapters)",
    )

    return parser.parse_args()


def main():
    """
    Main entry point for CLI training.

    This function:
    1. Parses command-line arguments
    2. Creates or loads configuration
    3. Executes training or utility commands
    """
    args = parse_args()

    # Handle merge command
    if args.merge_adapters:
        if not args.merge_output:
            raise ValueError("--merge_output is required when using --merge_adapters")
        merge_and_save(
            adapter_path=args.merge_adapters,
            output_path=args.merge_output,
            base_model_name=args.model_name if args.model_name != "mistralai/Mistral-7B-Instruct-v0.3" else None,
        )
        return

    # Create configuration
    if args.config:
        # Load from YAML file
        config = load_config_from_yaml(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    elif args.preset:
        # Use preset configuration
        config = get_preset_config(args.preset)
        logger.info(f"Using preset configuration: {args.preset}")
    else:
        # Build configuration from CLI arguments
        from .config import QuantizationConfig, LoRAConfig

        config = TrainingConfig(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            output_dir=args.output_dir,
            quantization=QuantizationConfig(
                bnb_4bit_compute_dtype="float16" if args.no_bf16 else "bfloat16",
            ),
            lora=LoRAConfig(
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            ),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            bf16=not args.no_bf16,
            train_data_path=args.train_data,
            eval_data_path=args.eval_data,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            report_to=args.report_to,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

    # Override config with CLI arguments if provided
    if args.train_data:
        config.train_data_path = args.train_data
    if args.eval_data:
        config.eval_data_path = args.eval_data
    if args.output_dir != "./outputs/ecommerce-llm":
        config.output_dir = args.output_dir
    if args.resume_from_checkpoint:
        config.resume_from_checkpoint = args.resume_from_checkpoint

    # Validate configuration
    if not config.train_data_path:
        raise ValueError(
            "Training data path is required. Use --train_data or set train_data_path in config."
        )

    # Print configuration summary
    logger.info("=" * 60)
    logger.info("Training Configuration Summary")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"LoRA Rank: {config.lora.lora_r}")
    logger.info(f"Epochs: {config.num_train_epochs}")
    logger.info(f"Effective Batch Size: {config.get_effective_batch_size()}")
    logger.info(f"Learning Rate: {config.learning_rate}")
    logger.info("=" * 60)

    # Run training
    trainer = train_model(
        config=config,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    logger.info("Training complete!")
    logger.info(f"Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
