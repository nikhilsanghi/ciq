"""
Training configuration for QLoRA fine-tuning.

This module provides comprehensive configuration management for fine-tuning
large language models using QLoRA (Quantized Low-Rank Adaptation). Each parameter
is documented with detailed explanations to aid understanding of the training process.

Key Concepts:
-------------
1. QLoRA: Combines 4-bit quantization with LoRA adapters to enable fine-tuning
   of 7B+ parameter models on consumer GPUs (8-12GB VRAM).

2. NF4 Quantization: Normal Float 4-bit format that better preserves the
   distribution of neural network weights compared to standard INT4.

3. LoRA: Low-Rank Adaptation freezes the base model and adds small trainable
   matrices (adapters) that learn task-specific adjustments.

Memory Calculation Example (Mistral-7B):
----------------------------------------
- Full precision (FP32): 7B * 4 bytes = 28GB
- Half precision (FP16): 7B * 2 bytes = 14GB
- 4-bit quantized: 7B * 0.5 bytes = 3.5GB
- With LoRA adapters (r=32): ~4.5GB total
- With gradients and optimizer states: ~8-10GB total
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import torch

# Lazy imports to avoid loading heavy libraries at module import time
# These are imported when needed in the functions below


@dataclass
class QuantizationConfig:
    """
    Configuration for 4-bit quantization using bitsandbytes.

    WHY QUANTIZATION?
    -----------------
    Quantization reduces the precision of model weights from 32/16 bits to 4 bits,
    dramatically reducing memory usage while maintaining most of the model's capability.
    This enables fine-tuning of 7B models on GPUs with only 8-12GB VRAM.

    Attributes:
        load_in_4bit: Enable 4-bit quantization. Reduces 7B model from ~14GB (FP16)
                      to ~4GB. Essential for consumer GPU fine-tuning.

        bnb_4bit_quant_type: Quantization format. Options:
            - "nf4": Normal Float 4 - Optimal for normally distributed weights
                     (which neural networks typically have). Better quality than INT4.
            - "fp4": Float Point 4 - Alternative format, generally worse than NF4.

        bnb_4bit_compute_dtype: Precision for computations during forward/backward pass.
            - "bfloat16": Best for Ampere+ GPUs (A10G, A100, RTX 3090+).
                          Has same range as FP32 but less precision.
            - "float16": For older GPUs (V100, RTX 2080). Better precision but
                         limited range can cause overflow.

        bnb_4bit_use_double_quant: Quantize the quantization constants themselves.
            Saves additional ~0.4 bits per parameter (~300MB for 7B model).
            Minimal quality impact, recommended to always enable.
    """
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    def to_bits_and_bytes_config(self):
        """
        Convert to HuggingFace BitsAndBytesConfig.

        Returns:
            BitsAndBytesConfig: Configuration object for transformers library.
        """
        from transformers import BitsAndBytesConfig

        # Map string dtype to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=dtype_map.get(self.bnb_4bit_compute_dtype, torch.bfloat16),
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )


@dataclass
class LoRAConfig:
    """
    Configuration for Low-Rank Adaptation (LoRA).

    WHY LoRA?
    ---------
    Instead of fine-tuning all 7B parameters (which requires storing gradients
    and optimizer states for each), LoRA:
    1. Freezes the base model weights
    2. Adds small trainable matrices (adapters) to specific layers
    3. Only trains these adapters (~1-5% of total parameters)

    This reduces memory usage from ~50GB to ~8GB for a 7B model.

    The Math:
    ---------
    For a weight matrix W (d x k), LoRA adds: W' = W + BA
    where B is (d x r) and A is (r x k), with r << min(d, k)

    Example: For Mistral-7B attention layers (4096 x 4096):
    - Original: 4096 * 4096 = 16.7M parameters
    - LoRA r=32: (4096 * 32) + (32 * 4096) = 262K parameters (1.5% of original)

    Attributes:
        lora_r: Rank of the low-rank matrices. Higher rank = more capacity but:
            - More VRAM usage
            - More trainable parameters
            - Risk of overfitting on small datasets
            Typical values: 8 (minimal), 16 (balanced), 32-64 (high capacity)

        lora_alpha: Scaling factor for LoRA updates. The effective learning rate
            for LoRA is scaled by (alpha/r). Common practice: alpha = 2 * r
            Higher alpha = stronger LoRA contribution to final output.

        lora_target_modules: Which layers to add LoRA adapters to.
            For attention: ["q_proj", "k_proj", "v_proj", "o_proj"]
            For FFN: ["gate_proj", "up_proj", "down_proj"]
            More modules = more parameters but better fine-tuning.

        lora_dropout: Dropout probability for LoRA layers. Helps prevent
            overfitting, especially important for small datasets.
            Typical: 0.05-0.1 for large datasets, 0.1-0.2 for small datasets.

        bias: How to handle bias terms. Options:
            - "none": Don't train any biases (recommended for QLoRA)
            - "all": Train all biases
            - "lora_only": Only train biases in LoRA layers

        task_type: The task type for PEFT. "CAUSAL_LM" for text generation.
    """
    lora_r: int = 32
    lora_alpha: int = 64
    lora_target_modules: List[str] = field(default_factory=lambda: [
        # Attention layers - where the model learns what to focus on
        "q_proj",  # Query projection - what am I looking for?
        "k_proj",  # Key projection - what information is available?
        "v_proj",  # Value projection - what is the actual content?
        "o_proj",  # Output projection - combine attention results
        # Feed-forward layers - where the model processes information
        "gate_proj",  # Gating mechanism in SwiGLU activation
        "up_proj",    # Upward projection (expands dimension)
        "down_proj",  # Downward projection (compresses back)
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_peft_config(self):
        """
        Convert to PEFT LoraConfig.

        Returns:
            LoraConfig: Configuration object for PEFT library.
        """
        from peft import LoraConfig, TaskType

        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "SEQ_CLS": TaskType.SEQ_CLS,
        }

        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=task_type_map.get(self.task_type, TaskType.CAUSAL_LM),
        )


@dataclass
class TrainingConfig:
    """
    Complete training configuration combining model, quantization, LoRA, and training settings.

    This configuration is designed for:
    - Fine-tuning 7B-8B parameter models (Mistral-7B, LLaMA-3-8B)
    - Running on single GPU with 8-24GB VRAM
    - E-commerce tasks: classification, extraction, Q&A

    Attributes:
        model_name: HuggingFace model identifier or local path.
            Recommended: "mistralai/Mistral-7B-Instruct-v0.3" or "meta-llama/Meta-Llama-3-8B-Instruct"

        max_seq_length: Maximum sequence length for training.
            - Longer = more context but quadratically more memory for attention
            - 2048: Good for most e-commerce tasks
            - 4096: For complex Q&A with long product descriptions

        output_dir: Directory to save checkpoints and final model.

        num_train_epochs: Number of passes through the training data.
            - 1-2: For very large datasets (>100K examples)
            - 3: Sweet spot for medium datasets (10K-100K)
            - 5+: Only for small datasets, watch for overfitting

        per_device_train_batch_size: Samples per GPU per step.
            Limited by VRAM. Start with 1-2 for 7B models on 12GB GPU.

        gradient_accumulation_steps: Accumulate gradients over N steps.
            Effective batch size = per_device_batch * accumulation * num_gpus
            Higher = more stable training but slower iteration.

        learning_rate: Step size for optimization.
            - QLoRA typically uses higher LR than full fine-tuning
            - 1e-4 to 2e-4 is typical range
            - Higher for larger datasets, lower for small datasets

        warmup_ratio: Fraction of training for learning rate warmup.
            Gradually increases LR from 0 to target. Prevents early instability.
            0.03-0.1 is typical.

        weight_decay: L2 regularization coefficient.
            Helps prevent overfitting. 0.01-0.1 is typical.

        logging_steps: Log metrics every N steps.

        save_steps: Save checkpoint every N steps.
            Important for spot instance recovery!

        save_total_limit: Maximum checkpoints to keep.
            Prevents disk space exhaustion.

        gradient_checkpointing: Trade compute for memory.
            Recomputes activations during backward pass instead of storing.
            ~30% slower but ~50% less VRAM. Essential for 7B models.

        bf16: Use bfloat16 mixed precision training.
            Faster and more stable than fp16 on Ampere+ GPUs.

        tf32: Use TensorFloat-32 for matmul on Ampere+ GPUs.
            ~3x faster with minimal precision loss.
    """
    # Model settings
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_seq_length: int = 2048
    output_dir: str = "./outputs/ecommerce-llm"

    # Quantization settings (nested config)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # LoRA settings (nested config)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Optimizer settings
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Gradient clipping for stability

    # Scheduler settings
    lr_scheduler_type: str = "cosine"  # Cosine annealing - smooth LR decay

    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    eval_steps: int = 100
    eval_strategy: str = "steps"

    # Memory optimization
    gradient_checkpointing: bool = True
    bf16: bool = True
    tf32: bool = True

    # Reproducibility
    seed: int = 42

    # Data settings
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None

    # Wandb settings
    wandb_project: Optional[str] = "ecommerce-llm"
    wandb_run_name: Optional[str] = None
    report_to: str = "wandb"  # Can be "wandb", "tensorboard", or "none"

    # Additional settings
    resume_from_checkpoint: Optional[str] = None
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size including gradient accumulation."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def estimate_vram_usage(self) -> Dict[str, float]:
        """
        Estimate VRAM usage for the current configuration.

        Returns rough estimates in GB. Actual usage may vary based on
        sequence lengths, batch composition, and framework overhead.
        """
        # Rough estimates based on typical 7B model
        base_model_gb = 3.5 if self.quantization.load_in_4bit else 14.0

        # LoRA parameters (rough estimate)
        lora_params = len(self.lora.lora_target_modules) * 2 * 4096 * self.lora.lora_r
        lora_gb = (lora_params * 4) / (1024**3)  # FP32 for training

        # Optimizer states (AdamW has 2 states per parameter)
        optimizer_gb = lora_gb * 2

        # Gradients
        gradients_gb = lora_gb

        # Activations (rough estimate, highly variable)
        seq_factor = self.max_seq_length / 2048
        batch_factor = self.per_device_train_batch_size
        activations_gb = 2.0 * seq_factor * batch_factor
        if self.gradient_checkpointing:
            activations_gb *= 0.5  # Gradient checkpointing saves ~50%

        return {
            "base_model": base_model_gb,
            "lora_adapters": lora_gb,
            "optimizer_states": optimizer_gb,
            "gradients": gradients_gb,
            "activations": activations_gb,
            "total_estimated": base_model_gb + lora_gb + optimizer_gb + gradients_gb + activations_gb,
        }

    def to_training_arguments(self):
        """
        Convert to HuggingFace TrainingArguments.

        Returns:
            TrainingArguments: Configuration for HuggingFace Trainer.
        """
        from transformers import TrainingArguments

        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            lr_scheduler_type=self.lr_scheduler_type,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            eval_steps=self.eval_steps,
            eval_strategy=self.eval_strategy,
            gradient_checkpointing=self.gradient_checkpointing,
            bf16=self.bf16,
            tf32=self.tf32,
            seed=self.seed,
            report_to=self.report_to,
            run_name=self.wandb_run_name,
            resume_from_checkpoint=self.resume_from_checkpoint,
            push_to_hub=self.push_to_hub,
            hub_model_id=self.hub_model_id,
            # Additional recommended settings
            optim="paged_adamw_8bit",  # Memory-efficient optimizer
            group_by_length=True,  # Batch similar-length sequences together
            dataloader_pin_memory=True,  # Faster data transfer to GPU
            remove_unused_columns=False,  # Keep all columns for custom formatting
        )


def get_qlora_config(
    load_in_4bit: bool = True,
    quant_type: str = "nf4",
    compute_dtype: str = "bfloat16",
    use_double_quant: bool = True,
):
    """
    Create a BitsAndBytesConfig for QLoRA quantization.

    This is a convenience function for quickly creating quantization configs
    without needing to understand the full TrainingConfig structure.

    Args:
        load_in_4bit: Enable 4-bit quantization (recommended: True)
        quant_type: Quantization type - "nf4" or "fp4" (recommended: "nf4")
        compute_dtype: Compute precision - "bfloat16", "float16" (recommended: "bfloat16")
        use_double_quant: Enable double quantization (recommended: True)

    Returns:
        BitsAndBytesConfig: Ready-to-use quantization config.

    Example:
        >>> quant_config = get_qlora_config()
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "mistralai/Mistral-7B-Instruct-v0.3",
        ...     quantization_config=quant_config,
        ... )
    """
    config = QuantizationConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
    )
    return config.to_bits_and_bytes_config()


def get_lora_config(
    r: int = 32,
    alpha: int = 64,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.05,
):
    """
    Create a PEFT LoraConfig for adapter training.

    This is a convenience function for quickly creating LoRA configs.

    Args:
        r: LoRA rank. Higher = more capacity but more VRAM.
           Typical values: 8 (minimal), 16 (balanced), 32-64 (high capacity)
        alpha: Scaling factor. Typically 2x the rank.
        target_modules: Which modules to apply LoRA to. Default includes all
                       attention and feed-forward projections.
        dropout: Dropout for regularization. Higher for smaller datasets.

    Returns:
        LoraConfig: Ready-to-use LoRA configuration.

    Example:
        >>> lora_config = get_lora_config(r=16, alpha=32)
        >>> model = get_peft_model(base_model, lora_config)
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    config = LoRAConfig(
        lora_r=r,
        lora_alpha=alpha,
        lora_target_modules=target_modules,
        lora_dropout=dropout,
    )
    return config.to_peft_config()


def load_config_from_yaml(path: str) -> TrainingConfig:
    """
    Load training configuration from a YAML file.

    The YAML file should have a structure matching the TrainingConfig dataclass.
    Nested configurations (quantization, lora) should be nested in the YAML.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        TrainingConfig: Loaded and validated configuration.

    Example YAML:
        ```yaml
        model_name: mistralai/Mistral-7B-Instruct-v0.3
        max_seq_length: 2048
        output_dir: ./outputs/my-model

        quantization:
          load_in_4bit: true
          bnb_4bit_quant_type: nf4

        lora:
          lora_r: 32
          lora_alpha: 64

        num_train_epochs: 3
        learning_rate: 0.0002
        ```

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
        ValueError: If required fields are missing or invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        yaml_config = yaml.safe_load(f)

    if yaml_config is None:
        raise ValueError(f"Empty configuration file: {path}")

    # Parse nested configurations
    quant_dict = yaml_config.pop("quantization", {})
    lora_dict = yaml_config.pop("lora", {})

    # Create nested config objects
    quantization = QuantizationConfig(**quant_dict) if quant_dict else QuantizationConfig()
    lora = LoRAConfig(**lora_dict) if lora_dict else LoRAConfig()

    # Create main config
    config = TrainingConfig(
        quantization=quantization,
        lora=lora,
        **yaml_config,
    )

    return config


def save_config_to_yaml(config: TrainingConfig, path: str) -> None:
    """
    Save training configuration to a YAML file.

    Useful for reproducibility - save the exact config used for each training run.

    Args:
        config: The training configuration to save.
        path: Path where the YAML file will be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dicts
    config_dict = {
        "model_name": config.model_name,
        "max_seq_length": config.max_seq_length,
        "output_dir": config.output_dir,
        "quantization": {
            "load_in_4bit": config.quantization.load_in_4bit,
            "bnb_4bit_quant_type": config.quantization.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": config.quantization.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": config.quantization.bnb_4bit_use_double_quant,
        },
        "lora": {
            "lora_r": config.lora.lora_r,
            "lora_alpha": config.lora.lora_alpha,
            "lora_target_modules": config.lora.lora_target_modules,
            "lora_dropout": config.lora.lora_dropout,
            "bias": config.lora.bias,
            "task_type": config.lora.task_type,
        },
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "lr_scheduler_type": config.lr_scheduler_type,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "eval_steps": config.eval_steps,
        "evaluation_strategy": config.evaluation_strategy,
        "gradient_checkpointing": config.gradient_checkpointing,
        "bf16": config.bf16,
        "tf32": config.tf32,
        "seed": config.seed,
        "train_data_path": config.train_data_path,
        "eval_data_path": config.eval_data_path,
        "wandb_project": config.wandb_project,
        "wandb_run_name": config.wandb_run_name,
        "report_to": config.report_to,
    }

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


# Preset configurations for common use cases
PRESETS = {
    "minimal": TrainingConfig(
        # Minimal config for testing on limited hardware (8GB VRAM)
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_seq_length=1024,
        lora=LoRAConfig(lora_r=8, lora_alpha=16),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
    ),
    "balanced": TrainingConfig(
        # Balanced config for 12-16GB VRAM
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_seq_length=2048,
        lora=LoRAConfig(lora_r=32, lora_alpha=64),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
    ),
    "high_quality": TrainingConfig(
        # High quality config for 24GB+ VRAM (A10G, A100)
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_seq_length=4096,
        lora=LoRAConfig(lora_r=64, lora_alpha=128),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
    ),
}


def get_preset_config(preset_name: str) -> TrainingConfig:
    """
    Get a preset configuration by name.

    Available presets:
    - "minimal": For testing on 8GB GPUs
    - "balanced": For 12-16GB GPUs (recommended for most users)
    - "high_quality": For 24GB+ GPUs

    Args:
        preset_name: Name of the preset.

    Returns:
        TrainingConfig: A copy of the preset configuration.

    Raises:
        ValueError: If preset_name is not recognized.
    """
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")

    # Return a copy to avoid modifying the original preset
    import copy
    return copy.deepcopy(PRESETS[preset_name])
