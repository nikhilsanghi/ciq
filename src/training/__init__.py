"""
Training utilities for QLoRA fine-tuning.

This module provides a complete training pipeline for fine-tuning
large language models on e-commerce tasks using QLoRA.

Quick Start:
------------
>>> from src.training import TrainingConfig, train_model
>>> config = TrainingConfig(
...     model_name="mistralai/Mistral-7B-Instruct-v0.3",
...     train_data_path="./data/train.jsonl",
... )
>>> trainer = train_model(config)

CLI Usage:
----------
$ python -m src.training.trainer --train_data ./data/train.jsonl --output_dir ./outputs/my-model
"""

from .config import (
    TrainingConfig,
    QuantizationConfig,
    LoRAConfig,
    get_qlora_config,
    get_lora_config,
    load_config_from_yaml,
    save_config_to_yaml,
    get_preset_config,
    PRESETS,
)
from .trainer import (
    train_model,
    load_base_model,
    setup_tokenizer,
    prepare_model_for_training,
    prepare_dataset,
    format_prompt,
    merge_and_save,
    check_gpu_availability,
)

__all__ = [
    # Configuration classes
    "TrainingConfig",
    "QuantizationConfig",
    "LoRAConfig",
    # Configuration helpers
    "get_qlora_config",
    "get_lora_config",
    "load_config_from_yaml",
    "save_config_to_yaml",
    "get_preset_config",
    "PRESETS",
    # Training functions
    "train_model",
    "load_base_model",
    "setup_tokenizer",
    "prepare_model_for_training",
    "prepare_dataset",
    "format_prompt",
    "merge_and_save",
    "check_gpu_availability",
]
