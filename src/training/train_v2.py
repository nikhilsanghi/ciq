"""
Training Script v2 - With Output Validation During Training.

Key improvements from v1 (failed ECInstruct attempt):
1. Validates actual model outputs during training, not just loss metrics
2. Logs sample generations every N steps
3. Tests on exact inference prompts
4. Includes validation set monitoring

LESSON LEARNED: "92% accuracy means nothing if you trained on the wrong task."

Usage:
    python -m src.training.train_v2 \
        --train_data data/processed/train.jsonl \
        --eval_data data/processed/val.jsonl \
        --output_dir outputs/ciq-model-v2 \
        --epochs 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Sample prompts for validation - MUST match inference format from src/api/main.py EXACTLY!
# Note: Classification removed since we don't have taxonomy-labeled training data
VALIDATION_PROMPTS = [
    {
        "task": "extract",
        "prompt": """[EXTRACT] Extract all product attributes from the following text as JSON key-value pairs.

Product: Apple iPhone 15 Pro 256GB Natural Titanium

Respond with a valid JSON object containing attribute names as keys and their values.
Example: {"brand": "Nike", "size": "Large", "color": "Blue"}

Attributes:""",
        "expected_contains": ["Apple", "iPhone", "256", "{"],
    },
    {
        "task": "qa",
        "prompt": """[QA] Answer the question about the product based on the provided information.

Product Information:
MacBook Air M3 15-inch with 18-hour battery life, 8GB RAM, 256GB SSD

Question: How long does the battery last?

Provide a concise and accurate answer based only on the available information.

Answer:""",
        "expected_contains": ["18", "hour"],
    },
]


class OutputValidationCallback(TrainerCallback):
    """
    Callback to validate model outputs during training.

    This is CRITICAL - we must test actual outputs, not just loss metrics!
    """

    def __init__(
        self,
        tokenizer,
        validation_prompts: List[Dict[str, Any]],
        eval_every_n_steps: int = 500,
        output_log_path: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.validation_prompts = validation_prompts
        self.eval_every_n_steps = eval_every_n_steps
        self.output_log_path = output_log_path
        self.generation_logs = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Check outputs every N steps."""
        if state.global_step % self.eval_every_n_steps == 0 and state.global_step > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"OUTPUT VALIDATION at step {state.global_step}")
            logger.info(f"{'='*60}")

            self._validate_outputs(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Final validation at end of training."""
        logger.info(f"\n{'='*60}")
        logger.info("FINAL OUTPUT VALIDATION")
        logger.info(f"{'='*60}")

        self._validate_outputs(model, state.global_step)

        # Save generation log
        if self.output_log_path:
            with open(self.output_log_path, 'w') as f:
                json.dump(self.generation_logs, f, indent=2)
            logger.info(f"Saved generation log to: {self.output_log_path}")

    def _validate_outputs(self, model, step: int):
        """Generate and validate outputs for test prompts."""
        model.eval()

        step_results = {"step": step, "generations": []}

        for prompt_info in self.validation_prompts:
            task = prompt_info["task"]
            prompt = prompt_info["prompt"]
            expected = prompt_info["expected_contains"]

            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Get only the new tokens (response)
            response = generated[len(prompt):].strip()

            # Check if response contains expected content
            matches = [exp for exp in expected if exp.lower() in response.lower()]
            is_valid = len(matches) > 0

            # Check for degenerate output (repetitive nonsense)
            is_degenerate = self._check_degenerate(response)

            # Log result
            status = "DEGENERATE" if is_degenerate else ("PASS" if is_valid else "CHECK")
            logger.info(f"\n[{task.upper()}] {status}")
            logger.info(f"Prompt: {prompt[:80]}...")
            logger.info(f"Response: {response[:200]}")
            if is_valid:
                logger.info(f"Matched: {matches}")

            step_results["generations"].append({
                "task": task,
                "prompt": prompt,
                "response": response,
                "expected": expected,
                "matches": matches,
                "is_valid": is_valid,
                "is_degenerate": is_degenerate,
            })

        self.generation_logs.append(step_results)
        model.train()

    def _check_degenerate(self, text: str) -> bool:
        """
        Check if output is degenerate (repetitive nonsense).

        This catches issues like: "onomyonomy [CLASSIFY] Classify theify..."
        """
        if not text:
            return True

        # Check for excessive repetition
        words = text.split()
        if len(words) < 3:
            return False

        # Count repeated sequences
        for n in [2, 3, 4]:  # Check 2-grams, 3-grams, 4-grams
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            if ngrams:
                from collections import Counter
                counts = Counter(ngrams)
                most_common_count = counts.most_common(1)[0][1]
                if most_common_count > 3:  # Same n-gram appears more than 3 times
                    return True

        # Check for character-level repetition
        if len(text) > 20:
            char_repeat = max(text.count(char * 5) for char in set(text) if char.isalpha())
            if char_repeat > 2:
                return True

        return False


def load_quantized_model(model_name: str, use_flash_attention: bool = True):
    """Load model with 4-bit quantization and Flash Attention 2."""
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Flash Attention 2: {'ENABLED' if use_flash_attention else 'DISABLED'}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Flash Attention 2 provides 2-3x speedup and reduced memory usage
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    return model


def setup_tokenizer(model_name: str, max_length: int = 2048):
    """Setup tokenizer with proper padding."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        model_max_length=max_length,
    )

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def add_lora_adapters(model, lora_r: int = 32, lora_alpha: int = 64):
    """Add LoRA adapters to model."""
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def format_example(example: Dict[str, Any]) -> str:
    """
    Format training example into prompt format.

    Format must EXACTLY match inference format!
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    # Build prompt
    if input_text:
        prompt = f"{instruction}\n\n{input_text}"
    else:
        prompt = instruction

    # Full training text includes response
    return f"{prompt}\n\n{output}"


def prepare_dataset(data_path: str, tokenizer) -> Dataset:
    """Load and prepare dataset."""
    logger.info(f"Loading dataset from: {data_path}")

    dataset = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"Loaded {len(dataset)} examples")

    # Format examples
    def format_fn(example):
        return {"text": format_example(example)}

    dataset = dataset.map(format_fn, remove_columns=dataset.column_names)

    return dataset


def train(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    train_data: str = "data/processed/train.jsonl",
    eval_data: Optional[str] = "data/processed/val.jsonl",
    output_dir: str = "outputs/ciq-model-v2",
    epochs: int = 3,
    batch_size: int = 4,  # Increased from 2 (flash attention allows larger batches)
    gradient_accumulation: int = 4,  # Reduced from 8 (same effective batch size, faster)
    learning_rate: float = 2e-4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    max_seq_length: int = 2048,
    eval_steps: int = 500,
    save_steps: int = 500,
    logging_steps: int = 50,
    output_validation_steps: int = 500,
    use_flash_attention: bool = True,  # Enable Flash Attention 2 for 2-3x speedup
    packing: bool = True,  # Pack multiple sequences for efficiency
    use_torch_compile: bool = False,  # torch.compile for extra 10-20% speedup (experimental)
):
    """
    Main training function with output validation.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Log training config
    config = {
        "model_name": model_name,
        "train_data": train_data,
        "eval_data": eval_data,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": batch_size * gradient_accumulation,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "max_seq_length": max_seq_length,
        "use_flash_attention": use_flash_attention,
        "packing": packing,
        "use_torch_compile": use_torch_compile,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("="*60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*60)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*60)

    # Load model and tokenizer
    model = load_quantized_model(model_name, use_flash_attention=use_flash_attention)
    tokenizer = setup_tokenizer(model_name, max_seq_length)

    # Add LoRA adapters
    model = add_lora_adapters(model, lora_r, lora_alpha)

    # Optional: torch.compile for extra speedup (experimental with quantized models)
    if use_torch_compile:
        logger.info("Applying torch.compile (this may take a few minutes on first run)...")
        model = torch.compile(model)

    # Prepare datasets
    train_dataset = prepare_dataset(train_data, tokenizer)
    eval_dataset = None
    if eval_data and Path(eval_data).exists():
        eval_dataset = prepare_dataset(eval_data, tokenizer)

    # Output validation callback
    output_callback = OutputValidationCallback(
        tokenizer=tokenizer,
        validation_prompts=VALIDATION_PROMPTS,
        eval_every_n_steps=output_validation_steps,
        output_log_path=str(output_path / "generation_log.json"),
    )

    # Training configuration
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=packing,  # Pack multiple short sequences into one for efficiency
        report_to="tensorboard",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[output_callback],
    )

    # Test outputs BEFORE training (baseline)
    logger.info("\n" + "="*60)
    logger.info("PRE-TRAINING BASELINE OUTPUTS")
    logger.info("="*60)
    output_callback._validate_outputs(model, step=0)

    # Train
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training interrupted! Saving checkpoint...")
        trainer.save_model(f"{output_dir}/interrupted-checkpoint")
        raise

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"\nTraining complete! Model saved to: {output_dir}")
    logger.info(f"Generation log saved to: {output_path / 'generation_log.json'}")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train e-commerce LLM with output validation")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/ciq-model-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Per-device batch size (default: 4 with flash attention)")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                       help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--output_validation_steps", type=int, default=500,
                       help="Validate actual model outputs every N steps")
    parser.add_argument("--no_flash_attention", action="store_true",
                       help="Disable Flash Attention 2 (not recommended)")
    parser.add_argument("--no_packing", action="store_true",
                       help="Disable sequence packing")
    parser.add_argument("--torch_compile", action="store_true",
                       help="Enable torch.compile for extra 10-20%% speedup (experimental)")

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        train_data=args.train_data,
        eval_data=args.eval_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_validation_steps=args.output_validation_steps,
        use_flash_attention=not args.no_flash_attention,
        packing=not args.no_packing,
        use_torch_compile=args.torch_compile,
    )


if __name__ == "__main__":
    main()
