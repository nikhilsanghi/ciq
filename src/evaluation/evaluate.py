"""
Evaluation script for e-commerce LLM.

Runs inference on test set and computes metrics for classification,
attribute extraction, and Q&A tasks.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from .metrics import (
    compute_all_metrics,
    compute_classification_metrics,
    compute_extraction_metrics,
    compute_qa_metrics,
    parse_json_output,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Task-specific prompt prefixes (as defined in CLAUDE.md)
TASK_PREFIXES = {
    "classification": "[CLASSIFY]",
    "extraction": "[EXTRACT]",
    "qa": "[QA]",
}


def load_model_and_tokenizer(
    model_path: str,
    device: Optional[str] = None,
    use_flash_attention: bool = True,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer for evaluation.

    Args:
        model_path: Path to model checkpoint or HuggingFace model ID
        device: Device to load model on (auto-detected if None)
        use_flash_attention: Whether to use Flash Attention 2
        load_in_4bit: Load model in 4-bit quantization
        load_in_8bit: Load model in 8-bit quantization

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    logger.info(f"Loading model from: {model_path}")

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure quantization if requested
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = device

    if use_flash_attention and device == "cuda":
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            logger.warning("Flash Attention 2 not available, using default attention")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()

    logger.info(f"Model loaded successfully. Parameters: {model.num_parameters():,}")

    return model, tokenizer


def load_test_data(
    test_data_path: str,
    task: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load test data from file.

    Supports JSON, JSONL, and CSV formats.

    Args:
        test_data_path: Path to test data file
        task: Task type for validation
        max_samples: Maximum number of samples to load

    Returns:
        List of test samples with 'input' and 'reference' keys
    """
    logger.info(f"Loading test data from: {test_data_path}")

    path = Path(test_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")

    data = []

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                data = loaded
            elif isinstance(loaded, dict) and "data" in loaded:
                data = loaded["data"]
            else:
                raise ValueError("JSON file must contain a list or have a 'data' key")
    elif path.suffix == ".csv":
        import csv

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Validate and normalize data format
    normalized_data = []
    for sample in data:
        normalized = _normalize_sample(sample, task)
        if normalized:
            normalized_data.append(normalized)

    if max_samples and len(normalized_data) > max_samples:
        normalized_data = normalized_data[:max_samples]

    logger.info(f"Loaded {len(normalized_data)} test samples")

    return normalized_data


def _normalize_sample(sample: Dict[str, Any], task: str) -> Optional[Dict[str, Any]]:
    """
    Normalize sample format to have consistent keys.

    Expected output format:
    {
        "input": str,  # Input text/prompt
        "reference": Any,  # Ground truth (str for classification/qa, dict for extraction)
        "metadata": dict  # Optional metadata
    }
    """
    # Common input key variations
    input_keys = ["input", "text", "prompt", "question", "product", "query"]
    # Common reference key variations
    ref_keys = ["reference", "label", "answer", "category", "attributes", "output", "target"]

    input_text = None
    reference = None

    for key in input_keys:
        if key in sample:
            input_text = sample[key]
            break

    for key in ref_keys:
        if key in sample:
            reference = sample[key]
            break

    if input_text is None or reference is None:
        logger.warning(f"Sample missing required fields: {sample.keys()}")
        return None

    return {
        "input": str(input_text),
        "reference": reference,
        "metadata": sample.get("metadata", {}),
    }


def run_inference_batch(
    model: Any,
    tokenizer: Any,
    test_data: List[Dict[str, Any]],
    task: str,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> List[str]:
    """
    Run batch inference on test data.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        test_data: List of test samples
        task: Task type (classification, extraction, qa)
        batch_size: Batch size for inference
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for deterministic)
        top_p: Top-p sampling parameter
        repetition_penalty: Repetition penalty

    Returns:
        List of generated predictions
    """
    logger.info(f"Running inference on {len(test_data)} samples (batch_size={batch_size})")

    predictions = []
    task_prefix = TASK_PREFIXES.get(task.lower(), "")

    # Generation config for deterministic outputs
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p if temperature > 0 else None,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Remove None values
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    # Process in batches
    for i in tqdm(range(0, len(test_data), batch_size), desc="Inference"):
        batch = test_data[i : i + batch_size]

        # Prepare prompts with task prefix
        prompts = [_format_prompt(sample["input"], task_prefix) for sample in batch]

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
            )

        # Decode predictions
        for j, output in enumerate(outputs):
            # Remove input tokens from output
            input_length = inputs.input_ids[j].shape[0]
            generated_tokens = output[input_length:]
            prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Post-process based on task
            prediction = _postprocess_prediction(prediction, task)
            predictions.append(prediction)

    logger.info(f"Generated {len(predictions)} predictions")

    return predictions


def _format_prompt(input_text: str, task_prefix: str) -> str:
    """Format input with task-specific prefix."""
    if task_prefix:
        return f"{task_prefix} {input_text}"
    return input_text


def _postprocess_prediction(prediction: str, task: str) -> str:
    """
    Post-process model prediction based on task.

    Args:
        prediction: Raw model output
        task: Task type

    Returns:
        Cleaned prediction
    """
    # Strip whitespace
    prediction = prediction.strip()

    # Remove common prefixes/suffixes
    prefixes_to_remove = ["Answer:", "Output:", "Result:", "Category:", "Response:"]
    for prefix in prefixes_to_remove:
        if prediction.lower().startswith(prefix.lower()):
            prediction = prediction[len(prefix) :].strip()

    # For extraction task, ensure valid JSON
    if task.lower() == "extraction":
        parsed = parse_json_output(prediction)
        if parsed is not None:
            # Return normalized JSON string
            return json.dumps(parsed)

    return prediction


def run_inference_vllm(
    model_path: str,
    test_data: List[Dict[str, Any]],
    task: str,
    vllm_url: str = "http://localhost:8000/v1",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> List[str]:
    """
    Run inference using vLLM server.

    Args:
        model_path: Model name for vLLM
        test_data: List of test samples
        task: Task type
        vllm_url: vLLM server URL
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        List of generated predictions
    """
    from openai import OpenAI

    client = OpenAI(base_url=vllm_url, api_key="EMPTY")

    predictions = []
    task_prefix = TASK_PREFIXES.get(task.lower(), "")

    for sample in tqdm(test_data, desc="vLLM Inference"):
        prompt = _format_prompt(sample["input"], task_prefix)

        try:
            response = client.completions.create(
                model=model_path,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            prediction = response.choices[0].text.strip()
            prediction = _postprocess_prediction(prediction, task)
            predictions.append(prediction)
        except Exception as e:
            logger.error(f"vLLM inference error: {e}")
            predictions.append("")

    return predictions


def evaluate_model(
    model_path: str,
    test_data_path: str,
    task: str,
    output_path: Optional[str] = None,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    use_vllm: bool = False,
    vllm_url: str = "http://localhost:8000/v1",
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    save_predictions: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on test data and compute metrics.

    Args:
        model_path: Path to model checkpoint or HuggingFace model ID
        test_data_path: Path to test data file
        task: Task type ("classification", "extraction", "qa")
        output_path: Optional path to save evaluation report
        batch_size: Batch size for inference
        max_samples: Maximum number of samples to evaluate
        use_vllm: Whether to use vLLM server for inference
        vllm_url: vLLM server URL
        temperature: Sampling temperature (0.0 for deterministic)
        max_new_tokens: Maximum tokens to generate
        load_in_4bit: Load model in 4-bit quantization
        load_in_8bit: Load model in 8-bit quantization
        save_predictions: Whether to save individual predictions

    Returns:
        Dictionary of evaluation metrics
    """
    # Validate task
    valid_tasks = ["classification", "extraction", "qa"]
    if task.lower() not in valid_tasks:
        raise ValueError(f"Invalid task: {task}. Must be one of: {valid_tasks}")

    task = task.lower()

    # Load test data
    test_data = load_test_data(test_data_path, task, max_samples)

    if not test_data:
        raise ValueError("No valid test samples found")

    # Run inference
    if use_vllm:
        logger.info("Using vLLM server for inference")
        predictions = run_inference_vllm(
            model_path=model_path,
            test_data=test_data,
            task=task,
            vllm_url=vllm_url,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    else:
        model, tokenizer = load_model_and_tokenizer(
            model_path,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
        predictions = run_inference_batch(
            model=model,
            tokenizer=tokenizer,
            test_data=test_data,
            task=task,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    # Extract references
    references = [sample["reference"] for sample in test_data]

    # Compute metrics
    logger.info(f"Computing {task} metrics...")
    metrics = compute_all_metrics(task, predictions, references)

    # Log metrics
    logger.info("=" * 50)
    logger.info(f"Evaluation Results ({task})")
    logger.info("=" * 50)
    for metric_name, value in metrics.items():
        if isinstance(value, dict):
            logger.info(f"{metric_name}:")
            for k, v in value.items():
                logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"{metric_name}: {value:.4f}")
    logger.info("=" * 50)

    # Generate report if output path specified
    if output_path:
        report_data = {
            "model_path": model_path,
            "test_data_path": test_data_path,
            "task": task,
            "num_samples": len(test_data),
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "config": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "batch_size": batch_size,
                "use_vllm": use_vllm,
            },
        }

        if save_predictions:
            report_data["predictions"] = [
                {
                    "input": test_data[i]["input"],
                    "prediction": predictions[i],
                    "reference": references[i],
                }
                for i in range(len(predictions))
            ]

        generate_evaluation_report(report_data, output_path)

    return metrics


def generate_evaluation_report(
    report_data: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Generate and save evaluation report.

    Args:
        report_data: Dictionary containing evaluation results
        output_path: Path to save the report
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    json_path = output_path if output_path.endswith(".json") else f"{output_path}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)
    logger.info(f"Saved JSON report to: {json_path}")

    # Generate markdown report
    md_path = json_path.replace(".json", ".md")
    md_content = _generate_markdown_report(report_data)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved Markdown report to: {md_path}")


def _generate_markdown_report(report_data: Dict[str, Any]) -> str:
    """Generate markdown-formatted evaluation report."""
    lines = [
        f"# Evaluation Report",
        f"",
        f"**Model:** {report_data['model_path']}",
        f"**Task:** {report_data['task']}",
        f"**Test Data:** {report_data['test_data_path']}",
        f"**Samples:** {report_data['num_samples']}",
        f"**Timestamp:** {report_data['timestamp']}",
        f"",
        f"## Metrics",
        f"",
    ]

    metrics = report_data.get("metrics", {})
    for metric_name, value in metrics.items():
        if isinstance(value, dict):
            lines.append(f"### {metric_name}")
            lines.append("")
            lines.append("| Level | Accuracy |")
            lines.append("|-------|----------|")
            for k, v in value.items():
                lines.append(f"| {k} | {v:.4f} |")
            lines.append("")
        else:
            lines.append(f"- **{metric_name}:** {value:.4f}")

    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    config = report_data.get("config", {})
    for key, value in config.items():
        lines.append(f"- **{key}:** {value}")

    # Add sample predictions if available
    predictions = report_data.get("predictions", [])
    if predictions:
        lines.append("")
        lines.append("## Sample Predictions")
        lines.append("")
        for i, pred in enumerate(predictions[:5]):  # Show first 5 samples
            lines.append(f"### Sample {i + 1}")
            lines.append(f"**Input:** {pred['input'][:200]}...")
            lines.append(f"**Prediction:** {pred['prediction']}")
            lines.append(f"**Reference:** {pred['reference']}")
            lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point for evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate e-commerce LLM on test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data file (JSON, JSONL, or CSV)",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "extraction", "qa"],
        help="Task type to evaluate",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation report",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for deterministic)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )

    # vLLM options
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM server for inference",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server URL",
    )

    # Quantization options
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization",
    )

    # Output options
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Do not save individual predictions in report",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run evaluation
    try:
        metrics = evaluate_model(
            model_path=args.model_path,
            test_data_path=args.test_data,
            task=args.task,
            output_path=args.output,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            use_vllm=args.use_vllm,
            vllm_url=args.vllm_url,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            save_predictions=not args.no_save_predictions,
        )

        # Print summary
        print("\nEvaluation Complete!")
        print(f"Task: {args.task}")
        print(f"Samples: {args.max_samples or 'all'}")
        print("\nKey Metrics:")
        for key, value in metrics.items():
            if not isinstance(value, dict):
                print(f"  {key}: {value:.4f}")

        if args.output:
            print(f"\nReport saved to: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
