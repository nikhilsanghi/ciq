"""
Format data for instruction fine-tuning.

Converts raw data to instruction/input/output format with task prefixes.
Supports Mistral and LLaMA prompt templates for consistent training.
"""

import json
import logging
from typing import Optional, Dict, Any, List, Callable, Union

from datasets import Dataset

logger = logging.getLogger(__name__)

# Task prefixes for multi-task learning
TASK_PREFIXES: Dict[str, str] = {
    "classification": "[CLASSIFY]",
    "extraction": "[EXTRACT]",
    "qa": "[QA]",
}

# Prompt templates for different model architectures
PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "mistral": {
        "template": "<s>[INST] {instruction}\n\n{input} [/INST] {output}</s>",
        "template_no_input": "<s>[INST] {instruction} [/INST] {output}</s>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    },
    "llama": {
        "template": "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction}\n\n{input} [/INST] {output}</s>",
        "template_no_input": "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST] {output}</s>",
        "system": "You are a helpful e-commerce AI assistant.",
        "bos_token": "<s>",
        "eos_token": "</s>",
    },
    "chatml": {
        "template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}\n\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>",
        "template_no_input": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>",
        "system": "You are a helpful e-commerce AI assistant.",
        "bos_token": "",
        "eos_token": "<|im_end|>",
    },
}

# Default system prompts for each task
TASK_SYSTEM_PROMPTS: Dict[str, str] = {
    "classification": "You are an expert e-commerce product classifier. Classify products into the correct category from the Google Product Taxonomy.",
    "extraction": "You are an expert at extracting product attributes. Extract attribute-value pairs from product descriptions as valid JSON.",
    "qa": "You are a helpful e-commerce assistant. Answer questions about products accurately and concisely based on the provided information.",
}


def format_for_training(
    dataset: Dataset,
    task_type: str,
    model_type: str = "mistral",
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    add_task_prefix: bool = True,
    custom_formatter: Optional[Callable[[Dict[str, Any]], Dict[str, str]]] = None,
) -> Dataset:
    """
    Main formatting function for instruction fine-tuning.

    Converts dataset to instruction/input/output format with task prefixes
    and applies the appropriate prompt template for the target model.

    Args:
        dataset: Input dataset to format.
        task_type: Type of task - 'classification', 'extraction', or 'qa'.
        model_type: Target model type - 'mistral', 'llama', or 'chatml'.
        instruction_field: Name of the instruction field in source data.
        input_field: Name of the input field in source data.
        output_field: Name of the output field in source data.
        add_task_prefix: Whether to add task prefix (e.g., [CLASSIFY]).
        custom_formatter: Optional custom formatting function.

    Returns:
        Formatted dataset with 'text' field containing the full prompt.

    Raises:
        ValueError: If task_type or model_type is invalid.

    Example:
        >>> formatted = format_for_training(dataset, task_type="classification")
        >>> print(formatted[0]["text"])
    """
    # Validate inputs
    if task_type not in TASK_PREFIXES:
        raise ValueError(
            f"Invalid task_type: {task_type}. "
            f"Valid options: {list(TASK_PREFIXES.keys())}"
        )

    if model_type not in PROMPT_TEMPLATES:
        raise ValueError(
            f"Invalid model_type: {model_type}. "
            f"Valid options: {list(PROMPT_TEMPLATES.keys())}"
        )

    logger.info(f"Formatting {len(dataset):,} examples for {task_type} task ({model_type} format)")

    task_prefix = TASK_PREFIXES[task_type] if add_task_prefix else ""
    template_config = PROMPT_TEMPLATES[model_type]

    def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
        # Use custom formatter if provided
        if custom_formatter:
            formatted = custom_formatter(example)
            instruction = formatted.get("instruction", "")
            input_text = formatted.get("input", "")
            output_text = formatted.get("output", "")
        else:
            instruction = example.get(instruction_field, "")
            input_text = example.get(input_field, "")
            output_text = example.get(output_field, "")

        # Add task prefix to instruction
        if task_prefix and not instruction.startswith(task_prefix):
            instruction = f"{task_prefix} {instruction}"

        # Create prompt from template
        text = create_prompt_template(
            instruction=instruction,
            input_text=input_text,
            output_text=output_text,
            model_type=model_type,
            system_prompt=TASK_SYSTEM_PROMPTS.get(task_type),
        )

        return {
            "text": text,
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "task_type": task_type,
        }

    formatted_dataset = dataset.map(
        format_example,
        desc=f"Formatting for {task_type}",
        remove_columns=dataset.column_names,
    )

    logger.info(f"Formatted {len(formatted_dataset):,} examples")
    return formatted_dataset


def format_classification_example(
    product_title: str,
    product_description: Optional[str] = None,
    category: str = "",
    taxonomy_level: str = "full",
) -> Dict[str, str]:
    """
    Format a product classification example with [CLASSIFY] prefix.

    Args:
        product_title: Product title/name.
        product_description: Optional product description.
        category: Target category from Google Product Taxonomy.
        taxonomy_level: Level of taxonomy - 'full', 'top', or 'leaf'.

    Returns:
        Dictionary with instruction, input, and output fields.

    Example:
        >>> example = format_classification_example(
        ...     product_title="Apple iPhone 15 Pro 256GB",
        ...     category="Electronics > Communications > Telephony > Mobile Phones"
        ... )
    """
    instruction = f"{TASK_PREFIXES['classification']} Classify the following product into the appropriate Google Product Taxonomy category."

    if product_description:
        input_text = f"Title: {product_title}\nDescription: {product_description}"
    else:
        input_text = f"Title: {product_title}"

    return {
        "instruction": instruction,
        "input": input_text,
        "output": category,
    }


def format_extraction_example(
    product_title: str,
    product_description: str,
    attributes: Dict[str, Any],
    attribute_schema: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Format a product attribute extraction example with [EXTRACT] prefix.

    The output is formatted as valid JSON for structured extraction.

    Args:
        product_title: Product title/name.
        product_description: Product description text.
        attributes: Dictionary of attribute-value pairs to extract.
        attribute_schema: Optional list of expected attributes.

    Returns:
        Dictionary with instruction, input, and output fields.

    Example:
        >>> example = format_extraction_example(
        ...     product_title="Nike Air Max 90",
        ...     product_description="Classic sneakers in white leather, size 10.",
        ...     attributes={"brand": "Nike", "color": "white", "size": "10"}
        ... )
    """
    if attribute_schema:
        schema_hint = f"Extract the following attributes: {', '.join(attribute_schema)}"
        instruction = f"{TASK_PREFIXES['extraction']} {schema_hint}"
    else:
        instruction = f"{TASK_PREFIXES['extraction']} Extract product attributes as JSON."

    input_text = f"Title: {product_title}\nDescription: {product_description}"

    # Format output as valid JSON
    try:
        output_text = json.dumps(attributes, ensure_ascii=False, indent=None)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize attributes to JSON: {e}")
        output_text = str(attributes)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def format_qa_example(
    question: str,
    answer: str,
    product_context: Optional[str] = None,
    product_title: Optional[str] = None,
) -> Dict[str, str]:
    """
    Format a product Q&A example with [QA] prefix.

    Args:
        question: User question about the product.
        answer: Answer to the question.
        product_context: Optional product description/context for RAG.
        product_title: Optional product title.

    Returns:
        Dictionary with instruction, input, and output fields.

    Example:
        >>> example = format_qa_example(
        ...     question="Is this laptop compatible with Windows 11?",
        ...     answer="Yes, this laptop comes pre-installed with Windows 11 Pro.",
        ...     product_title="Dell XPS 15"
        ... )
    """
    instruction = f"{TASK_PREFIXES['qa']} Answer the following question about the product."

    # Build input with available context
    input_parts = []
    if product_title:
        input_parts.append(f"Product: {product_title}")
    if product_context:
        input_parts.append(f"Context: {product_context}")
    input_parts.append(f"Question: {question}")

    input_text = "\n".join(input_parts)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": answer,
    }


def create_prompt_template(
    instruction: str,
    input_text: str = "",
    output_text: str = "",
    model_type: str = "mistral",
    system_prompt: Optional[str] = None,
    include_output: bool = True,
) -> str:
    """
    Create a formatted prompt using model-specific templates.

    Supports Mistral, LLaMA, and ChatML formats for instruction tuning.

    Args:
        instruction: The instruction/task description.
        input_text: Optional input context.
        output_text: Expected output (included during training).
        model_type: Target model format - 'mistral', 'llama', or 'chatml'.
        system_prompt: Optional custom system prompt.
        include_output: Whether to include output in template (True for training).

    Returns:
        Formatted prompt string.

    Example:
        >>> prompt = create_prompt_template(
        ...     instruction="Classify this product",
        ...     input_text="iPhone 15 Pro",
        ...     output_text="Electronics > Mobile Phones",
        ...     model_type="mistral"
        ... )
    """
    if model_type not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown model_type: {model_type}")

    config = PROMPT_TEMPLATES[model_type]

    # Determine which template to use
    has_input = bool(input_text and input_text.strip())

    if model_type == "mistral":
        if has_input:
            prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
        else:
            prompt = f"<s>[INST] {instruction} [/INST]"

        if include_output and output_text:
            prompt = f"{prompt} {output_text}</s>"

    elif model_type == "llama":
        system = system_prompt or config.get("system", "")
        if has_input:
            prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction}\n\n{input_text} [/INST]"
        else:
            prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"

        if include_output and output_text:
            prompt = f"{prompt} {output_text}</s>"

    elif model_type == "chatml":
        system = system_prompt or config.get("system", "")
        if has_input:
            prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

        if include_output and output_text:
            prompt = f"{prompt}{output_text}<|im_end|>"

    return prompt


def create_inference_prompt(
    instruction: str,
    input_text: str = "",
    model_type: str = "mistral",
    task_type: Optional[str] = None,
) -> str:
    """
    Create a prompt for inference (without output).

    Args:
        instruction: The instruction/task description.
        input_text: Optional input context.
        model_type: Target model format.
        task_type: Optional task type to add prefix.

    Returns:
        Formatted inference prompt.

    Example:
        >>> prompt = create_inference_prompt(
        ...     instruction="Classify this product",
        ...     input_text="iPhone 15 Pro",
        ...     task_type="classification"
        ... )
    """
    # Add task prefix if specified
    if task_type and task_type in TASK_PREFIXES:
        if not instruction.startswith(TASK_PREFIXES[task_type]):
            instruction = f"{TASK_PREFIXES[task_type]} {instruction}"

    return create_prompt_template(
        instruction=instruction,
        input_text=input_text,
        model_type=model_type,
        include_output=False,
    )


def convert_alpaca_format(
    dataset: Dataset,
    model_type: str = "mistral",
) -> Dataset:
    """
    Convert Alpaca-format dataset to model-specific prompt format.

    Alpaca format has 'instruction', 'input', and 'output' fields.

    Args:
        dataset: Alpaca-format dataset.
        model_type: Target model format.

    Returns:
        Dataset with 'text' field containing formatted prompts.

    Example:
        >>> formatted = convert_alpaca_format(alpaca_dataset, model_type="mistral")
    """
    def format_example(example: Dict[str, Any]) -> Dict[str, str]:
        text = create_prompt_template(
            instruction=example.get("instruction", ""),
            input_text=example.get("input", ""),
            output_text=example.get("output", ""),
            model_type=model_type,
        )
        return {"text": text}

    return dataset.map(
        format_example,
        desc=f"Converting to {model_type} format",
    )


def convert_ecinstruct_format(
    dataset: Dataset,
    model_type: str = "mistral",
) -> Dataset:
    """
    Convert ECInstruct dataset to model-specific prompt format.

    Handles ECInstruct's specific field names and task types.

    Args:
        dataset: ECInstruct dataset.
        model_type: Target model format.

    Returns:
        Dataset with 'text' field containing formatted prompts.
    """
    def detect_task_type(instruction: str) -> str:
        """Detect task type from instruction text."""
        instruction_lower = instruction.lower()
        if any(kw in instruction_lower for kw in ["classify", "categorize", "category"]):
            return "classification"
        elif any(kw in instruction_lower for kw in ["extract", "attribute", "json"]):
            return "extraction"
        elif any(kw in instruction_lower for kw in ["answer", "question", "what", "how", "why"]):
            return "qa"
        return "qa"  # Default to QA

    def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")

        # Detect and add task prefix
        task_type = detect_task_type(instruction)
        task_prefix = TASK_PREFIXES.get(task_type, "")

        if task_prefix and not instruction.startswith(task_prefix):
            instruction = f"{task_prefix} {instruction}"

        text = create_prompt_template(
            instruction=instruction,
            input_text=input_text,
            output_text=output_text,
            model_type=model_type,
            system_prompt=TASK_SYSTEM_PROMPTS.get(task_type),
        )

        return {
            "text": text,
            "task_type": task_type,
        }

    return dataset.map(
        format_example,
        desc=f"Converting ECInstruct to {model_type} format",
    )


def validate_json_output(output: str) -> bool:
    """
    Validate that output is valid JSON.

    Useful for extraction task outputs.

    Args:
        output: Output string to validate.

    Returns:
        True if valid JSON, False otherwise.
    """
    try:
        json.loads(output)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def batch_format(
    examples: List[Dict[str, Any]],
    task_type: str,
    model_type: str = "mistral",
) -> List[str]:
    """
    Batch format multiple examples for efficiency.

    Args:
        examples: List of examples to format.
        task_type: Task type for all examples.
        model_type: Target model format.

    Returns:
        List of formatted prompt strings.
    """
    task_prefix = TASK_PREFIXES.get(task_type, "")

    formatted = []
    for example in examples:
        instruction = example.get("instruction", "")
        if task_prefix and not instruction.startswith(task_prefix):
            instruction = f"{task_prefix} {instruction}"

        text = create_prompt_template(
            instruction=instruction,
            input_text=example.get("input", ""),
            output_text=example.get("output", ""),
            model_type=model_type,
        )
        formatted.append(text)

    return formatted
