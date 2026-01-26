"""
Model loading utilities for inference.

This module provides utilities for loading QLoRA-trained models, merging LoRA
adapters, and performing inference on e-commerce tasks (classification,
attribute extraction, and Q&A).

Supports:
- Loading base models with optional LoRA adapters
- Various quantization methods (4-bit, 8-bit, AWQ)
- vLLM integration for high-throughput serving
- Batch inference for efficiency

Example usage:
    >>> from src.inference.model import EcommerceInference
    >>> inference = EcommerceInference("mistralai/Mistral-7B-Instruct-v0.3")
    >>> category = inference.classify("Apple iPhone 15 Pro 256GB Titanium Blue")
    >>> print(category)  # "Electronics > Communications > Telephony > Mobile Phones"
"""

from typing import Optional, List, Dict, Any, Tuple, Union
import logging
import json
import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel, PeftConfig

logger = logging.getLogger(__name__)


def load_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    quantization: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
    use_flash_attention: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model with optional LoRA adapter and quantization.

    This function handles the complexity of loading models in various configurations:
    - Base models (full precision or quantized)
    - Models with LoRA adapters applied
    - Different quantization schemes for memory efficiency

    Args:
        model_path: HuggingFace model ID or local path to the base model.
            Examples: "mistralai/Mistral-7B-Instruct-v0.3", "./models/my-model"
        adapter_path: Optional path to LoRA adapter weights. If provided,
            the adapter will be loaded and applied to the base model.
        quantization: Quantization method to use. Options:
            - None: Load in full precision (requires most VRAM)
            - "4bit": 4-bit NF4 quantization via bitsandbytes (8-12GB VRAM for 7B)
            - "8bit": 8-bit quantization via bitsandbytes (12-16GB VRAM for 7B)
            - "awq": Load AWQ-quantized model (requires pre-quantized weights)
        device_map: Device placement strategy.
            - "auto": Automatically distribute across available devices
            - "cuda:0": Load on specific GPU
            - "cpu": Load on CPU (slow, for testing only)
        torch_dtype: Data type for model weights. Defaults to float16 for GPU.
        trust_remote_code: Whether to trust and execute remote code in model files.
        use_flash_attention: Whether to use Flash Attention 2 for faster inference.

    Returns:
        Tuple of (model, tokenizer) ready for inference.

    Raises:
        ValueError: If invalid quantization method specified.
        RuntimeError: If model loading fails (e.g., OOM, missing files).

    Example:
        >>> # Load quantized model for memory-constrained environment
        >>> model, tokenizer = load_model(
        ...     "mistralai/Mistral-7B-Instruct-v0.3",
        ...     quantization="4bit",
        ...     device_map="auto"
        ... )

        >>> # Load base model with fine-tuned LoRA adapter
        >>> model, tokenizer = load_model(
        ...     "meta-llama/Meta-Llama-3-8B-Instruct",
        ...     adapter_path="./outputs/ecommerce-lora",
        ...     quantization="4bit"
        ... )

    Notes:
        - For production deployment, consider using vLLM instead of this function
          for much higher throughput via continuous batching and PagedAttention.
        - 4-bit quantization provides the best memory/quality tradeoff for 7B models.
        - AWQ quantization requires the model to be pre-quantized; it cannot be
          applied at load time like bitsandbytes quantization.
    """
    logger.info(f"Loading model from {model_path}")

    # Determine torch dtype
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Configure quantization
    quantization_config = None
    if quantization == "4bit":
        logger.info("Using 4-bit NF4 quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
        )
    elif quantization == "8bit":
        logger.info("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif quantization == "awq":
        # AWQ models are loaded directly - quantization is baked into weights
        logger.info("Loading AWQ-quantized model")
        pass
    elif quantization is not None:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Supported: '4bit', '8bit', 'awq', or None"
        )

    # Model loading kwargs
    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    # Enable Flash Attention 2 if available and requested
    if use_flash_attention and torch.cuda.is_available():
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        except Exception as e:
            logger.warning(f"Flash Attention 2 not available: {e}")

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    # Set padding token if not present (common for LLaMA-based models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA adapter if specified
    if adapter_path is not None:
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    # Set model to evaluation mode
    model.eval()

    logger.info(f"Model loaded successfully. Device: {next(model.parameters()).device}")
    return model, tokenizer


def merge_lora_weights(
    base_model: PreTrainedModel,
    adapter_path: str,
    output_path: Optional[str] = None,
    safe_serialization: bool = True,
) -> PreTrainedModel:
    """
    Merge LoRA adapter weights into the base model.

    Merging eliminates the adapter overhead during inference, resulting in
    a single model file that runs at full speed without the PEFT library.
    This is recommended for production deployment.

    Args:
        base_model: The base model (must NOT already have adapter applied).
        adapter_path: Path to the LoRA adapter weights.
        output_path: Optional path to save the merged model. If None,
            the merged model is only returned, not saved.
        safe_serialization: Use safetensors format (recommended for security).

    Returns:
        The merged model with LoRA weights baked in.

    Example:
        >>> # Merge and save for production deployment
        >>> base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        >>> merged = merge_lora_weights(
        ...     base_model,
        ...     adapter_path="./outputs/ecommerce-lora",
        ...     output_path="./outputs/merged-model"
        ... )

    Notes:
        - The merged model will be larger than the adapter alone (full model size).
        - After merging, quantization can be applied using AWQ or GPTQ.
        - For vLLM serving, merged models are preferred over adapter models.
    """
    logger.info(f"Merging LoRA weights from {adapter_path}")

    # Load adapter onto base model
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge weights
    model = model.merge_and_unload()

    logger.info("LoRA weights merged successfully")

    # Save if output path specified
    if output_path is not None:
        logger.info(f"Saving merged model to {output_path}")
        model.save_pretrained(output_path, safe_serialization=safe_serialization)

    return model


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.95,
    do_sample: bool = False,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    stop_strings: Optional[List[str]] = None,
) -> str:
    """
    Generate a response from the model given a prompt.

    Args:
        model: The loaded language model.
        tokenizer: The tokenizer corresponding to the model.
        prompt: The input prompt text.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. Use 0.0 for deterministic output
            (recommended for evaluation and production).
        top_p: Nucleus sampling probability threshold.
        do_sample: Whether to use sampling. Set False with temperature=0
            for deterministic greedy decoding.
        num_beams: Number of beams for beam search. 1 = greedy decoding.
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty).
        stop_strings: Optional list of strings that stop generation.

    Returns:
        The generated text response (excluding the input prompt).

    Example:
        >>> response = generate_response(
        ...     model, tokenizer,
        ...     prompt="[CLASSIFY] Product: Apple iPhone 15 Pro\\nCategory:",
        ...     max_new_tokens=64,
        ...     temperature=0.0
        ... )
        >>> print(response)
        "Electronics > Communications > Telephony > Mobile Phones"

    Notes:
        - Use temperature=0 during evaluation for reproducible results.
        - For production, consider batch_generate for higher throughput.
        - For very high throughput, use vLLM instead.
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generation config
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p if temperature > 0 else None,
        "do_sample": do_sample and temperature > 0,
        "num_beams": num_beams,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Remove None values
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    # Decode only the new tokens (exclude input prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0, input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Apply stop strings if specified
    if stop_strings:
        for stop_str in stop_strings:
            if stop_str in response:
                response = response.split(stop_str)[0]

    return response.strip()


def batch_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 8,
    show_progress: bool = True,
    **kwargs,
) -> List[str]:
    """
    Generate responses for multiple prompts in batches.

    Batched inference is more efficient than sequential calls due to
    better GPU utilization. However, for production workloads with
    high throughput requirements, consider using vLLM instead.

    Args:
        model: The loaded language model.
        tokenizer: The tokenizer corresponding to the model.
        prompts: List of input prompts.
        max_new_tokens: Maximum number of tokens to generate per response.
        temperature: Sampling temperature (0.0 for deterministic).
        batch_size: Number of prompts to process simultaneously.
            Larger batches are faster but use more memory.
        show_progress: Whether to show a progress bar.
        **kwargs: Additional arguments passed to generate_response.

    Returns:
        List of generated responses in the same order as input prompts.

    Example:
        >>> prompts = [
        ...     "[CLASSIFY] Product: iPhone 15\\nCategory:",
        ...     "[CLASSIFY] Product: Nike Air Max\\nCategory:",
        ...     "[CLASSIFY] Product: Instant Pot\\nCategory:",
        ... ]
        >>> responses = batch_generate(model, tokenizer, prompts, batch_size=4)
        >>> for prompt, response in zip(prompts, responses):
        ...     print(f"{prompt} -> {response}")

    Notes:
        - Memory usage scales with batch_size. Reduce if OOM errors occur.
        - For heterogeneous prompt lengths, padding overhead may reduce efficiency.
        - vLLM provides better batching via continuous batching.
    """
    try:
        from tqdm import tqdm
        progress_bar = tqdm if show_progress else lambda x, **kw: x
    except ImportError:
        progress_bar = lambda x, **kw: x

    responses = []

    # Process in batches
    for i in progress_bar(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize batch with padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generation config
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else None,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            **kwargs,
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        # Decode each response
        for j, output in enumerate(outputs):
            # Find where the actual input ends (accounting for padding)
            input_length = (inputs["attention_mask"][j] == 1).sum().item()
            generated_tokens = output[input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response.strip())

    return responses


class EcommerceInference:
    """
    High-level inference class for e-commerce tasks.

    This class provides a simplified interface for the three core e-commerce
    tasks: product classification, attribute extraction, and Q&A. It handles
    prompt formatting and response parsing internally.

    Attributes:
        model: The loaded language model.
        tokenizer: The tokenizer for the model.
        task_prefixes: Mapping of task names to prompt prefixes.

    Example:
        >>> inference = EcommerceInference(
        ...     model_path="mistralai/Mistral-7B-Instruct-v0.3",
        ...     adapter_path="./outputs/ecommerce-lora",
        ...     quantization="4bit"
        ... )

        >>> # Classification
        >>> category = inference.classify("Apple iPhone 15 Pro 256GB")
        >>> print(category)
        "Electronics > Communications > Telephony > Mobile Phones"

        >>> # Attribute extraction
        >>> attrs = inference.extract_attributes("Nike Air Max 90, Size 10, White/Black")
        >>> print(attrs)
        {"brand": "Nike", "model": "Air Max 90", "size": "10", "color": "White/Black"}

        >>> # Q&A
        >>> answer = inference.answer_question(
        ...     "Instant Pot Duo 7-in-1 Electric Pressure Cooker",
        ...     "What cooking functions does this have?"
        ... )
    """

    # Task-specific prompt prefixes as defined in the architecture
    TASK_PREFIXES = {
        "classify": "[CLASSIFY]",
        "extract": "[EXTRACT]",
        "qa": "[QA]",
    }

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        quantization: Optional[str] = "4bit",
        device_map: str = "auto",
        **model_kwargs,
    ):
        """
        Initialize the inference engine.

        Args:
            model_path: HuggingFace model ID or local path.
            adapter_path: Optional path to LoRA adapter.
            quantization: Quantization method ("4bit", "8bit", "awq", or None).
            device_map: Device placement strategy.
            **model_kwargs: Additional arguments for load_model.
        """
        self.model, self.tokenizer = load_model(
            model_path=model_path,
            adapter_path=adapter_path,
            quantization=quantization,
            device_map=device_map,
            **model_kwargs,
        )
        self._default_generation_kwargs = {
            "temperature": 0.0,  # Deterministic for consistency
            "max_new_tokens": 256,
        }

    def _format_prompt(self, task: str, content: str) -> str:
        """Format prompt with task prefix and instruction template."""
        prefix = self.TASK_PREFIXES.get(task, "")

        # Use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": f"{prefix} {content}"}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Fallback to simple format
        return f"{prefix} {content}"

    def classify(
        self,
        product_text: str,
        max_new_tokens: int = 128,
        **kwargs,
    ) -> str:
        """
        Classify a product into the Google Product Taxonomy.

        Args:
            product_text: Product title and/or description.
            max_new_tokens: Maximum tokens for the category path.
            **kwargs: Additional generation parameters.

        Returns:
            Category path string (e.g., "Electronics > Computers > Laptops").

        Example:
            >>> category = inference.classify(
            ...     "Samsung Galaxy S24 Ultra 512GB Titanium Black Smartphone"
            ... )
            >>> print(category)
            "Electronics > Communications > Telephony > Mobile Phones"

        Notes:
            - Returns hierarchical category using " > " separator.
            - Categories follow Google Product Taxonomy (5,595+ categories).
            - For batch classification, use classify_batch() instead.
        """
        prompt = self._format_prompt(
            "classify",
            f"Product: {product_text}\nCategory:"
        )

        response = generate_response(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            **{**self._default_generation_kwargs, **kwargs},
        )

        # Clean up response (remove potential extra content after category)
        # Categories typically don't have newlines
        response = response.split("\n")[0].strip()

        return response

    def extract_attributes(
        self,
        product_text: str,
        attributes: Optional[List[str]] = None,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Extract attribute-value pairs from product text.

        Args:
            product_text: Product title and/or description.
            attributes: Optional list of specific attributes to extract.
                If None, extracts all detected attributes.
            max_new_tokens: Maximum tokens for the JSON output.
            **kwargs: Additional generation parameters.

        Returns:
            Dictionary of attribute-value pairs.

        Example:
            >>> attrs = inference.extract_attributes(
            ...     "Apple MacBook Pro 14-inch M3 Pro, 18GB RAM, 512GB SSD, Space Black"
            ... )
            >>> print(attrs)
            {
                "brand": "Apple",
                "product_line": "MacBook Pro",
                "screen_size": "14-inch",
                "processor": "M3 Pro",
                "memory": "18GB",
                "storage": "512GB SSD",
                "color": "Space Black"
            }

        Notes:
            - Output is always a valid JSON dictionary.
            - If JSON parsing fails, returns {"raw_output": <response>}.
            - Common attributes: brand, model, color, size, material, etc.
        """
        if attributes:
            attr_str = ", ".join(attributes)
            content = f"Product: {product_text}\nExtract these attributes: {attr_str}\nAttributes (JSON):"
        else:
            content = f"Product: {product_text}\nAttributes (JSON):"

        prompt = self._format_prompt("extract", content)

        response = generate_response(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            **{**self._default_generation_kwargs, **kwargs},
        )

        # Parse JSON from response
        return self._parse_json_response(response)

    def answer_question(
        self,
        product_text: str,
        question: str,
        context: Optional[str] = None,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        """
        Answer a question about a product.

        Args:
            product_text: Product title and/or description.
            question: The question to answer.
            context: Optional additional context (e.g., from RAG retrieval).
            max_new_tokens: Maximum tokens for the answer.
            **kwargs: Additional generation parameters.

        Returns:
            The answer string.

        Example:
            >>> answer = inference.answer_question(
            ...     product_text="Instant Pot Duo 7-in-1 Electric Pressure Cooker, 6 Quart",
            ...     question="What cooking functions does this have?"
            ... )
            >>> print(answer)
            "The Instant Pot Duo 7-in-1 has the following cooking functions:
             pressure cooker, slow cooker, rice cooker, steamer, saute pan,
             yogurt maker, and warmer."

        Notes:
            - For better answers, provide context from RAG retrieval.
            - Answers are grounded in the provided product information.
            - Use with ChromaDB for RAG-enhanced Q&A.
        """
        if context:
            content = (
                f"Product: {product_text}\n"
                f"Context: {context}\n"
                f"Question: {question}\n"
                f"Answer:"
            )
        else:
            content = (
                f"Product: {product_text}\n"
                f"Question: {question}\n"
                f"Answer:"
            )

        prompt = self._format_prompt("qa", content)

        response = generate_response(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            **{**self._default_generation_kwargs, **kwargs},
        )

        return response.strip()

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from model response with fallback handling.

        The model may produce JSON with extra text before/after, or slightly
        malformed JSON. This method attempts multiple parsing strategies.
        """
        # Try direct parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from response (model might include extra text)
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try to find JSON with nested objects
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: return raw output
        logger.warning(f"Failed to parse JSON from response: {response[:100]}...")
        return {"raw_output": response, "_parse_error": True}

    def classify_batch(
        self,
        product_texts: List[str],
        batch_size: int = 8,
        **kwargs,
    ) -> List[str]:
        """
        Classify multiple products in batches.

        Args:
            product_texts: List of product titles/descriptions.
            batch_size: Number of products to process simultaneously.
            **kwargs: Additional generation parameters.

        Returns:
            List of category paths in the same order as inputs.
        """
        prompts = [
            self._format_prompt("classify", f"Product: {text}\nCategory:")
            for text in product_texts
        ]

        responses = batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            batch_size=batch_size,
            max_new_tokens=128,
            **{**self._default_generation_kwargs, **kwargs},
        )

        # Clean up responses
        return [r.split("\n")[0].strip() for r in responses]

    def extract_attributes_batch(
        self,
        product_texts: List[str],
        batch_size: int = 8,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Extract attributes from multiple products in batches.

        Args:
            product_texts: List of product titles/descriptions.
            batch_size: Number of products to process simultaneously.
            **kwargs: Additional generation parameters.

        Returns:
            List of attribute dictionaries in the same order as inputs.
        """
        prompts = [
            self._format_prompt("extract", f"Product: {text}\nAttributes (JSON):")
            for text in product_texts
        ]

        responses = batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            batch_size=batch_size,
            max_new_tokens=512,
            **{**self._default_generation_kwargs, **kwargs},
        )

        return [self._parse_json_response(r) for r in responses]


class VLLMInference:
    """
    High-throughput inference using vLLM server.

    This class provides the same interface as EcommerceInference but uses
    vLLM's OpenAI-compatible API for much higher throughput. Recommended
    for production deployments.

    vLLM provides:
    - Continuous batching for optimal GPU utilization
    - PagedAttention for efficient KV cache management
    - Up to 24x higher throughput than HuggingFace

    Example:
        >>> # Start vLLM server first:
        >>> # python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.3

        >>> inference = VLLMInference(base_url="http://localhost:8000/v1")
        >>> category = inference.classify("Apple iPhone 15 Pro")

    Notes:
        - Requires vLLM server running separately.
        - Uses OpenAI client for compatibility.
        - Prefix caching enabled for repeated template queries.
    """

    TASK_PREFIXES = EcommerceInference.TASK_PREFIXES

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        api_key: str = "EMPTY",  # vLLM doesn't require real key
    ):
        """
        Initialize vLLM inference client.

        Args:
            base_url: URL of the vLLM OpenAI-compatible API.
            model: Model name as registered in vLLM server.
            api_key: API key (use "EMPTY" for local vLLM server).
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self._default_kwargs = {
            "temperature": 0.0,
            "max_tokens": 256,
        }

    def _generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        """Generate response via vLLM API."""
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=kwargs.get("temperature", 0.0),
        )
        return response.choices[0].text.strip()

    def classify(self, product_text: str, **kwargs) -> str:
        """Classify a product (same interface as EcommerceInference)."""
        prompt = f"{self.TASK_PREFIXES['classify']} Product: {product_text}\nCategory:"
        response = self._generate(prompt, max_tokens=128, **kwargs)
        return response.split("\n")[0].strip()

    def extract_attributes(self, product_text: str, **kwargs) -> Dict[str, Any]:
        """Extract attributes (same interface as EcommerceInference)."""
        prompt = f"{self.TASK_PREFIXES['extract']} Product: {product_text}\nAttributes (JSON):"
        response = self._generate(prompt, max_tokens=512, **kwargs)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {"raw_output": response, "_parse_error": True}

    def answer_question(
        self,
        product_text: str,
        question: str,
        context: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Answer a question (same interface as EcommerceInference)."""
        if context:
            prompt = (
                f"{self.TASK_PREFIXES['qa']} Product: {product_text}\n"
                f"Context: {context}\nQuestion: {question}\nAnswer:"
            )
        else:
            prompt = (
                f"{self.TASK_PREFIXES['qa']} Product: {product_text}\n"
                f"Question: {question}\nAnswer:"
            )

        return self._generate(prompt, max_tokens=256, **kwargs)
