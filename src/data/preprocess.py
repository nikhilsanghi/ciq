"""
Data preprocessing for e-commerce LLM training.

Handles train/val/test splits, data mixing (90% e-commerce + 10% general),
and text cleaning utilities for consistent data preparation.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

from datasets import Dataset, DatasetDict, concatenate_datasets

logger = logging.getLogger(__name__)

# Default split ratios
DEFAULT_TRAIN_RATIO = 0.85
DEFAULT_VAL_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.05

# Data mixing ratios (prevent catastrophic forgetting)
ECOMMERCE_RATIO = 0.90
GENERAL_RATIO = 0.10


def preprocess_data(
    ecommerce_dataset: Dataset,
    general_dataset: Optional[Dataset] = None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    ecommerce_ratio: float = ECOMMERCE_RATIO,
    clean_text_fields: Optional[List[str]] = None,
    seed: int = 42,
) -> DatasetDict:
    """
    Main preprocessing function for e-commerce LLM training data.

    Performs the following steps:
    1. Cleans text fields if specified
    2. Creates train/val/test splits
    3. Mixes e-commerce data with general instruction data (if provided)

    Args:
        ecommerce_dataset: Primary e-commerce dataset (e.g., ECInstruct).
        general_dataset: General instruction dataset (e.g., Alpaca) for mixing.
        train_ratio: Proportion of data for training (default: 0.85).
        val_ratio: Proportion of data for validation (default: 0.10).
        test_ratio: Proportion of data for testing (default: 0.05).
        ecommerce_ratio: Ratio of e-commerce data in final mix (default: 0.90).
        clean_text_fields: List of field names to clean. If None, skips cleaning.
        seed: Random seed for reproducibility.

    Returns:
        DatasetDict with train, validation, and test splits.

    Raises:
        ValueError: If split ratios don't sum to 1.0.

    Example:
        >>> from src.data.download import download_ecinstruct, download_alpaca
        >>> ecom_data = download_ecinstruct()
        >>> alpaca_data = download_alpaca()
        >>> processed = preprocess_data(ecom_data, alpaca_data)
        >>> print(f"Train: {len(processed['train'])} examples")
    """
    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio:.3f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    logger.info(f"Preprocessing {len(ecommerce_dataset):,} e-commerce examples")

    # Step 1: Clean text fields if specified
    if clean_text_fields:
        logger.info(f"Cleaning text fields: {clean_text_fields}")
        ecommerce_dataset = _clean_dataset_fields(ecommerce_dataset, clean_text_fields)
        if general_dataset:
            general_dataset = _clean_dataset_fields(general_dataset, clean_text_fields)

    # Step 2: Create splits for e-commerce data
    ecom_splits = create_splits(
        ecommerce_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Step 3: Mix with general data if provided
    if general_dataset:
        logger.info(f"Mixing with {len(general_dataset):,} general instruction examples")
        logger.info(f"Mix ratio: {ecommerce_ratio*100:.0f}% e-commerce, {(1-ecommerce_ratio)*100:.0f}% general")

        general_splits = create_splits(
            general_dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        mixed_splits = mix_datasets(
            ecom_splits,
            general_splits,
            ecommerce_ratio=ecommerce_ratio,
            seed=seed,
        )
        return mixed_splits

    return ecom_splits


def create_splits(
    dataset: Dataset,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
) -> DatasetDict:
    """
    Create train/validation/test splits from a dataset.

    Args:
        dataset: Input dataset to split.
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.
        test_ratio: Proportion for test set.
        seed: Random seed for reproducibility.

    Returns:
        DatasetDict with 'train', 'validation', and 'test' splits.

    Example:
        >>> splits = create_splits(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        >>> print(f"Train: {len(splits['train'])}, Val: {len(splits['validation'])}")
    """
    logger.info(
        f"Creating splits: train={train_ratio:.0%}, "
        f"val={val_ratio:.0%}, test={test_ratio:.0%}"
    )

    # First split: separate test set
    train_val_test = dataset.train_test_split(
        test_size=test_ratio,
        seed=seed,
    )

    # Second split: separate train and validation from remaining data
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_val = train_val_test["train"].train_test_split(
        test_size=val_ratio_adjusted,
        seed=seed,
    )

    splits = DatasetDict({
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": train_val_test["test"],
    })

    logger.info(
        f"Split sizes: train={len(splits['train']):,}, "
        f"validation={len(splits['validation']):,}, "
        f"test={len(splits['test']):,}"
    )

    return splits


def mix_datasets(
    ecommerce_splits: DatasetDict,
    general_splits: DatasetDict,
    ecommerce_ratio: float = ECOMMERCE_RATIO,
    seed: int = 42,
) -> DatasetDict:
    """
    Mix e-commerce and general instruction datasets.

    Combines 90% e-commerce data with 10% general data by default
    to prevent catastrophic forgetting during fine-tuning.

    Args:
        ecommerce_splits: DatasetDict with e-commerce data splits.
        general_splits: DatasetDict with general instruction data splits.
        ecommerce_ratio: Ratio of e-commerce data (default: 0.90).
        seed: Random seed for shuffling.

    Returns:
        DatasetDict with mixed train/validation/test splits.

    Example:
        >>> mixed = mix_datasets(ecom_splits, alpaca_splits, ecommerce_ratio=0.9)
        >>> # Result: 90% e-commerce + 10% general instruction data
    """
    general_ratio = 1.0 - ecommerce_ratio

    mixed_splits = {}

    for split_name in ["train", "validation", "test"]:
        ecom_data = ecommerce_splits[split_name]
        general_data = general_splits[split_name]

        # Calculate target sizes based on e-commerce data as reference
        ecom_size = len(ecom_data)
        # general_size should be (general_ratio / ecommerce_ratio) * ecom_size
        target_general_size = int(ecom_size * (general_ratio / ecommerce_ratio))

        # Sample from general data if needed
        if len(general_data) > target_general_size:
            general_data = general_data.shuffle(seed=seed).select(range(target_general_size))
        elif len(general_data) < target_general_size:
            logger.warning(
                f"Not enough general data for {split_name} split. "
                f"Using all {len(general_data):,} examples instead of {target_general_size:,}"
            )

        # Add source label for tracking
        ecom_data = ecom_data.map(
            lambda x: {"_source": "ecommerce"},
            desc=f"Labeling e-commerce {split_name}",
        )
        general_data = general_data.map(
            lambda x: {"_source": "general"},
            desc=f"Labeling general {split_name}",
        )

        # Concatenate and shuffle
        mixed = concatenate_datasets([ecom_data, general_data])
        mixed = mixed.shuffle(seed=seed)

        mixed_splits[split_name] = mixed

        logger.info(
            f"{split_name}: {len(ecom_data):,} e-commerce + "
            f"{len(general_data):,} general = {len(mixed):,} total"
        )

    return DatasetDict(mixed_splits)


def clean_text(text: str) -> str:
    """
    Clean text for consistent preprocessing.

    Performs the following cleaning steps:
    1. Unicode normalization (NFKC)
    2. Remove control characters
    3. Normalize whitespace (collapse multiple spaces)
    4. Strip leading/trailing whitespace
    5. Remove null bytes and other problematic characters

    Args:
        text: Input text to clean.

    Returns:
        Cleaned text string.

    Example:
        >>> clean_text("  Hello   World  \\n\\n ")
        'Hello World'
    """
    if not text or not isinstance(text, str):
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove null bytes and control characters (except newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize newlines and tabs to spaces for single-line output
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Collapse multiple whitespace to single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def clean_text_preserve_structure(text: str) -> str:
    """
    Clean text while preserving paragraph structure.

    Similar to clean_text but keeps paragraph breaks (double newlines).
    Useful for longer text content where structure matters.

    Args:
        text: Input text to clean.

    Returns:
        Cleaned text with preserved paragraph structure.

    Example:
        >>> clean_text_preserve_structure("Hello\\n\\nWorld")
        'Hello\\n\\nWorld'
    """
    if not text or not isinstance(text, str):
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove null bytes and control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize paragraph breaks (2+ newlines become double newline)
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Collapse single newlines and multiple spaces within paragraphs
    paragraphs = text.split("\n\n")
    cleaned_paragraphs = []
    for para in paragraphs:
        para = re.sub(r"\s+", " ", para)
        para = para.strip()
        if para:
            cleaned_paragraphs.append(para)

    return "\n\n".join(cleaned_paragraphs)


def _clean_dataset_fields(
    dataset: Dataset,
    fields: List[str],
) -> Dataset:
    """
    Clean specified text fields in a dataset.

    Args:
        dataset: Input dataset.
        fields: List of field names to clean.

    Returns:
        Dataset with cleaned text fields.
    """
    def clean_example(example: Dict[str, Any]) -> Dict[str, Any]:
        for field in fields:
            if field in example and isinstance(example[field], str):
                example[field] = clean_text(example[field])
        return example

    return dataset.map(
        clean_example,
        desc=f"Cleaning fields: {fields}",
    )


def filter_by_length(
    dataset: Dataset,
    text_field: str,
    min_length: int = 10,
    max_length: int = 8192,
) -> Dataset:
    """
    Filter dataset examples by text length.

    Args:
        dataset: Input dataset.
        text_field: Name of the text field to check length.
        min_length: Minimum character length (default: 10).
        max_length: Maximum character length (default: 8192).

    Returns:
        Filtered dataset.

    Example:
        >>> filtered = filter_by_length(dataset, "text", min_length=50, max_length=4096)
    """
    original_size = len(dataset)

    def length_filter(example: Dict[str, Any]) -> bool:
        text = example.get(text_field, "")
        if not isinstance(text, str):
            return False
        return min_length <= len(text) <= max_length

    filtered = dataset.filter(length_filter, desc="Filtering by length")

    removed = original_size - len(filtered)
    if removed > 0:
        logger.info(
            f"Filtered {removed:,} examples ({removed/original_size:.1%}) "
            f"outside length range [{min_length}, {max_length}]"
        )

    return filtered


def deduplicate(
    dataset: Dataset,
    text_field: str,
) -> Dataset:
    """
    Remove duplicate examples based on a text field.

    Args:
        dataset: Input dataset.
        text_field: Field to use for deduplication.

    Returns:
        Deduplicated dataset.

    Example:
        >>> deduped = deduplicate(dataset, "instruction")
    """
    original_size = len(dataset)

    seen = set()
    indices_to_keep = []

    for idx, example in enumerate(dataset):
        text = example.get(text_field, "")
        text_hash = hash(text)
        if text_hash not in seen:
            seen.add(text_hash)
            indices_to_keep.append(idx)

    deduped = dataset.select(indices_to_keep)

    removed = original_size - len(deduped)
    if removed > 0:
        logger.info(
            f"Removed {removed:,} duplicate examples ({removed/original_size:.1%})"
        )

    return deduped


def stratified_sample(
    dataset: Dataset,
    label_field: str,
    n_samples: int,
    seed: int = 42,
) -> Dataset:
    """
    Create a stratified sample maintaining label distribution.

    Args:
        dataset: Input dataset.
        label_field: Field containing labels for stratification.
        n_samples: Total number of samples to select.
        seed: Random seed for reproducibility.

    Returns:
        Stratified sample of the dataset.

    Example:
        >>> sample = stratified_sample(dataset, "category", n_samples=1000)
    """
    from collections import Counter

    # Count label distribution
    labels = [example[label_field] for example in dataset]
    label_counts = Counter(labels)

    # Calculate samples per label (proportional)
    total_examples = len(dataset)
    samples_per_label = {
        label: max(1, int(count / total_examples * n_samples))
        for label, count in label_counts.items()
    }

    # Group indices by label
    label_indices: Dict[Any, List[int]] = {label: [] for label in label_counts}
    for idx, example in enumerate(dataset):
        label_indices[example[label_field]].append(idx)

    # Sample from each label group
    import random
    random.seed(seed)

    selected_indices = []
    for label, indices in label_indices.items():
        n_to_sample = min(samples_per_label[label], len(indices))
        selected = random.sample(indices, n_to_sample)
        selected_indices.extend(selected)

    # Shuffle final selection
    random.shuffle(selected_indices)

    sampled = dataset.select(selected_indices[:n_samples])
    logger.info(f"Stratified sample: {len(sampled):,} examples from {len(label_counts)} labels")

    return sampled
