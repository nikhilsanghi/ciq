"""
Dataset download utilities for e-commerce LLM training.

Downloads and caches the following datasets:
- ECInstruct (primary): 116K multi-task e-commerce examples
- MAVE: 3M attribute annotations from Google
- Amazon ESCI: 2.6M relevance judgments
- AmazonQA: 923K product Q&A pairs
- Alpaca (general): 52K general instruction data for preventing catastrophic forgetting
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from datasets import load_dataset, Dataset, DatasetDict

logger = logging.getLogger(__name__)

# Dataset registry with metadata
DATASETS: Dict[str, Dict[str, str]] = {
    "ecinstruct": {
        "name": "NingLab/ECInstruct",
        "description": "116K multi-task e-commerce examples",
        "splits": "train",
    },
    "mave": {
        "name": "google-research-datasets/MAVE",
        "description": "3M attribute annotations",
        "splits": "train,validation,test",
    },
    "esci": {
        "name": "amazon-science/esci-data",
        "description": "2.6M relevance judgments",
        "splits": "train,test",
    },
    "amazonqa": {
        "name": "amazonqa/amazonqa",
        "description": "923K product Q&A pairs",
        "splits": "train,test",
    },
    "alpaca": {
        "name": "tatsu-lab/alpaca",
        "description": "52K general instruction data",
        "splits": "train",
    },
}


def download_datasets(
    dataset_names: Optional[List[str]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
) -> Dict[str, Union[Dataset, DatasetDict]]:
    """
    Download specified datasets and cache them locally.

    Args:
        dataset_names: List of dataset names to download. If None, downloads all.
                      Valid names: ecinstruct, mave, esci, amazonqa, alpaca
        cache_dir: Directory to cache downloaded datasets. Uses HF default if None.
        force_download: If True, re-download even if cached.

    Returns:
        Dictionary mapping dataset names to loaded Dataset/DatasetDict objects.

    Raises:
        ValueError: If an invalid dataset name is provided.
        Exception: If download fails for any dataset.

    Example:
        >>> datasets = download_datasets(["ecinstruct", "alpaca"])
        >>> print(datasets["ecinstruct"])
    """
    if dataset_names is None:
        dataset_names = list(DATASETS.keys())

    # Validate dataset names
    invalid_names = set(dataset_names) - set(DATASETS.keys())
    if invalid_names:
        raise ValueError(
            f"Invalid dataset names: {invalid_names}. "
            f"Valid options: {list(DATASETS.keys())}"
        )

    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using cache directory: {cache_path}")

    downloaded: Dict[str, Union[Dataset, DatasetDict]] = {}

    for name in dataset_names:
        dataset_info = DATASETS[name]
        logger.info(f"Downloading {name}: {dataset_info['description']}")

        try:
            dataset = load_dataset(
                dataset_info["name"],
                cache_dir=str(cache_path) if cache_path else None,
                download_mode="force_redownload" if force_download else None,
                trust_remote_code=True,
            )
            downloaded[name] = dataset
            logger.info(f"Successfully downloaded {name}")

            # Log dataset statistics
            if isinstance(dataset, DatasetDict):
                for split_name, split_data in dataset.items():
                    logger.info(f"  {split_name}: {len(split_data):,} examples")
            else:
                logger.info(f"  Total: {len(dataset):,} examples")

        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            raise

    return downloaded


def download_ecinstruct(
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
) -> Dataset:
    """
    Download the ECInstruct dataset - primary e-commerce training data.

    ECInstruct contains 116K multi-task e-commerce instruction examples
    covering classification, extraction, and Q&A tasks.

    Args:
        cache_dir: Directory to cache the dataset.
        force_download: If True, re-download even if cached.

    Returns:
        The ECInstruct dataset.

    Example:
        >>> dataset = download_ecinstruct()
        >>> print(f"Downloaded {len(dataset)} examples")
    """
    logger.info("Downloading ECInstruct dataset (116K multi-task e-commerce examples)")

    try:
        dataset = load_dataset(
            DATASETS["ecinstruct"]["name"],
            cache_dir=str(cache_dir) if cache_dir else None,
            download_mode="force_redownload" if force_download else None,
            trust_remote_code=True,
        )

        # ECInstruct typically has a single train split
        if isinstance(dataset, DatasetDict) and "train" in dataset:
            dataset = dataset["train"]

        logger.info(f"Successfully downloaded ECInstruct: {len(dataset):,} examples")
        return dataset

    except Exception as e:
        logger.error(f"Failed to download ECInstruct: {e}")
        raise


def download_alpaca(
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
) -> Dataset:
    """
    Download the Alpaca dataset for general instruction following.

    Alpaca contains 52K general instruction examples used to prevent
    catastrophic forgetting during domain-specific fine-tuning.
    Recommended to mix ~10% Alpaca with e-commerce data.

    Args:
        cache_dir: Directory to cache the dataset.
        force_download: If True, re-download even if cached.

    Returns:
        The Alpaca dataset.

    Example:
        >>> alpaca = download_alpaca()
        >>> print(f"Downloaded {len(alpaca)} general instruction examples")
    """
    logger.info("Downloading Alpaca dataset (52K general instruction examples)")

    try:
        dataset = load_dataset(
            DATASETS["alpaca"]["name"],
            cache_dir=str(cache_dir) if cache_dir else None,
            download_mode="force_redownload" if force_download else None,
            trust_remote_code=True,
        )

        # Alpaca has a single train split
        if isinstance(dataset, DatasetDict) and "train" in dataset:
            dataset = dataset["train"]

        logger.info(f"Successfully downloaded Alpaca: {len(dataset):,} examples")
        return dataset

    except Exception as e:
        logger.error(f"Failed to download Alpaca: {e}")
        raise


def download_mave(
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
) -> DatasetDict:
    """
    Download the MAVE dataset for attribute extraction.

    MAVE contains 3M attribute annotations from Google Research,
    useful for training attribute extraction models.

    Args:
        cache_dir: Directory to cache the dataset.
        force_download: If True, re-download even if cached.

    Returns:
        The MAVE dataset with train/validation/test splits.

    Example:
        >>> mave = download_mave()
        >>> print(f"Train: {len(mave['train'])} examples")
    """
    logger.info("Downloading MAVE dataset (3M attribute annotations)")

    try:
        dataset = load_dataset(
            DATASETS["mave"]["name"],
            cache_dir=str(cache_dir) if cache_dir else None,
            download_mode="force_redownload" if force_download else None,
            trust_remote_code=True,
        )

        if isinstance(dataset, DatasetDict):
            for split_name, split_data in dataset.items():
                logger.info(f"  {split_name}: {len(split_data):,} examples")

        return dataset

    except Exception as e:
        logger.error(f"Failed to download MAVE: {e}")
        raise


def download_amazonqa(
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
) -> DatasetDict:
    """
    Download the AmazonQA dataset for product Q&A.

    AmazonQA contains 923K product question-answer pairs,
    useful for training Q&A models with RAG.

    Args:
        cache_dir: Directory to cache the dataset.
        force_download: If True, re-download even if cached.

    Returns:
        The AmazonQA dataset.

    Example:
        >>> qa_data = download_amazonqa()
        >>> print(f"Downloaded {len(qa_data['train'])} Q&A pairs")
    """
    logger.info("Downloading AmazonQA dataset (923K product Q&A pairs)")

    try:
        dataset = load_dataset(
            DATASETS["amazonqa"]["name"],
            cache_dir=str(cache_dir) if cache_dir else None,
            download_mode="force_redownload" if force_download else None,
            trust_remote_code=True,
        )

        if isinstance(dataset, DatasetDict):
            for split_name, split_data in dataset.items():
                logger.info(f"  {split_name}: {len(split_data):,} examples")

        return dataset

    except Exception as e:
        logger.error(f"Failed to download AmazonQA: {e}")
        raise


def download_esci(
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
) -> DatasetDict:
    """
    Download the Amazon ESCI dataset for relevance judgments.

    ESCI contains 2.6M relevance judgments for e-commerce search,
    useful for training ranking and relevance models.

    Args:
        cache_dir: Directory to cache the dataset.
        force_download: If True, re-download even if cached.

    Returns:
        The ESCI dataset.

    Example:
        >>> esci = download_esci()
        >>> print(f"Downloaded {len(esci['train'])} relevance judgments")
    """
    logger.info("Downloading ESCI dataset (2.6M relevance judgments)")

    try:
        dataset = load_dataset(
            DATASETS["esci"]["name"],
            cache_dir=str(cache_dir) if cache_dir else None,
            download_mode="force_redownload" if force_download else None,
            trust_remote_code=True,
        )

        if isinstance(dataset, DatasetDict):
            for split_name, split_data in dataset.items():
                logger.info(f"  {split_name}: {len(split_data):,} examples")

        return dataset

    except Exception as e:
        logger.error(f"Failed to download ESCI: {e}")
        raise


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get metadata information about a dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        Dictionary with dataset metadata.

    Raises:
        ValueError: If dataset name is not recognized.
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Valid options: {list(DATASETS.keys())}"
        )

    return DATASETS[dataset_name].copy()


def list_available_datasets() -> List[str]:
    """
    List all available datasets for download.

    Returns:
        List of dataset names.
    """
    return list(DATASETS.keys())
