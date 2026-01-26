#!/usr/bin/env python3
"""
Dataset Download Script for E-Commerce LLM Project

Downloads all required datasets from HuggingFace and saves them locally.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --include-optional  # Also download larger optional datasets

Required datasets (~550 MB):
    - ECInstruct: 116K multi-task e-commerce examples
    - Alpaca: 52K general instruction examples (prevents catastrophic forgetting)

Optional datasets (~10 GB):
    - MAVE: 3M attribute annotations
    - Amazon ESCI: 2.6M relevance judgments
    - AmazonQA: 923K product Q&A pairs
"""

import os
import argparse
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package not installed.")
    print("Install with: pip install datasets")
    exit(1)


def download_ecinstruct(output_dir: Path) -> int:
    """Download ECInstruct - main e-commerce training dataset."""
    print("\n" + "=" * 60)
    print("Downloading ECInstruct (Primary Training Data)")
    print("Source: https://huggingface.co/datasets/NingLab/ECInstruct")
    print("=" * 60)

    dataset = load_dataset("NingLab/ECInstruct", split="train")
    output_path = output_dir / "ecinstruct.jsonl"
    dataset.to_json(str(output_path))

    print(f"✓ Saved {len(dataset):,} examples to {output_path}")
    return len(dataset)


def download_alpaca(output_dir: Path) -> int:
    """Download Alpaca - general instruction data to prevent catastrophic forgetting."""
    print("\n" + "=" * 60)
    print("Downloading Alpaca (General Instructions)")
    print("Source: https://huggingface.co/datasets/tatsu-lab/alpaca")
    print("=" * 60)

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    output_path = output_dir / "alpaca.jsonl"
    dataset.to_json(str(output_path))

    print(f"✓ Saved {len(dataset):,} examples to {output_path}")
    return len(dataset)


def download_esci(output_dir: Path) -> int:
    """Download Amazon ESCI - search relevance dataset."""
    print("\n" + "=" * 60)
    print("Downloading Amazon ESCI (Search Relevance)")
    print("Source: https://github.com/amazon-science/esci-data")
    print("=" * 60)

    try:
        # ESCI is available on HuggingFace as well
        dataset = load_dataset("tasksource/esci", split="train")
        output_path = output_dir / "esci.jsonl"
        dataset.to_json(str(output_path))
        print(f"✓ Saved {len(dataset):,} examples to {output_path}")
        return len(dataset)
    except Exception as e:
        print(f"⚠ Could not download ESCI from HuggingFace: {e}")
        print("  Manual download: git clone https://github.com/amazon-science/esci-data.git")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for E-Commerce LLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_datasets.py                    # Required datasets only
    python scripts/download_datasets.py --include-optional # All datasets
    python scripts/download_datasets.py --output ./mydata  # Custom output directory
        """
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/raw",
        help="Output directory for downloaded datasets (default: ./data/raw)"
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also download optional datasets (ESCI, MAVE, AmazonQA)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("E-Commerce LLM Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")

    total_examples = 0

    # Required datasets
    print("\n📦 REQUIRED DATASETS")
    print("-" * 40)

    total_examples += download_ecinstruct(output_dir)
    total_examples += download_alpaca(output_dir)

    # Optional datasets
    if args.include_optional:
        print("\n📦 OPTIONAL DATASETS")
        print("-" * 40)
        total_examples += download_esci(output_dir)

        print("\n⚠ Note: MAVE and AmazonQA require manual download:")
        print("  MAVE: gsutil -m cp -r gs://mave_dataset/ ./data/mave/")
        print("  AmazonQA: git clone https://github.com/amazonqa/amazonqa.git")

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Total examples downloaded: {total_examples:,}")
    print(f"Files saved to: {output_dir.absolute()}")
    print("\nContents:")
    for f in sorted(output_dir.glob("*.jsonl")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}: {size_mb:.1f} MB")

    print("\n✅ Ready for training!")
    print("Next step: python -m src.training.trainer --train_data ./data/raw/ecinstruct.jsonl")


if __name__ == "__main__":
    main()
