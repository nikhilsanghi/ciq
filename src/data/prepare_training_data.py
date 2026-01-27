"""
Data Preparation Script for E-Commerce LLM Fine-tuning v2.

Converts raw datasets into unified training format for all 3 tasks:
1. Classification - Google Taxonomy categories from MAVE product categories
2. Extraction - MAVE attribute annotations
3. Q&A - AmazonQA question-answer pairs

CRITICAL LESSON FROM v1: Always inspect data format BEFORE training!
This script includes sample output preview for verification.

Usage:
    python -m src.data.prepare_training_data \
        --mave_dir data/raw/mave \
        --amazonqa_dir data/raw/amazonqa \
        --taxonomy_file data/raw/taxonomy/google_taxonomy.txt \
        --output_dir data/processed \
        --max_samples 50000
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Task prefixes - must match inference prompts exactly!
TASK_PREFIXES = {
    "classify": "[CLASSIFY]",
    "extract": "[EXTRACT]",
    "qa": "[QA]",
}


def load_taxonomy(taxonomy_file: str) -> List[str]:
    """
    Load Google Product Taxonomy categories.

    Format: One category per line, hierarchy separated by " > "
    Example: "Electronics > Audio > Headphones"
    """
    categories = []
    with open(taxonomy_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                categories.append(line)

    logger.info(f"Loaded {len(categories)} taxonomy categories")
    return categories


def process_mave_for_extraction(mave_dir: str, max_samples: int = 20000) -> List[Dict[str, str]]:
    """
    Process MAVE dataset for attribute extraction task.

    MAVE format (from GitHub):
    - Contains product paragraphs with attribute annotations
    - Attributes have name, values with text and span positions

    Output format:
    {
        "instruction": "[EXTRACT] Extract product attributes as JSON.",
        "input": "Apple iPhone 15 Pro 256GB Natural Titanium",
        "output": '{"brand": "Apple", "model": "iPhone 15 Pro", "storage": "256GB", "color": "Natural Titanium"}'
    }
    """
    examples = []
    mave_path = Path(mave_dir)

    # Look for MAVE data files
    data_files = list(mave_path.glob("**/*.jsonl")) + list(mave_path.glob("**/*.json"))

    if not data_files:
        logger.warning(f"No MAVE data files found in {mave_dir}")
        logger.info("MAVE requires downloading Amazon Review Data 2018 separately")
        logger.info("See: https://github.com/google-research-datasets/MAVE")
        return []

    for data_file in data_files:
        logger.info(f"Processing MAVE file: {data_file}")
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if len(examples) >= max_samples:
                        break

                    try:
                        item = json.loads(line.strip())

                        # Extract product text from paragraphs
                        product_text = ""
                        if "paragraphs" in item:
                            for para in item["paragraphs"]:
                                if para.get("source") == "title":
                                    product_text = para.get("text", "")
                                    break
                            if not product_text and item["paragraphs"]:
                                product_text = item["paragraphs"][0].get("text", "")
                        elif "title" in item:
                            product_text = item["title"]
                        elif "text" in item:
                            product_text = item["text"]

                        if not product_text:
                            continue

                        # Extract attributes
                        attributes = {}
                        if "attributes" in item:
                            for attr in item["attributes"]:
                                attr_name = attr.get("key", attr.get("name", ""))
                                attr_values = attr.get("values", [])
                                if attr_values and attr_name:
                                    # Get first value text
                                    if isinstance(attr_values[0], dict):
                                        attr_value = attr_values[0].get("text", attr_values[0].get("value", ""))
                                    else:
                                        attr_value = str(attr_values[0])
                                    if attr_value:
                                        attributes[attr_name.lower().replace(" ", "_")] = attr_value

                        if attributes:
                            example = {
                                "instruction": f"{TASK_PREFIXES['extract']} Extract product attributes as JSON.",
                                "input": f"Product: {product_text}",
                                "output": json.dumps(attributes, ensure_ascii=False)
                            }
                            examples.append(example)

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Error processing {data_file}: {e}")

    logger.info(f"Created {len(examples)} extraction examples from MAVE")
    return examples


def process_mave_for_classification(mave_dir: str, taxonomy: List[str], max_samples: int = 15000) -> List[Dict[str, str]]:
    """
    Process MAVE dataset for classification task.

    Uses MAVE product categories and maps to Google Taxonomy where possible.

    Output format:
    {
        "instruction": "[CLASSIFY] Classify into Google Product Taxonomy.",
        "input": "Sony WH-1000XM5 Wireless Headphones",
        "output": "Electronics > Audio > Headphones"
    }
    """
    examples = []
    mave_path = Path(mave_dir)

    # Build taxonomy lookup for fuzzy matching
    taxonomy_lookup = {cat.lower(): cat for cat in taxonomy}
    taxonomy_parts = defaultdict(list)
    for cat in taxonomy:
        parts = cat.split(" > ")
        for i, part in enumerate(parts):
            taxonomy_parts[part.lower()].append(cat)

    data_files = list(mave_path.glob("**/*.jsonl")) + list(mave_path.glob("**/*.json"))

    for data_file in data_files:
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(examples) >= max_samples:
                        break

                    try:
                        item = json.loads(line.strip())

                        # Get product text
                        product_text = ""
                        if "paragraphs" in item:
                            for para in item["paragraphs"]:
                                if para.get("source") == "title":
                                    product_text = para.get("text", "")
                                    break
                        elif "title" in item:
                            product_text = item["title"]

                        if not product_text:
                            continue

                        # Get category
                        category = item.get("category", "")
                        if not category:
                            continue

                        # Try to map to Google Taxonomy
                        mapped_category = category
                        category_lower = category.lower()

                        # Direct match
                        if category_lower in taxonomy_lookup:
                            mapped_category = taxonomy_lookup[category_lower]
                        else:
                            # Try matching last part of category
                            parts = category.split(" > ")
                            if parts:
                                last_part = parts[-1].lower()
                                if last_part in taxonomy_parts:
                                    mapped_category = taxonomy_parts[last_part][0]

                        example = {
                            "instruction": f"{TASK_PREFIXES['classify']} Classify into Google Product Taxonomy.",
                            "input": f"Product: {product_text}",
                            "output": mapped_category
                        }
                        examples.append(example)

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Error processing {data_file}: {e}")

    logger.info(f"Created {len(examples)} classification examples from MAVE")
    return examples


def process_amazonqa(amazonqa_dir: str, max_samples: int = 15000) -> List[Dict[str, str]]:
    """
    Process AmazonQA dataset for Q&A task.

    AmazonQA format:
    {
        "questionText": "Is this compatible with iPhone 14?",
        "questionType": "yesno" or "descriptive",
        "asin": "B0EXAMPLE",
        "answers": [{"answerText": "Yes, it works great!", "helpful": [10, 12]}],
        "review_snippets": ["Works with all iPhones..."]
    }

    Output format:
    {
        "instruction": "[QA] Answer the question about this product.",
        "input": "Product: iPhone 14 Case with MagSafe\nQuestion: Is this compatible with iPhone 14?",
        "output": "Yes, it works great!"
    }
    """
    examples = []
    amazonqa_path = Path(amazonqa_dir)

    # Find data files
    data_files = list(amazonqa_path.glob("*.jsonl"))

    if not data_files:
        logger.warning(f"No AmazonQA files found in {amazonqa_dir}")
        return []

    for data_file in data_files:
        logger.info(f"Processing AmazonQA file: {data_file}")
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(examples) >= max_samples:
                        break

                    try:
                        item = json.loads(line.strip())

                        question = item.get("questionText", item.get("question", ""))
                        if not question:
                            continue

                        # Get answers
                        answers = item.get("answers", [])
                        if not answers:
                            continue

                        # Get best answer (highest helpful score)
                        best_answer = None
                        best_score = -1
                        for ans in answers:
                            ans_text = ans.get("answerText", ans.get("answer", ""))
                            if not ans_text:
                                continue
                            helpful = ans.get("helpful", [0, 0])
                            if isinstance(helpful, list) and len(helpful) >= 2:
                                score = helpful[0] / max(helpful[1], 1)
                            else:
                                score = 0
                            if score > best_score or best_answer is None:
                                best_answer = ans_text
                                best_score = score

                        if not best_answer:
                            continue

                        # Get product context from review snippets or product info
                        product_context = ""
                        if "review_snippets" in item and item["review_snippets"]:
                            # Use first review snippet as context
                            product_context = item["review_snippets"][0][:200]
                        elif "product_title" in item:
                            product_context = item["product_title"]
                        elif "asin" in item:
                            product_context = f"Product ASIN: {item['asin']}"

                        # Build example
                        input_text = f"Product: {product_context}\nQuestion: {question}"

                        example = {
                            "instruction": f"{TASK_PREFIXES['qa']} Answer the question about this product.",
                            "input": input_text,
                            "output": best_answer
                        }
                        examples.append(example)

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Error processing {data_file}: {e}")

    logger.info(f"Created {len(examples)} Q&A examples from AmazonQA")
    return examples


def add_general_instruction_data(num_samples: int = 5000) -> List[Dict[str, str]]:
    """
    Add general instruction data to prevent catastrophic forgetting.

    Uses Alpaca dataset from HuggingFace.
    """
    try:
        from datasets import load_dataset

        logger.info("Loading Alpaca dataset for general instruction data...")
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")

        # Sample and convert
        indices = random.sample(range(len(alpaca)), min(num_samples, len(alpaca)))
        examples = []

        for idx in indices:
            item = alpaca[idx]
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")

            if instruction and output:
                if input_text:
                    full_input = f"{instruction}\n\nInput: {input_text}"
                else:
                    full_input = instruction

                example = {
                    "instruction": "[GENERAL] " + instruction[:100],  # Truncate long instructions
                    "input": full_input,
                    "output": output
                }
                examples.append(example)

        logger.info(f"Added {len(examples)} general instruction examples")
        return examples

    except Exception as e:
        logger.warning(f"Could not load Alpaca dataset: {e}")
        logger.warning("Skipping general instruction data")
        return []


def create_training_splits(
    all_examples: List[Dict[str, str]],
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Create train/validation splits.
    """
    random.seed(seed)
    random.shuffle(all_examples)

    split_idx = int(len(all_examples) * train_ratio)
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]

    return train_data, val_data


def preview_samples(examples: List[Dict[str, str]], task_name: str, num_samples: int = 3):
    """
    Preview sample outputs - CRITICAL for validation before training!
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PREVIEW: {task_name} ({len(examples)} total)")
    logger.info(f"{'='*60}")

    for i, ex in enumerate(examples[:num_samples]):
        logger.info(f"\n--- Sample {i+1} ---")
        logger.info(f"Instruction: {ex['instruction']}")
        logger.info(f"Input: {ex['input'][:200]}...")
        logger.info(f"Output: {ex['output'][:200]}...")

    logger.info(f"\n{'='*60}\n")


def save_jsonl(data: List[Dict[str, str]], filepath: str):
    """Save data as JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(data)} examples to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for e-commerce LLM")
    parser.add_argument("--mave_dir", type=str, default="data/raw/mave",
                       help="Directory containing MAVE data")
    parser.add_argument("--amazonqa_dir", type=str, default="data/raw/amazonqa",
                       help="Directory containing AmazonQA data")
    parser.add_argument("--taxonomy_file", type=str, default="data/raw/taxonomy/google_taxonomy.txt",
                       help="Google Product Taxonomy file")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--max_samples", type=int, default=50000,
                       help="Maximum total samples (distributed across tasks)")
    parser.add_argument("--include_general", action="store_true",
                       help="Include general instruction data (Alpaca)")
    parser.add_argument("--general_ratio", type=float, default=0.1,
                       help="Ratio of general instruction data (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate samples per task
    samples_per_task = args.max_samples // 3

    logger.info("="*60)
    logger.info("E-Commerce LLM Data Preparation v2")
    logger.info("="*60)
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Samples per task: ~{samples_per_task}")
    logger.info("="*60)

    # Load taxonomy
    taxonomy = []
    if Path(args.taxonomy_file).exists():
        taxonomy = load_taxonomy(args.taxonomy_file)
    else:
        logger.warning(f"Taxonomy file not found: {args.taxonomy_file}")

    # Process each dataset
    all_examples = []

    # 1. Classification from MAVE
    classification_examples = process_mave_for_classification(
        args.mave_dir, taxonomy, max_samples=samples_per_task
    )
    if classification_examples:
        preview_samples(classification_examples, "Classification")
        all_examples.extend(classification_examples)

    # 2. Extraction from MAVE
    extraction_examples = process_mave_for_extraction(
        args.mave_dir, max_samples=samples_per_task
    )
    if extraction_examples:
        preview_samples(extraction_examples, "Extraction")
        all_examples.extend(extraction_examples)

    # 3. Q&A from AmazonQA
    qa_examples = process_amazonqa(
        args.amazonqa_dir, max_samples=samples_per_task
    )
    if qa_examples:
        preview_samples(qa_examples, "Q&A")
        all_examples.extend(qa_examples)

    # 4. General instruction data (optional)
    if args.include_general:
        general_samples = int(len(all_examples) * args.general_ratio / (1 - args.general_ratio))
        general_examples = add_general_instruction_data(num_samples=general_samples)
        if general_examples:
            preview_samples(general_examples, "General Instructions")
            all_examples.extend(general_examples)

    if not all_examples:
        logger.error("No examples created! Check your data paths.")
        return

    # Create splits
    train_data, val_data = create_training_splits(all_examples, seed=args.seed)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("DATA SUMMARY")
    logger.info("="*60)
    logger.info(f"Classification examples: {len(classification_examples)}")
    logger.info(f"Extraction examples: {len(extraction_examples)}")
    logger.info(f"Q&A examples: {len(qa_examples)}")
    if args.include_general:
        logger.info(f"General examples: {len(general_examples)}")
    logger.info(f"Total examples: {len(all_examples)}")
    logger.info(f"Train split: {len(train_data)}")
    logger.info(f"Validation split: {len(val_data)}")
    logger.info("="*60)

    # Save
    save_jsonl(train_data, str(output_dir / "train.jsonl"))
    save_jsonl(val_data, str(output_dir / "val.jsonl"))

    # Save task-specific files for analysis
    if classification_examples:
        save_jsonl(classification_examples, str(output_dir / "classification.jsonl"))
    if extraction_examples:
        save_jsonl(extraction_examples, str(output_dir / "extraction.jsonl"))
    if qa_examples:
        save_jsonl(qa_examples, str(output_dir / "qa.jsonl"))

    logger.info("\nData preparation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("\nNEXT STEPS:")
    logger.info("1. Review the preview samples above")
    logger.info("2. Inspect: head -5 data/processed/train.jsonl | python -m json.tool")
    logger.info("3. Test prompts on base model before training")
    logger.info("4. Train: python -m src.training.trainer --train_data data/processed/train.jsonl --eval_data data/processed/val.jsonl")


if __name__ == "__main__":
    main()
