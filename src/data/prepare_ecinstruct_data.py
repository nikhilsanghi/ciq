"""
Data Preparation Script - ECInstruct-based Extraction & Q/A

This script properly processes ECInstruct dataset for fine-tuning.

Key differences from prepare_training_data_v2.py:
1. Uses ECInstruct as PRIMARY source (not ESCI)
2. Properly parses ECInstruct's JSON string fields
3. Focuses on Extraction and Q/A tasks (no taxonomy classification)
4. Includes Alpaca for general instruction following

Dataset sources:
- ECInstruct Attribute_Value_Extraction (30K) -> [EXTRACT] task
- ECInstruct Answer_Generation (30K) -> [QA] task
- Alpaca (5K sample) -> [GENERAL] task

Usage:
    python -m src.data.prepare_ecinstruct_data \
        --ecinstruct_path data/raw/ecinstruct.jsonl \
        --alpaca_path data/raw/alpaca.jsonl \
        --output_dir data/processed
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# PROMPT TEMPLATES - Must match inference format from src/api/main.py
# ==============================================================================

EXTRACT_TEMPLATE = """[EXTRACT] Extract all product attributes from the following text as JSON key-value pairs.

Product: {product_text}

Respond with a valid JSON object containing attribute names as keys and their values.
Example: {{"brand": "Nike", "size": "Large", "color": "Blue"}}

Attributes: {json_attributes}"""

QA_TEMPLATE = """[QA] Answer the question about the product based on the provided information.

Product Information:
{product_text}

Question: {question}

Provide a concise and accurate answer based only on the available information.

Answer: {answer}"""

GENERAL_TEMPLATE = """[GENERAL] {instruction}

{input_text}

{output}"""


# ==============================================================================
# Quality Filters
# ==============================================================================

def quality_filter_extraction(text: str) -> bool:
    """Filter extraction examples for quality."""
    if not text or len(text) < 50:
        return False

    if 'Attributes:' not in text:
        return False

    # Check for valid JSON in output
    parts = text.split('Attributes:')
    if len(parts) < 2:
        return False

    json_part = parts[1].strip()
    try:
        if '{' in json_part:
            start = json_part.find('{')
            end = json_part.rfind('}') + 1
            parsed = json.loads(json_part[start:end])
            # Must have at least one attribute
            if not parsed or len(parsed) == 0:
                return False
    except (json.JSONDecodeError, ValueError):
        return False

    return True


def quality_filter_qa(text: str) -> bool:
    """Filter Q&A examples for quality."""
    if not text or len(text) < 50:
        return False

    if 'Answer:' not in text:
        return False

    # Check answer is not empty
    parts = text.split('Answer:')
    if len(parts) < 2:
        return False

    answer = parts[1].strip()
    if len(answer) < 3:
        return False

    return True


def quality_filter_general(text: str) -> bool:
    """Filter general examples for quality."""
    if not text or len(text) < 30:
        return False
    return True


# ==============================================================================
# ECInstruct Processors
# ==============================================================================

def process_attribute_extraction(
    ecinstruct_path: str,
    max_samples: int = 25000
) -> List[Dict[str, str]]:
    """
    Process ECInstruct Attribute_Value_Extraction task.

    ECInstruct format:
    - input: JSON string with product title, description, brand, target attributes
    - output: JSON array of {"attribute", "value", "source"} objects

    We transform this to:
    - Extract all available attributes into a simple {"key": "value"} dict
    """
    examples = []
    skipped_no_attrs = 0
    skipped_parse_error = 0
    skipped_quality = 0

    logger.info(f"Processing ECInstruct Attribute_Value_Extraction from: {ecinstruct_path}")

    try:
        with open(ecinstruct_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(examples) >= max_samples:
                    break

                try:
                    item = json.loads(line.strip())

                    # Filter for extraction task
                    if item.get('task') != 'Attribute_Value_Extraction':
                        continue

                    # Parse input JSON string
                    input_data = json.loads(item['input'])
                    product_title = input_data.get('product title', '')
                    product_desc = input_data.get('product description', '')
                    product_brand = input_data.get('product brand', '')
                    product_price = input_data.get('product price', '')

                    # Build product text
                    product_text = product_title
                    if product_desc:
                        product_text += f"\n\nDescription: {product_desc[:500]}"

                    if not product_text or len(product_text) < 10:
                        continue

                    # Parse output JSON array
                    output_str = item.get('output', '')
                    if not output_str:
                        skipped_no_attrs += 1
                        continue

                    output_array = json.loads(output_str)

                    # Convert to simple dict, skip "None" values
                    attributes_dict = {}
                    for attr_item in output_array:
                        attr_name = attr_item.get('attribute', '').lower()
                        attr_value = attr_item.get('value', '')

                        # Skip None/empty values
                        if attr_value and attr_value.lower() not in ['none', 'n/a', 'unknown', '']:
                            attributes_dict[attr_name] = attr_value

                    # Add brand if available and not already present
                    if product_brand and 'brand' not in attributes_dict:
                        if product_brand.lower() not in ['none', 'n/a', 'unknown', '']:
                            attributes_dict['brand'] = product_brand

                    # Add price if available
                    if product_price and 'price' not in attributes_dict:
                        attributes_dict['price'] = product_price

                    # Need at least one attribute
                    if not attributes_dict:
                        skipped_no_attrs += 1
                        continue

                    # Format as JSON
                    json_attributes = json.dumps(attributes_dict, ensure_ascii=False)

                    # Create formatted example
                    text = EXTRACT_TEMPLATE.format(
                        product_text=product_text,
                        json_attributes=json_attributes
                    )

                    # Quality check
                    if not quality_filter_extraction(text):
                        skipped_quality += 1
                        continue

                    examples.append({'text': text, 'task': 'extract'})

                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    skipped_parse_error += 1
                    continue

    except FileNotFoundError:
        logger.error(f"ECInstruct file not found: {ecinstruct_path}")
        return []

    logger.info(f"Created {len(examples)} extraction examples")
    logger.info(f"  Skipped (no attributes): {skipped_no_attrs}")
    logger.info(f"  Skipped (parse error): {skipped_parse_error}")
    logger.info(f"  Skipped (quality filter): {skipped_quality}")

    return examples


def process_answer_generation(
    ecinstruct_path: str,
    max_samples: int = 25000
) -> List[Dict[str, str]]:
    """
    Process ECInstruct Answer_Generation task.

    ECInstruct format:
    - input: JSON string with question and document (array of review texts)
    - output: Plain text answer

    We transform this to standard Q&A format.
    """
    examples = []
    skipped_empty = 0
    skipped_parse_error = 0
    skipped_quality = 0

    logger.info(f"Processing ECInstruct Answer_Generation from: {ecinstruct_path}")

    try:
        with open(ecinstruct_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(examples) >= max_samples:
                    break

                try:
                    item = json.loads(line.strip())

                    # Filter for Q&A task
                    if item.get('task') != 'Answer_Generation':
                        continue

                    # Parse input JSON string
                    input_data = json.loads(item['input'])
                    question = input_data.get('question', '')
                    documents = input_data.get('document', [])

                    if not question:
                        skipped_empty += 1
                        continue

                    # Combine documents into product information
                    # Use first 3 reviews to limit length
                    if isinstance(documents, list):
                        product_text = '\n\n'.join(documents[:3])
                    else:
                        product_text = str(documents)

                    if not product_text or len(product_text) < 20:
                        skipped_empty += 1
                        continue

                    # Truncate if too long
                    if len(product_text) > 1500:
                        product_text = product_text[:1500] + "..."

                    # Get answer
                    answer = item.get('output', '')
                    if not answer or len(answer) < 3:
                        skipped_empty += 1
                        continue

                    # Create formatted example
                    text = QA_TEMPLATE.format(
                        product_text=product_text,
                        question=question,
                        answer=answer
                    )

                    # Quality check
                    if not quality_filter_qa(text):
                        skipped_quality += 1
                        continue

                    examples.append({'text': text, 'task': 'qa'})

                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    skipped_parse_error += 1
                    continue

    except FileNotFoundError:
        logger.error(f"ECInstruct file not found: {ecinstruct_path}")
        return []

    logger.info(f"Created {len(examples)} Q&A examples")
    logger.info(f"  Skipped (empty): {skipped_empty}")
    logger.info(f"  Skipped (parse error): {skipped_parse_error}")
    logger.info(f"  Skipped (quality filter): {skipped_quality}")

    return examples


def process_alpaca(
    alpaca_path: str,
    num_samples: int = 5000
) -> List[Dict[str, str]]:
    """
    Process Alpaca dataset for general instruction following.

    This prevents catastrophic forgetting during fine-tuning.
    """
    examples = []

    logger.info(f"Processing Alpaca for general instructions from: {alpaca_path}")

    try:
        all_items = []
        with open(alpaca_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    all_items.append(item)
                except json.JSONDecodeError:
                    continue

        # Random sample
        if len(all_items) > num_samples:
            all_items = random.sample(all_items, num_samples)

        for item in all_items:
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output = item.get('output', '')

            if not instruction or not output:
                continue

            # Format using general template
            text = GENERAL_TEMPLATE.format(
                instruction=instruction,
                input_text=input_text if input_text else "",
                output=output
            )

            # Clean up extra newlines
            text = '\n'.join(line for line in text.split('\n') if line.strip() or line == '')

            if quality_filter_general(text):
                examples.append({'text': text, 'task': 'general'})

    except FileNotFoundError:
        logger.error(f"Alpaca file not found: {alpaca_path}")
        return []

    logger.info(f"Created {len(examples)} general instruction examples")
    return examples


# ==============================================================================
# Main Pipeline
# ==============================================================================

def create_splits(
    all_examples: List[Dict[str, str]],
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Create train/validation splits."""
    random.seed(seed)
    random.shuffle(all_examples)

    split_idx = int(len(all_examples) * train_ratio)
    return all_examples[:split_idx], all_examples[split_idx:]


def preview_samples(examples: List[Dict[str, str]], task: str, num: int = 2):
    """Preview samples for verification."""
    task_examples = [e for e in examples if e.get('task') == task]

    logger.info(f"\n{'='*60}")
    logger.info(f"PREVIEW: {task.upper()} ({len(task_examples)} total)")
    logger.info(f"{'='*60}")

    for i, ex in enumerate(task_examples[:num]):
        logger.info(f"\n--- Sample {i+1} ---")
        # Show first 600 chars
        logger.info(ex['text'][:600])
        if len(ex['text']) > 600:
            logger.info("...")

    logger.info(f"\n{'='*60}\n")


def save_jsonl(data: List[Dict[str, str]], filepath: str):
    """Save data as JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(data)} examples to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from ECInstruct (Extraction + Q&A)"
    )
    parser.add_argument(
        "--ecinstruct_path", type=str,
        default="data/raw/ecinstruct.jsonl",
        help="Path to ECInstruct dataset"
    )
    parser.add_argument(
        "--alpaca_path", type=str,
        default="data/raw/alpaca.jsonl",
        help="Path to Alpaca dataset"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="data/processed",
        help="Output directory"
    )
    parser.add_argument(
        "--max_extract", type=int,
        default=25000,
        help="Max extraction samples"
    )
    parser.add_argument(
        "--max_qa", type=int,
        default=25000,
        help="Max Q&A samples"
    )
    parser.add_argument(
        "--general_samples", type=int,
        default=5000,
        help="Number of general instruction samples (Alpaca)"
    )
    parser.add_argument(
        "--seed", type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    logger.info("="*60)
    logger.info("E-Commerce LLM Data Preparation (ECInstruct-based)")
    logger.info("="*60)
    logger.info(f"ECInstruct path: {args.ecinstruct_path}")
    logger.info(f"Alpaca path: {args.alpaca_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Max extraction: {args.max_extract}")
    logger.info(f"Max Q&A: {args.max_qa}")
    logger.info(f"General samples: {args.general_samples}")
    logger.info("="*60)

    all_examples = []

    # Process extraction task
    extract_examples = process_attribute_extraction(
        args.ecinstruct_path,
        args.max_extract
    )
    if extract_examples:
        preview_samples(extract_examples, 'extract')
        all_examples.extend(extract_examples)

    # Process Q&A task
    qa_examples = process_answer_generation(
        args.ecinstruct_path,
        args.max_qa
    )
    if qa_examples:
        preview_samples(qa_examples, 'qa')
        all_examples.extend(qa_examples)

    # Process general instructions
    general_examples = process_alpaca(
        args.alpaca_path,
        args.general_samples
    )
    if general_examples:
        preview_samples(general_examples, 'general', num=1)
        all_examples.extend(general_examples)

    if not all_examples:
        logger.error("No examples created! Check your data paths.")
        return

    # Create splits
    train_data, val_data = create_splits(all_examples, seed=args.seed)

    # Summary
    task_counts = Counter(e['task'] for e in all_examples)

    logger.info("\n" + "="*60)
    logger.info("DATA SUMMARY")
    logger.info("="*60)
    logger.info(f"Extraction [EXTRACT]: {task_counts.get('extract', 0)}")
    logger.info(f"Q&A [QA]: {task_counts.get('qa', 0)}")
    logger.info(f"General [GENERAL]: {task_counts.get('general', 0)}")
    logger.info(f"Total: {len(all_examples)}")
    logger.info(f"Train: {len(train_data)}")
    logger.info(f"Validation: {len(val_data)}")
    logger.info("="*60)

    # Save
    save_jsonl(train_data, str(output_dir / "train.jsonl"))
    save_jsonl(val_data, str(output_dir / "val.jsonl"))

    # Validation checks
    logger.info("\n" + "="*60)
    logger.info("VALIDATION CHECKS")
    logger.info("="*60)

    # Check extraction has JSON
    extract_sample = next((e for e in train_data if e['task'] == 'extract'), None)
    if extract_sample:
        has_json = 'Attributes:' in extract_sample['text'] and '{' in extract_sample['text']
        logger.info(f"Extraction has JSON output: {'PASS' if has_json else 'FAIL'}")

    # Check Q&A has answer
    qa_sample = next((e for e in train_data if e['task'] == 'qa'), None)
    if qa_sample:
        has_answer = 'Answer:' in qa_sample['text']
        logger.info(f"Q&A has answer: {'PASS' if has_answer else 'FAIL'}")

    # Check no empty outputs
    empty_count = sum(1 for e in all_examples if len(e['text']) < 100)
    logger.info(f"Short examples (<100 chars): {empty_count} ({100*empty_count/len(all_examples):.1f}%)")

    logger.info("="*60)
    logger.info("\nData preparation complete!")
    logger.info(f"Output: {output_dir}")
    logger.info("\nNEXT STEPS:")
    logger.info("1. Inspect: head -1 data/processed/train.jsonl | python -m json.tool")
    logger.info("2. Train: python -m src.training.train_v2 --train_data data/processed/train.jsonl")


if __name__ == "__main__":
    main()
