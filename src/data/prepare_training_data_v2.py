"""
Data Preparation Script v2 - Using ESCI + ECInstruct

Key changes from v1:
1. Uses Amazon ESCI (has actual product text) instead of MAVE
2. Uses ECInstruct (pre-formatted) instead of AmazonQA
3. Training format EXACTLY matches inference prompts from src/api/main.py
4. Includes Alpaca data by DEFAULT (not optional)
5. Quality filtering is mandatory

CRITICAL LESSON: Training format must match inference format exactly!
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# PROMPT TEMPLATES - Must match src/api/main.py EXACTLY!
# ==============================================================================

CLASSIFY_TEMPLATE = """[CLASSIFY] Classify the following product into the most specific Google Product Taxonomy category.

Product: {product_text}

Respond with only the category path, e.g., "Electronics > Computers > Laptops"

Category: {category}"""

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

def quality_filter(example: Dict[str, Any], task: str) -> bool:
    """
    Filter out low-quality examples.

    Returns True if example passes quality checks.
    """
    text = example.get('text', '')

    # Must have content
    if not text or len(text) < 50:
        return False

    # Task-specific checks
    if task == 'classify':
        # Must have category in output
        if 'Category:' not in text:
            return False
        # Category must have at least one level
        parts = text.split('Category:')
        if len(parts) < 2 or '>' not in parts[1] and len(parts[1].strip()) < 5:
            return False

    elif task == 'extract':
        # Must have valid JSON in output
        if 'Attributes:' not in text:
            return False
        parts = text.split('Attributes:')
        if len(parts) < 2:
            return False
        json_part = parts[1].strip()
        try:
            # Find JSON object
            if '{' in json_part:
                start = json_part.find('{')
                end = json_part.rfind('}') + 1
                json.loads(json_part[start:end])
        except (json.JSONDecodeError, ValueError):
            return False

    elif task == 'qa':
        # Must have answer
        if 'Answer:' not in text:
            return False
        parts = text.split('Answer:')
        if len(parts) < 2 or len(parts[1].strip()) < 3:
            return False

    return True


# ==============================================================================
# Dataset Processors
# ==============================================================================

def process_esci_for_classification(esci_path: str, max_samples: int = 15000) -> List[Dict[str, str]]:
    """
    Process Amazon ESCI for classification task.

    ESCI has real product titles, descriptions, and categories.
    """
    examples = []
    category_counts = Counter()

    logger.info(f"Processing ESCI for classification from: {esci_path}")

    try:
        with open(esci_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(examples) >= max_samples:
                    break

                try:
                    item = json.loads(line.strip())

                    # Get product text
                    title = item.get('product_title', '')
                    if not title or len(title) < 10:
                        continue

                    # Get category
                    category = item.get('product_category', '')
                    if not category:
                        continue

                    # Clean category - some have extra whitespace
                    category = ' > '.join([c.strip() for c in category.split('>')])

                    # Skip if we have too many of this category (balance dataset)
                    top_level = category.split('>')[0].strip()
                    if category_counts[top_level] >= max_samples // 20:  # Max 5% per top category
                        continue
                    category_counts[top_level] += 1

                    # Add description if available
                    description = item.get('product_description', '')
                    if description and len(description) > 50:
                        product_text = f"{title}\n\nDescription: {description[:500]}"
                    else:
                        product_text = title

                    # Format using exact template
                    text = CLASSIFY_TEMPLATE.format(
                        product_text=product_text,
                        category=category
                    )

                    example = {'text': text, 'task': 'classify'}
                    if quality_filter(example, 'classify'):
                        examples.append(example)

                except (json.JSONDecodeError, KeyError):
                    continue

    except FileNotFoundError:
        logger.error(f"ESCI file not found: {esci_path}")
        return []

    logger.info(f"Created {len(examples)} classification examples from ESCI")
    return examples


def process_esci_for_extraction(esci_path: str, max_samples: int = 15000) -> List[Dict[str, str]]:
    """
    Process Amazon ESCI for attribute extraction task.

    Uses product_brand, product_color, etc. fields as ground truth.
    """
    examples = []

    logger.info(f"Processing ESCI for extraction from: {esci_path}")

    # Attribute field mapping
    attribute_fields = {
        'product_brand': 'brand',
        'product_color': 'color',
        'product_size': 'size',
        'product_material': 'material',
        'product_style': 'style',
    }

    try:
        with open(esci_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(examples) >= max_samples:
                    break

                try:
                    item = json.loads(line.strip())

                    # Get product text
                    title = item.get('product_title', '')
                    if not title or len(title) < 10:
                        continue

                    # Extract available attributes
                    attributes = {}
                    for field, attr_name in attribute_fields.items():
                        value = item.get(field, '')
                        if value and value.lower() not in ['n/a', 'unknown', 'none', '']:
                            attributes[attr_name] = str(value).strip()

                    # Also try to extract from title (simple patterns)
                    if 'brand' not in attributes:
                        # First word might be brand
                        words = title.split()
                        if words and words[0][0].isupper():
                            attributes['brand'] = words[0]

                    # Need at least 2 attributes
                    if len(attributes) < 2:
                        continue

                    # Add description for context
                    description = item.get('product_description', '')
                    if description and len(description) > 50:
                        product_text = f"{title}\n\nDescription: {description[:300]}"
                    else:
                        product_text = title

                    # Format JSON attributes
                    json_attributes = json.dumps(attributes, ensure_ascii=False)

                    # Format using exact template
                    text = EXTRACT_TEMPLATE.format(
                        product_text=product_text,
                        json_attributes=json_attributes
                    )

                    example = {'text': text, 'task': 'extract'}
                    if quality_filter(example, 'extract'):
                        examples.append(example)

                except (json.JSONDecodeError, KeyError):
                    continue

    except FileNotFoundError:
        logger.error(f"ESCI file not found: {esci_path}")
        return []

    logger.info(f"Created {len(examples)} extraction examples from ESCI")
    return examples


def process_ecinstruct_for_qa(ecinstruct_path: str, max_samples: int = 10000) -> List[Dict[str, str]]:
    """
    Process ECInstruct for Q&A task.

    Filters for product_qa task type only (skips ranking/matching tasks).
    """
    examples = []
    task_counts = Counter()

    logger.info(f"Processing ECInstruct for Q&A from: {ecinstruct_path}")

    # Task types to include
    qa_task_types = {'product_qa', 'qa', 'question_answering', 'answer_generation'}

    try:
        with open(ecinstruct_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(examples) >= max_samples:
                    break

                try:
                    item = json.loads(line.strip())

                    # Get task type
                    task_type = item.get('task_type', item.get('task', '')).lower()
                    task_counts[task_type] += 1

                    # Only process Q&A tasks
                    if not any(qt in task_type for qt in qa_task_types):
                        continue

                    # Get instruction/input/output
                    instruction = item.get('instruction', '')
                    input_text = item.get('input', '')
                    output = item.get('output', '')

                    if not instruction or not output:
                        continue

                    # Try to extract product text and question
                    # ECInstruct format varies, so we handle multiple patterns
                    product_text = ""
                    question = ""

                    if 'Product:' in input_text or 'product:' in input_text.lower():
                        parts = input_text.split('Question:')
                        if len(parts) == 2:
                            product_text = parts[0].replace('Product:', '').strip()
                            question = parts[1].strip()
                        else:
                            product_text = input_text
                            question = instruction
                    else:
                        product_text = input_text if input_text else instruction
                        question = instruction if input_text else "What can you tell me about this product?"

                    if not product_text or not question:
                        continue

                    # Format using exact template
                    text = QA_TEMPLATE.format(
                        product_text=product_text[:500],
                        question=question,
                        answer=output
                    )

                    example = {'text': text, 'task': 'qa'}
                    if quality_filter(example, 'qa'):
                        examples.append(example)

                except (json.JSONDecodeError, KeyError):
                    continue

    except FileNotFoundError:
        logger.error(f"ECInstruct file not found: {ecinstruct_path}")
        return []

    logger.info(f"Task type distribution in ECInstruct: {dict(task_counts.most_common(10))}")
    logger.info(f"Created {len(examples)} Q&A examples from ECInstruct")
    return examples


def process_alpaca_for_general(alpaca_path: str, num_samples: int = 5000) -> List[Dict[str, str]]:
    """
    Process Alpaca dataset for general instruction data.

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

            examples.append({'text': text, 'task': 'general'})

    except FileNotFoundError:
        logger.error(f"Alpaca file not found: {alpaca_path}")
        return []

    logger.info(f"Created {len(examples)} general instruction examples from Alpaca")
    return examples


# ==============================================================================
# Main Processing Pipeline
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
        # Show first 500 chars
        logger.info(ex['text'][:500])
        if len(ex['text']) > 500:
            logger.info("...")

    logger.info(f"\n{'='*60}\n")


def save_jsonl(data: List[Dict[str, str]], filepath: str):
    """Save data as JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(data)} examples to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data v2 (ESCI + ECInstruct)")
    parser.add_argument("--esci_path", type=str, default="data/raw/esci.jsonl",
                       help="Path to ESCI dataset")
    parser.add_argument("--ecinstruct_path", type=str, default="data/raw/ecinstruct.jsonl",
                       help="Path to ECInstruct dataset")
    parser.add_argument("--alpaca_path", type=str, default="data/raw/alpaca.jsonl",
                       help="Path to Alpaca dataset")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                       help="Output directory")
    parser.add_argument("--max_classify", type=int, default=15000,
                       help="Max classification samples")
    parser.add_argument("--max_extract", type=int, default=15000,
                       help="Max extraction samples")
    parser.add_argument("--max_qa", type=int, default=10000,
                       help="Max Q&A samples")
    parser.add_argument("--general_samples", type=int, default=5000,
                       help="Number of general instruction samples (Alpaca)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    logger.info("="*60)
    logger.info("E-Commerce LLM Data Preparation v2")
    logger.info("="*60)
    logger.info(f"ESCI path: {args.esci_path}")
    logger.info(f"ECInstruct path: {args.ecinstruct_path}")
    logger.info(f"Alpaca path: {args.alpaca_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("="*60)

    all_examples = []

    # Process each dataset
    classify_examples = process_esci_for_classification(args.esci_path, args.max_classify)
    if classify_examples:
        preview_samples(classify_examples, 'classify')
        all_examples.extend(classify_examples)

    extract_examples = process_esci_for_extraction(args.esci_path, args.max_extract)
    if extract_examples:
        preview_samples(extract_examples, 'extract')
        all_examples.extend(extract_examples)

    qa_examples = process_ecinstruct_for_qa(args.ecinstruct_path, args.max_qa)
    if qa_examples:
        preview_samples(qa_examples, 'qa')
        all_examples.extend(qa_examples)

    # Always include general instruction data
    general_examples = process_alpaca_for_general(args.alpaca_path, args.general_samples)
    if general_examples:
        preview_samples(general_examples, 'general', num=1)
        all_examples.extend(general_examples)

    if not all_examples:
        logger.error("No examples created! Check your data paths.")
        logger.error("Download datasets first:")
        logger.error("  python -c \"from datasets import load_dataset; load_dataset('tasksource/esci', 'us').to_json('data/raw/esci.jsonl')\"")
        return

    # Create splits
    train_data, val_data = create_splits(all_examples, seed=args.seed)

    # Summary
    task_counts = Counter(e['task'] for e in all_examples)

    logger.info("\n" + "="*60)
    logger.info("DATA SUMMARY")
    logger.info("="*60)
    logger.info(f"Classification: {task_counts.get('classify', 0)}")
    logger.info(f"Extraction: {task_counts.get('extract', 0)}")
    logger.info(f"Q&A: {task_counts.get('qa', 0)}")
    logger.info(f"General: {task_counts.get('general', 0)}")
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

    # Check format contains terminators
    sample = train_data[0]['text']
    has_terminator = any(t in sample for t in ['Category:', 'Attributes:', 'Answer:'])
    logger.info(f"Format has terminators: {'PASS' if has_terminator else 'FAIL'}")

    # Check no empty outputs
    empty_count = sum(1 for e in all_examples if len(e['text']) < 100)
    logger.info(f"Short examples (<100 chars): {empty_count} ({100*empty_count/len(all_examples):.1f}%)")

    logger.info("="*60)
    logger.info("\nData preparation complete!")
    logger.info(f"Output: {output_dir}")
    logger.info("\nNEXT STEPS:")
    logger.info("1. Inspect: head -1 data/processed/train.jsonl | python -m json.tool")
    logger.info("2. Check format: grep 'Category:' data/processed/train.jsonl | head -1")
    logger.info("3. Train: python -m src.training.train_v2 --train_data data/processed/train.jsonl")


if __name__ == "__main__":
    main()
