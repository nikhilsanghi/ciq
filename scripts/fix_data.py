#!/usr/bin/env python3
"""
Data Cleaning Script for ECInstruct Dataset

Fixes null value issues that cause "Couldn't cast array of type string to null" errors
when loading with HuggingFace datasets library.

Usage:
    python scripts/fix_data.py
    python scripts/fix_data.py --input ./data/raw/ecinstruct.jsonl --output ./data/raw/ecinstruct_clean.jsonl
"""

import json
import argparse
from pathlib import Path


def clean_ecinstruct(input_path: str, output_path: str) -> dict:
    """
    Clean ECInstruct dataset by keeping only rows with valid string fields.

    Args:
        input_path: Path to input jsonl file
        output_path: Path to output cleaned jsonl file

    Returns:
        Dictionary with cleaning statistics
    """
    print(f"Cleaning {input_path}...")

    clean_rows = []
    skipped_null = 0
    skipped_type = 0
    skipped_empty = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                row = json.loads(line.strip())

                # Get required fields
                instruction = row.get('instruction')
                input_text = row.get('input')
                output = row.get('output')

                # Skip if instruction or output is None/null
                if instruction is None or output is None:
                    skipped_null += 1
                    continue

                # Skip if not strings
                if not isinstance(instruction, str) or not isinstance(output, str):
                    skipped_type += 1
                    continue

                # Skip if instruction or output is empty
                if not instruction.strip() or not output.strip():
                    skipped_empty += 1
                    continue

                # Create clean row with only essential fields (all as strings)
                clean_row = {
                    'instruction': str(instruction),
                    'input': str(input_text) if input_text else '',
                    'output': str(output)
                }
                clean_rows.append(clean_row)

            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping line {line_num} - JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"  Warning: Skipping line {line_num} - Error: {e}")
                continue

    # Write cleaned data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for row in clean_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    stats = {
        'total_input': len(clean_rows) + skipped_null + skipped_type + skipped_empty,
        'kept': len(clean_rows),
        'skipped_null': skipped_null,
        'skipped_type': skipped_type,
        'skipped_empty': skipped_empty,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Clean ECInstruct dataset for training compatibility"
    )
    parser.add_argument(
        "--input", "-i",
        default="./data/raw/ecinstruct.jsonl",
        help="Input jsonl file path"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/raw/ecinstruct_clean.jsonl",
        help="Output cleaned jsonl file path"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ECInstruct Data Cleaner")
    print("=" * 60)

    stats = clean_ecinstruct(args.input, args.output)

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"Total input rows:    {stats['total_input']:,}")
    print(f"Kept (valid):        {stats['kept']:,}")
    print(f"Skipped (null):      {stats['skipped_null']:,}")
    print(f"Skipped (wrong type):{stats['skipped_type']:,}")
    print(f"Skipped (empty):     {stats['skipped_empty']:,}")
    print(f"\nOutput saved to: {args.output}")
    print("\nNext step:")
    print(f"  python -m src.training.trainer --train_data {args.output}")


if __name__ == "__main__":
    main()
