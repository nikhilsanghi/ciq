#!/bin/bash
# Download datasets for E-Commerce LLM training
#
# Downloads:
# 1. Amazon ESCI (HuggingFace) - Has actual product text
# 2. ECInstruct (HuggingFace) - Pre-formatted e-commerce tasks
# 3. Alpaca (HuggingFace) - General instructions
# 4. Google Taxonomy (Direct) - Category labels
#
# Usage:
#   ./scripts/download_datasets.sh
#
# Requirements:
#   pip install datasets

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_success() { echo -e "${GREEN}✓ $1${NC}"; }
log_error() { echo -e "${RED}✗ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }

echo "=========================================="
echo "E-Commerce LLM - Dataset Download"
echo "=========================================="

# Create directories
mkdir -p data/raw data/processed

# Check if datasets library is installed
if ! python -c "import datasets" 2>/dev/null; then
    log_error "datasets library not installed. Run: pip install datasets"
    exit 1
fi

# 1. Download Amazon ESCI
echo ""
echo "1. Downloading Amazon ESCI..."
echo "   (This has actual product titles, descriptions, categories)"
python -c "
from datasets import load_dataset
import sys

try:
    print('  Loading tasksource/esci (us)...')
    d = load_dataset('tasksource/esci', 'us', split='train')
    d.to_json('data/raw/esci.jsonl')
    print(f'  Downloaded {len(d)} products')
except Exception as e:
    print(f'  Error: {e}')
    sys.exit(1)
"
if [ $? -eq 0 ]; then
    log_success "ESCI downloaded: $(wc -l < data/raw/esci.jsonl) records"
else
    log_error "ESCI download failed"
fi

# 2. Download ECInstruct
echo ""
echo "2. Downloading ECInstruct..."
echo "   (Pre-formatted e-commerce tasks including Q&A)"
python -c "
from datasets import load_dataset
import sys

try:
    print('  Loading NingLab/ECInstruct...')
    d = load_dataset('NingLab/ECInstruct', split='train')
    d.to_json('data/raw/ecinstruct.jsonl')
    print(f'  Downloaded {len(d)} examples')
except Exception as e:
    print(f'  Error: {e}')
    sys.exit(1)
"
if [ $? -eq 0 ]; then
    log_success "ECInstruct downloaded: $(wc -l < data/raw/ecinstruct.jsonl) records"
else
    log_error "ECInstruct download failed"
fi

# 3. Download Alpaca
echo ""
echo "3. Downloading Alpaca..."
echo "   (General instructions - prevents catastrophic forgetting)"
python -c "
from datasets import load_dataset
import sys

try:
    print('  Loading tatsu-lab/alpaca...')
    d = load_dataset('tatsu-lab/alpaca', split='train')
    d.to_json('data/raw/alpaca.jsonl')
    print(f'  Downloaded {len(d)} examples')
except Exception as e:
    print(f'  Error: {e}')
    sys.exit(1)
"
if [ $? -eq 0 ]; then
    log_success "Alpaca downloaded: $(wc -l < data/raw/alpaca.jsonl) records"
else
    log_error "Alpaca download failed"
fi

# 4. Download Google Taxonomy
echo ""
echo "4. Downloading Google Product Taxonomy..."
mkdir -p data/raw/taxonomy
wget -q https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt \
    -O data/raw/taxonomy/google_taxonomy.txt
if [ $? -eq 0 ]; then
    log_success "Google Taxonomy downloaded: $(wc -l < data/raw/taxonomy/google_taxonomy.txt) categories"
else
    log_error "Google Taxonomy download failed"
fi

# Summary
echo ""
echo "=========================================="
echo "Download Summary"
echo "=========================================="
echo ""
echo "Files in data/raw/:"
ls -lh data/raw/*.jsonl 2>/dev/null || echo "  (no jsonl files)"
ls -lh data/raw/taxonomy/*.txt 2>/dev/null || echo "  (no taxonomy file)"

echo ""
echo "Record counts:"
echo "  ESCI:       $(wc -l < data/raw/esci.jsonl 2>/dev/null || echo 'N/A')"
echo "  ECInstruct: $(wc -l < data/raw/ecinstruct.jsonl 2>/dev/null || echo 'N/A')"
echo "  Alpaca:     $(wc -l < data/raw/alpaca.jsonl 2>/dev/null || echo 'N/A')"

# Inspect samples
echo ""
echo "=========================================="
echo "Sample Inspection (CRITICAL!)"
echo "=========================================="

echo ""
echo "=== ESCI Sample ==="
head -1 data/raw/esci.jsonl | python -m json.tool 2>/dev/null | head -20

echo ""
echo "=== ECInstruct Sample ==="
head -1 data/raw/ecinstruct.jsonl | python -m json.tool 2>/dev/null | head -20

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Prepare training data:"
echo "   python -m src.data.prepare_training_data_v2 --output_dir data/processed"
echo ""
echo "2. Validate format:"
echo "   head -1 data/processed/train.jsonl | python -m json.tool"
echo ""
echo "3. Check task distribution:"
echo "   grep -c '\[CLASSIFY\]' data/processed/train.jsonl"
echo "   grep -c '\[EXTRACT\]' data/processed/train.jsonl"
echo "   grep -c '\[QA\]' data/processed/train.jsonl"
echo ""
