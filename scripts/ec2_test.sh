#!/bin/bash
# EC2 Testing Script for Training Code
# Tests TRL 0.12+ API compatibility after refactoring

set -e  # Exit on first error

echo "=========================================="
echo "EC2 Training Code Test Suite"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((TESTS_PASSED++))
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((TESTS_FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
}

# ==========================================
# Step 1: Check Python and CUDA
# ==========================================
echo ""
echo "Step 1: Environment Check"
echo "------------------------------------------"

python --version || { fail "Python not found"; exit 1; }

python -c "import torch; print(f'PyTorch: {torch.__version__}')" || fail "PyTorch not installed"

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    pass "CUDA available: $GPU_NAME"
else
    warn "CUDA not available - training will fail without GPU"
fi

# ==========================================
# Step 2: Check Library Versions
# ==========================================
echo ""
echo "Step 2: Library Version Check"
echo "------------------------------------------"

check_version() {
    local pkg=$1
    local min_version=$2
    local actual=$(pip show $pkg 2>/dev/null | grep "^Version:" | cut -d' ' -f2)

    if [ -z "$actual" ]; then
        fail "$pkg not installed"
        return 1
    fi

    # Compare versions using Python
    if python -c "from packaging import version; exit(0 if version.parse('$actual') >= version.parse('$min_version') else 1)" 2>/dev/null; then
        pass "$pkg >= $min_version (installed: $actual)"
        return 0
    else
        fail "$pkg >= $min_version required (installed: $actual)"
        return 1
    fi
}

check_version "transformers" "4.47.0"
check_version "trl" "0.12.0"
check_version "peft" "0.13.0"
check_version "bitsandbytes" "0.44.0"
check_version "datasets" "2.14.0"
check_version "accelerate" "0.24.0"

# ==========================================
# Step 3: Import Tests
# ==========================================
echo ""
echo "Step 3: Import Tests"
echo "------------------------------------------"

# Test TRL imports
if python -c "from trl import SFTTrainer, SFTConfig; print('SFTTrainer and SFTConfig imported')" 2>/dev/null; then
    pass "TRL imports (SFTTrainer, SFTConfig)"
else
    fail "TRL imports failed"
fi

# Test PEFT imports
if python -c "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training" 2>/dev/null; then
    pass "PEFT imports"
else
    fail "PEFT imports failed"
fi

# Test our config module
if python -c "from src.training.config import TrainingConfig" 2>/dev/null; then
    pass "src.training.config imports"
else
    fail "src.training.config imports (check PYTHONPATH)"
fi

# Test our trainer module
if python -c "from src.training.trainer import train_model" 2>/dev/null; then
    pass "src.training.trainer imports"
else
    fail "src.training.trainer imports (check PYTHONPATH)"
fi

# ==========================================
# Step 4: API Compatibility Tests
# ==========================================
echo ""
echo "Step 4: API Compatibility Tests"
echo "------------------------------------------"

# Test SFTConfig creation
if python -c "
from src.training.config import TrainingConfig
c = TrainingConfig()
sft = c.to_sft_config()
print(f'SFTConfig created: max_seq_length={sft.max_seq_length}')
" 2>/dev/null; then
    pass "TrainingConfig.to_sft_config() works"
else
    fail "TrainingConfig.to_sft_config() failed"
fi

# Test PEFT prepare_model_for_kbit_training signature
PEFT_CHECK=$(python -c "
from peft import prepare_model_for_kbit_training
import inspect
sig = inspect.signature(prepare_model_for_kbit_training)
params = list(sig.parameters.keys())
if 'use_gradient_checkpointing' in params:
    print('OLD')
else:
    print('NEW')
" 2>/dev/null)

if [ "$PEFT_CHECK" = "NEW" ]; then
    pass "PEFT API is updated (no deprecated params)"
elif [ "$PEFT_CHECK" = "OLD" ]; then
    warn "PEFT has deprecated use_gradient_checkpointing param - may show warnings"
else
    fail "Could not check PEFT API"
fi

# ==========================================
# Step 5: Create Test Data
# ==========================================
echo ""
echo "Step 5: Creating Test Data"
echo "------------------------------------------"

TEST_DATA_DIR="/tmp/ciq_test_data"
mkdir -p "$TEST_DATA_DIR"

cat > "$TEST_DATA_DIR/train.jsonl" << 'EOF'
{"text": "[CLASSIFY] Product: Blue Cotton T-Shirt\nCategory: Apparel > Shirts"}
{"text": "[CLASSIFY] Product: iPhone 15 Pro Case\nCategory: Electronics > Accessories"}
{"text": "[CLASSIFY] Product: Running Shoes Nike\nCategory: Apparel > Footwear"}
{"text": "[CLASSIFY] Product: Organic Coffee Beans\nCategory: Food > Beverages"}
{"text": "[CLASSIFY] Product: Yoga Mat Premium\nCategory: Sports > Fitness"}
EOF

if [ -f "$TEST_DATA_DIR/train.jsonl" ]; then
    pass "Test data created at $TEST_DATA_DIR/train.jsonl"
else
    fail "Failed to create test data"
fi

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"

if [ $TESTS_FAILED -gt 0 ]; then
    echo ""
    echo "Some tests failed. Fix the issues above before running training."
    exit 1
fi

# ==========================================
# Step 6: Optional Training Test
# ==========================================
echo ""
echo "=========================================="
echo "Ready for Training Test"
echo "=========================================="
echo ""
echo "All pre-flight checks passed! Run the following command to test training:"
echo ""
echo "  python -m src.training.trainer \\"
echo "      --train_data $TEST_DATA_DIR/train.jsonl \\"
echo "      --epochs 1 \\"
echo "      --batch_size 1 \\"
echo "      --gradient_accumulation 1 \\"
echo "      --output_dir /tmp/test_output \\"
echo "      --report_to none \\"
echo "      --save_steps 999999"
echo ""
echo "Or run with --run-training flag to execute now:"
echo "  ./scripts/ec2_test.sh --run-training"

# Check if --run-training flag was passed
if [[ "$1" == "--run-training" ]]; then
    echo ""
    echo "=========================================="
    echo "Running Training Test (1 epoch, 5 samples)"
    echo "=========================================="

    python -m src.training.trainer \
        --train_data "$TEST_DATA_DIR/train.jsonl" \
        --epochs 1 \
        --batch_size 1 \
        --gradient_accumulation 1 \
        --output_dir /tmp/test_output \
        --report_to none \
        --save_steps 999999

    if [ $? -eq 0 ]; then
        echo ""
        pass "Training completed successfully!"
    else
        echo ""
        fail "Training failed"
        exit 1
    fi
fi
