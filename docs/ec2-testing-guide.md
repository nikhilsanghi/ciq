# EC2 Testing Guide for Training Code

**Status**: Refactoring COMPLETE. Ready for EC2 testing.

---

## Quick Start

### 1. Push & Pull

```bash
# Local machine (if changes not yet pushed)
git add -A && git commit -m "Refactor for TRL 0.12+ API" && git push

# On EC2
cd ~/ciq && git pull
```

### 2. Install Dependencies

```bash
pip install transformers>=4.47 trl>=0.12 peft>=0.13 bitsandbytes>=0.44 datasets accelerate packaging
```

### 3. Run Automated Tests

```bash
cd ~/ciq
./scripts/ec2_test.sh
```

### 4. Run Training Test (Optional)

```bash
./scripts/ec2_test.sh --run-training
```

---

## Manual Validation Commands

If you prefer to run tests manually:

### Test 1: Verify TRL Imports

```bash
python -c "from trl import SFTTrainer, SFTConfig; print('✓ TRL imports OK')"
```

**Expected**: `✓ TRL imports OK`

### Test 2: Verify Config Creates SFTConfig

```bash
python -c "
from src.training.config import TrainingConfig
c = TrainingConfig()
sft = c.to_sft_config()
print('✓ SFTConfig creation OK')
print(f'  max_seq_length: {sft.max_seq_length}')
"
```

**Expected**:
```
✓ SFTConfig creation OK
  max_seq_length: 2048
```

### Test 3: Verify PEFT Function Signature

```bash
python -c "
from peft import prepare_model_for_kbit_training
import inspect
sig = inspect.signature(prepare_model_for_kbit_training)
params = list(sig.parameters.keys())
if 'use_gradient_checkpointing' in params:
    print('⚠ Old PEFT version - has deprecated param')
else:
    print('✓ PEFT is updated - no deprecated params')
"
```

**Expected**: `✓ PEFT is updated - no deprecated params`

### Test 4: Check Library Versions

```bash
pip list | grep -E "trl|peft|transformers|bitsandbytes"
```

**Expected**:
```
bitsandbytes      0.44.x or higher
peft              0.13.x or higher
transformers      4.47.x or higher
trl               0.12.x or higher
```

---

## Minimal Training Test

### Create Test Data

```bash
cat > /tmp/test_data.jsonl << 'EOF'
{"text": "[CLASSIFY] Product: Blue Cotton T-Shirt\nCategory: Apparel > Shirts"}
{"text": "[CLASSIFY] Product: iPhone 15 Pro Case\nCategory: Electronics > Accessories"}
{"text": "[CLASSIFY] Product: Running Shoes Nike\nCategory: Apparel > Footwear"}
{"text": "[CLASSIFY] Product: Organic Coffee Beans\nCategory: Food > Beverages"}
{"text": "[CLASSIFY] Product: Yoga Mat Premium\nCategory: Sports > Fitness"}
EOF
```

### Run Training (5 samples, 1 epoch)

```bash
python -m src.training.trainer \
    --train_data /tmp/test_data.jsonl \
    --epochs 1 \
    --batch_size 1 \
    --gradient_accumulation 1 \
    --output_dir /tmp/test_output \
    --report_to none \
    --save_steps 999999
```

---

## What Success Looks Like

1. **No deprecation warnings** about:
   - `use_gradient_checkpointing`
   - `enable_input_require_grads`
   - `evaluation_strategy`

2. **Training starts** and shows:
   ```
   Step 6/6: Initializing SFTTrainer...
   Starting training...
   ```

3. **Training completes** without errors:
   ```
   Training complete! Saving model...
   Model saved to: /tmp/test_output
   ```

---

## Troubleshooting

### Wrong Library Versions

```bash
# Check versions
pip list | grep -E "trl|peft|transformers|bitsandbytes"

# Upgrade if needed
pip install --upgrade transformers>=4.47 trl>=0.12 peft>=0.13 bitsandbytes>=0.44
```

### CUDA/GPU Issues

```bash
# Check GPU availability
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Import Errors

```bash
# Ensure you're in the project root
cd ~/ciq

# Add to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Common Error Messages

| Error | Cause | Fix |
|-------|-------|-----|
| `ImportError: cannot import SFTConfig` | TRL too old | `pip install trl>=0.12` |
| `TypeError: unexpected keyword argument 'use_gradient_checkpointing'` | PEFT too old | `pip install peft>=0.13` |
| `CUDA out of memory` | GPU VRAM insufficient | Reduce `--batch_size` to 1 |
| `ModuleNotFoundError: No module named 'src'` | PYTHONPATH issue | Run from project root or set PYTHONPATH |

---

## Full Training Commands

### Minimal Test (8GB VRAM)

```bash
python -m src.training.trainer \
    --preset minimal \
    --train_data /path/to/train.jsonl \
    --output_dir ./outputs/minimal-test \
    --report_to none
```

### Balanced Config (12-16GB VRAM)

```bash
python -m src.training.trainer \
    --preset balanced \
    --train_data /path/to/train.jsonl \
    --output_dir ./outputs/balanced \
    --wandb_project ecommerce-llm
```

### High Quality (24GB+ VRAM)

```bash
python -m src.training.trainer \
    --preset high_quality \
    --train_data /path/to/train.jsonl \
    --eval_data /path/to/eval.jsonl \
    --output_dir ./outputs/high-quality \
    --wandb_project ecommerce-llm
```

---

## Key Changes in TRL 0.12+ Refactoring

The following changes were made to support TRL 0.12+:

### 1. SFTConfig Instead of TrainingArguments

```python
# Old API (TRL < 0.12)
trainer = SFTTrainer(
    model=model,
    args=training_args,  # TrainingArguments
    max_seq_length=2048,
    dataset_text_field="text",
    ...
)

# New API (TRL 0.12+)
trainer = SFTTrainer(
    model=model,
    args=sft_config,  # SFTConfig (includes max_seq_length, etc.)
    ...
)
```

### 2. processing_class Instead of tokenizer

```python
# Old API
trainer = SFTTrainer(..., tokenizer=tokenizer)

# New API
trainer = SFTTrainer(..., processing_class=tokenizer)
```

### 3. Removed Deprecated PEFT Parameters

```python
# Old API
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True  # Deprecated in peft 0.13+
)

# New API
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()  # Called separately
```

### 4. eval_strategy Instead of evaluation_strategy

```python
# Old parameter name
evaluation_strategy="steps"

# New parameter name (TRL 0.12+)
eval_strategy="steps"
```
