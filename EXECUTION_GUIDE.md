# E-Commerce LLM Execution Guide

## Quick Summary: Running Two Models on One GPU

The `start_multi_model.sh` script already handles this. Both models use `--gpu-memory-utilization 0.45` (45% each), fitting comfortably on a single A10G (24GB).

```bash
# After training, run both models:
./scripts/start_multi_model.sh

# Results:
# - Fine-tuned model: http://localhost:8000
# - Base model: http://localhost:8001
```

---

## Adding More GPUs (Instance Type Change)

**You cannot add GPUs to a running instance.** GPU count is fixed per instance type.

### Options:

| Instance | GPUs | VRAM | Cost/hr | Use Case |
|----------|------|------|---------|----------|
| g5.xlarge | 1x A10G | 24GB | ~$1.00 | Current setup |
| g5.2xlarge | 1x A10G | 24GB | ~$1.21 | More RAM for data loading |
| g5.12xlarge | 4x A10G | 96GB | ~$5.67 | Multiple large models |
| g5.48xlarge | 8x A10G | 192GB | ~$16.29 | Large-scale experiments |

**Recommendation:** Stick with g5.2xlarge. Two 7B models in 4-bit quantization (~5GB each) fit easily on one 24GB GPU.

---

## Step-by-Step Execution

### Phase 1: Launch EC2 Instance

1. **AWS Console → EC2 → Launch Instance**
   - Name: `ciq-training-g5-2xlarge`
   - AMI: Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04)
   - Instance type: `g5.2xlarge`
   - Key pair: Your existing key
   - Storage: 200 GB gp3

2. **Security Group Ports:**
   - SSH (22) - Your IP
   - Custom TCP (8000) - Your IP (vLLM fine-tuned)
   - Custom TCP (8001) - Your IP (vLLM base)
   - Custom TCP (8501) - Your IP (Streamlit)

### Phase 2: Instance Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<EC2-IP>

# Clone repo and run setup
git clone https://github.com/nikhilsanghi/ciq.git ~/ciq
cd ~/ciq
bash scripts/setup_instance.sh
```

The setup script handles:
- System updates
- Git LFS installation
- Conda environment creation
- PyTorch + CUDA installation
- All requirements installation
- Dataset downloads (Google Taxonomy, AmazonQA, MAVE)

### Phase 3: Prepare Training Data

```bash
# Activate environment
conda activate ciq

# Prepare data with general instruction data (prevents catastrophic forgetting)
python -m src.data.prepare_training_data \
    --output_dir data/processed \
    --include_general \
    --max_samples 50000

# Inspect outputs (CRITICAL!)
head -5 data/processed/train.jsonl | python -m json.tool
```

### Phase 4: Test Base Model First

```bash
# Start base model vLLM
nohup python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype half \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85 \
    > logs/vllm.log 2>&1 &

# Wait for startup (~2-3 min)
tail -f logs/vllm.log

# Test with curl
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "prompt": "[CLASSIFY] Classify into Google Product Taxonomy.\n\nProduct: Apple iPhone 15 Pro",
    "max_tokens": 50,
    "temperature": 0
  }'
```

### Phase 5: Train Model

```bash
# Stop vLLM first (frees GPU memory)
pkill -f vllm.entrypoints

# Run training
python -m src.training.train_v2 \
    --train_data data/processed/train.jsonl \
    --eval_data data/processed/val.jsonl \
    --output_dir outputs/ciq-model-v2 \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation 8

# Monitor training
tail -f outputs/ciq-model-v2/runs/*/events.out.tfevents.*
# Or use TensorBoard:
tensorboard --logdir outputs/ciq-model-v2 --port 6006
```

### Phase 6: Merge LoRA Adapters

```bash
# Merge adapters into base model
python -c "
from src.inference.model import merge_lora_weights
merge_lora_weights(
    base_model='mistralai/Mistral-7B-Instruct-v0.3',
    adapter_path='outputs/ciq-model-v2',
    output_path='outputs/ciq-model-merged'
)
"
```

### Phase 7: Deploy Both Models

```bash
# Run multi-model script (both on same GPU)
./scripts/start_multi_model.sh

# Or manually:
# Fine-tuned on port 8000 (45% VRAM)
nohup python -m vllm.entrypoints.openai.api_server \
    --model ./outputs/ciq-model-merged \
    --host 0.0.0.0 --port 8000 \
    --dtype half --max-model-len 2048 \
    --gpu-memory-utilization 0.45 \
    > logs/vllm_finetuned.log 2>&1 &

# Base model on port 8001 (45% VRAM)
nohup python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 --port 8001 \
    --dtype half --max-model-len 2048 \
    --gpu-memory-utilization 0.45 \
    > logs/vllm_base.log 2>&1 &
```

### Phase 8: Compare Models

```bash
# Test classification
echo "=== FINE-TUNED ===" && curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "./outputs/ciq-model-merged", "prompt": "[CLASSIFY] Classify into Google Product Taxonomy.\n\nProduct: Sony WH-1000XM5 Wireless Headphones", "max_tokens": 50, "temperature": 0}' | jq -r '.choices[0].text'

echo "=== BASE ===" && curl -s http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistralai/Mistral-7B-Instruct-v0.3", "prompt": "[CLASSIFY] Classify into Google Product Taxonomy.\n\nProduct: Sony WH-1000XM5 Wireless Headphones", "max_tokens": 50, "temperature": 0}' | jq -r '.choices[0].text'
```

---

## Troubleshooting

### Out of VRAM when running two models
```bash
# Reduce context length and VRAM usage
--max-model-len 1024 --gpu-memory-utilization 0.40
```

### vLLM won't start
```bash
# Check GPU
nvidia-smi

# Check logs
tail -100 logs/vllm_*.log

# Kill stuck processes
pkill -9 -f vllm.entrypoints
```

### Training crashes with OOM
```bash
# Reduce batch size
--batch_size 1 --gradient_accumulation 16
```

### MAVE data not loading
MAVE requires Amazon Review Data 2018. For a quick start, use only AmazonQA:
```bash
python -m src.data.prepare_training_data \
    --amazonqa_dir data/raw/amazonqa \
    --output_dir data/processed \
    --max_samples 30000
```

---

## File Reference

| File | Purpose |
|------|---------|
| `scripts/setup_instance.sh` | Fresh EC2 setup |
| `scripts/start_multi_model.sh` | Run both models |
| `src/data/prepare_training_data.py` | Data prep |
| `src/training/train_v2.py` | Training with validation |
| `src/inference/model.py` | Model loading & merging |
| `app/demo.py` | Streamlit demo |
