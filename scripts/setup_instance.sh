#!/bin/bash
# Fresh Instance Setup Script for E-Commerce LLM Project
# Run this on a new g5.2xlarge EC2 instance with Deep Learning AMI
#
# Usage: bash scripts/setup_instance.sh

set -e  # Exit on any error

echo "=========================================="
echo "E-Commerce LLM - Fresh Instance Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_success() { echo -e "${GREEN}✓ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
log_error() { echo -e "${RED}✗ $1${NC}"; }

# Step 1: System updates
echo ""
echo "Step 1: System Updates"
echo "----------------------"
sudo apt update && sudo apt upgrade -y
log_success "System updated"

# Step 2: Install git-lfs (needed for MAVE dataset)
echo ""
echo "Step 2: Installing git-lfs"
echo "--------------------------"
sudo apt install git-lfs -y
git lfs install
log_success "git-lfs installed"

# Step 3: Create directory structure
echo ""
echo "Step 3: Creating Directory Structure"
echo "------------------------------------"
mkdir -p ~/ciq/{data/raw,data/processed,outputs,logs}
cd ~/ciq
log_success "Directories created"

# Step 4: Clone or update repository
echo ""
echo "Step 4: Setting Up Repository"
echo "-----------------------------"
if [ -d ".git" ]; then
    log_warning "Repository exists, pulling latest..."
    git pull origin main
else
    log_warning "Cloning repository..."
    git clone https://github.com/nikhilsanghi/ciq.git .
fi
log_success "Repository ready"

# Step 5: Setup Python environment
echo ""
echo "Step 5: Setting Up Python Environment"
echo "--------------------------------------"
if conda info --envs | grep -q "ciq"; then
    log_warning "Conda env 'ciq' exists, activating..."
    source ~/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ciq
else
    log_warning "Creating conda env 'ciq'..."
    conda create -n ciq python=3.10 -y
    source ~/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ciq
fi
log_success "Python environment ready"

# Step 6: Install PyTorch with CUDA
echo ""
echo "Step 6: Installing PyTorch"
echo "--------------------------"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
log_success "PyTorch installed"

# Step 7: Install requirements
echo ""
echo "Step 7: Installing Requirements"
echo "-------------------------------"
pip install -r requirements.txt
log_success "Requirements installed"

# Step 7b: Install Flash Attention (2-3x faster training)
echo ""
echo "Step 7b: Installing Flash Attention"
echo "------------------------------------"
echo "This may take 5-10 minutes to compile..."
pip install flash-attn --no-build-isolation || log_warning "Flash Attention install failed (optional, training will still work)"
python -c "import flash_attn; print('Flash Attention: OK')" && log_success "Flash Attention installed" || log_warning "Flash Attention not available"

# Step 8: Verify installation
echo ""
echo "Step 8: Verifying Installation"
echo "------------------------------"

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Check key packages
python -c "import vllm; print('vLLM: OK')" || log_error "vLLM import failed"
python -c "from transformers import AutoModelForCausalLM; print('Transformers: OK')" || log_error "Transformers import failed"
python -c "from peft import LoraConfig; print('PEFT: OK')" || log_error "PEFT import failed"
python -c "from trl import SFTTrainer; print('TRL: OK')" || log_error "TRL import failed"
python -c "import streamlit; print('Streamlit: OK')" || log_error "Streamlit import failed"
python -c "import tensorboard; print('TensorBoard: OK')" || log_error "TensorBoard import failed"

log_success "All packages verified"

# Step 9: Download datasets (ESCI + ECInstruct + Alpaca)
# NOTE: These datasets are self-contained and don't require external dependencies
# Unlike MAVE (needs 21GB Amazon data) and AmazonQA (lacks product descriptions)
echo ""
echo "Step 9: Downloading Datasets"
echo "----------------------------"
mkdir -p data/raw/taxonomy

# Google Taxonomy
echo "Downloading Google Product Taxonomy..."
wget -q https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt \
    -O data/raw/taxonomy/google_taxonomy.txt
log_success "Google Taxonomy downloaded"

# Amazon ESCI (has actual product titles, descriptions, categories)
echo "Downloading Amazon ESCI (130K products with full text)..."
python -c "
from datasets import load_dataset
print('  Loading tasksource/esci (us)...')
d = load_dataset('tasksource/esci', 'us', split='train')
d.to_json('data/raw/esci.jsonl')
print(f'  Downloaded {len(d)} products')
"
log_success "Amazon ESCI downloaded"

# ECInstruct (pre-formatted e-commerce tasks including Q&A)
echo "Downloading ECInstruct (116K e-commerce task examples)..."
python -c "
from datasets import load_dataset
print('  Loading NingLab/ECInstruct...')
d = load_dataset('NingLab/ECInstruct', split='train')
d.to_json('data/raw/ecinstruct.jsonl')
print(f'  Downloaded {len(d)} examples')
"
log_success "ECInstruct downloaded"

# Alpaca (general instructions to prevent catastrophic forgetting)
echo "Downloading Alpaca (52K general instructions)..."
python -c "
from datasets import load_dataset
print('  Loading tatsu-lab/alpaca...')
d = load_dataset('tatsu-lab/alpaca', split='train')
d.to_json('data/raw/alpaca.jsonl')
print(f'  Downloaded {len(d)} examples')
"
log_success "Alpaca downloaded"

# Step 10: Verify datasets
echo ""
echo "Step 10: Verifying Datasets"
echo "---------------------------"
echo "Google Taxonomy categories: $(wc -l < data/raw/taxonomy/google_taxonomy.txt)"
echo "ESCI products: $(wc -l < data/raw/esci.jsonl)"
echo "ECInstruct examples: $(wc -l < data/raw/ecinstruct.jsonl)"
echo "Alpaca examples: $(wc -l < data/raw/alpaca.jsonl)"

echo ""
echo "=== Sample ESCI Entry (has actual product text!) ==="
head -1 data/raw/esci.jsonl | python -m json.tool 2>/dev/null | head -15

echo ""
echo "=== Sample ECInstruct Entry ==="
head -1 data/raw/ecinstruct.jsonl | python -m json.tool 2>/dev/null | head -15

# Final summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Environment:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA: {torch.version.cuda}')"
python -c "import flash_attn; print('  Flash Attention: Available')" 2>/dev/null || echo "  Flash Attention: Not available"
echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Prepare training data (v2 - uses ESCI + ECInstruct):"
echo "   python -m src.data.prepare_training_data_v2 --output_dir data/processed"
echo ""
echo "2. Validate data format:"
echo "   head -1 data/processed/train.jsonl | python -m json.tool"
echo "   echo 'Classification:' && grep -c '\\[CLASSIFY\\]' data/processed/train.jsonl"
echo "   echo 'Extraction:' && grep -c '\\[EXTRACT\\]' data/processed/train.jsonl"
echo "   echo 'Q&A:' && grep -c '\\[QA\\]' data/processed/train.jsonl"
echo ""
echo "3. Start vLLM for base model testing:"
echo "   nohup python -m vllm.entrypoints.openai.api_server \\"
echo "       --model mistralai/Mistral-7B-Instruct-v0.3 \\"
echo "       --host 0.0.0.0 --port 8000 --dtype half \\"
echo "       --max-model-len 2048 --gpu-memory-utilization 0.85 \\"
echo "       > logs/vllm.log 2>&1 &"
echo ""
echo "4. Start Streamlit demo:"
echo "   nohup streamlit run app/demo.py --server.port 8501 --server.address 0.0.0.0 > logs/streamlit.log 2>&1 &"
echo ""
echo "5. Train model (OPTIMIZED - uses Flash Attention + larger batch):"
echo "   python -m src.training.train_v2 \\"
echo "       --train_data data/processed/train.jsonl \\"
echo "       --eval_data data/processed/val.jsonl \\"
echo "       --output_dir outputs/ciq-model-v2 \\"
echo "       --epochs 3 \\"
echo "       --batch_size 8 \\"
echo "       --gradient_accumulation 4 \\"
echo "       --max_seq_length 1024"
echo ""
echo "   Expected: ~40-50 min for 45K examples (vs 3 hours with default settings)"
echo ""
echo "6. After training, compare models:"
echo "   ./scripts/start_multi_model.sh"
echo "   ./scripts/start_compare_demo.sh"
echo ""
