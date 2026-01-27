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

# Step 9: Download datasets
echo ""
echo "Step 9: Downloading Datasets"
echo "----------------------------"
mkdir -p data/raw/{mave,amazonqa,taxonomy}

# Google Taxonomy
echo "Downloading Google Product Taxonomy..."
wget -q https://www.google.com/basepages/producttype/taxonomy.en-US.txt \
    -O data/raw/taxonomy/google_taxonomy.txt
log_success "Google Taxonomy downloaded"

# AmazonQA
echo "Downloading AmazonQA train set..."
wget -q https://amazon-qa.s3-us-west-2.amazonaws.com/train-qar.jsonl \
    -O data/raw/amazonqa/train.jsonl
echo "Downloading AmazonQA validation set..."
wget -q https://amazon-qa.s3-us-west-2.amazonaws.com/val-qar.jsonl \
    -O data/raw/amazonqa/val.jsonl
log_success "AmazonQA downloaded"

# MAVE (just clone, actual data needs Amazon Review Data 2018)
echo "Cloning MAVE repository..."
if [ ! -d "data/raw/mave/.git" ]; then
    git clone https://github.com/google-research-datasets/MAVE.git data/raw/mave/
fi
log_success "MAVE cloned (note: requires Amazon Review Data 2018 for full data)"

# Step 10: Verify datasets
echo ""
echo "Step 10: Verifying Datasets"
echo "---------------------------"
echo "Google Taxonomy categories: $(wc -l < data/raw/taxonomy/google_taxonomy.txt)"
echo "AmazonQA train samples: $(wc -l < data/raw/amazonqa/train.jsonl)"
echo "AmazonQA val samples: $(wc -l < data/raw/amazonqa/val.jsonl)"

echo ""
echo "Sample AmazonQA entry:"
head -1 data/raw/amazonqa/train.jsonl | python -m json.tool 2>/dev/null || head -1 data/raw/amazonqa/train.jsonl

# Final summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Prepare training data:"
echo "   python -m src.data.prepare_training_data --output_dir data/processed --include_general"
echo ""
echo "2. Start vLLM for base model testing:"
echo "   nohup python -m vllm.entrypoints.openai.api_server \\"
echo "       --model mistralai/Mistral-7B-Instruct-v0.3 \\"
echo "       --host 0.0.0.0 --port 8000 --dtype half \\"
echo "       --max-model-len 2048 --gpu-memory-utilization 0.85 \\"
echo "       > logs/vllm.log 2>&1 &"
echo ""
echo "3. Start Streamlit demo:"
echo "   nohup streamlit run app/demo.py --server.port 8501 --server.address 0.0.0.0 > logs/streamlit.log 2>&1 &"
echo ""
echo "4. Train model:"
echo "   python -m src.training.train_v2 \\"
echo "       --train_data data/processed/train.jsonl \\"
echo "       --eval_data data/processed/val.jsonl \\"
echo "       --output_dir outputs/ciq-model-v2 \\"
echo "       --epochs 3"
echo ""
