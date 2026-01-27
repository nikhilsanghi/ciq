#!/bin/bash
# Full deployment script for E-Commerce LLM Demo
#
# This script:
# 1. Merges LoRA adapters with base model (if not done)
# 2. Starts vLLM server in background
# 3. Starts Streamlit demo
#
# Usage:
#   ./scripts/deploy_full.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "E-Commerce LLM Full Deployment"
echo "=========================================="

# Check if we're in the right directory
if [[ ! -f "app/demo.py" ]]; then
    echo -e "${RED}Error: Run this script from the project root (~/ciq)${NC}"
    exit 1
fi

# Step 1: Check/Merge model
echo ""
echo -e "${GREEN}Step 1: Checking model...${NC}"

if [[ -d "./outputs/ciq-model-merged" ]]; then
    echo "✓ Merged model found at ./outputs/ciq-model-merged"
else
    if [[ -d "./outputs/ciq-model" ]]; then
        echo "Merging LoRA adapters with base model..."
        python -m src.training.trainer \
            --merge_adapters ./outputs/ciq-model \
            --merge_output ./outputs/ciq-model-merged
        echo "✓ Model merged successfully"
    else
        echo -e "${RED}Error: No trained model found at ./outputs/ciq-model${NC}"
        echo "Please complete training first."
        exit 1
    fi
fi

# Step 2: Start vLLM server
echo ""
echo -e "${GREEN}Step 2: Starting vLLM server...${NC}"

# Kill any existing vLLM process
pkill -f "vllm.entrypoints" || true
sleep 2

# Start vLLM in background
nohup ./scripts/start_vllm.sh > logs/vllm.log 2>&1 &
VLLM_PID=$!
echo "vLLM starting (PID: $VLLM_PID)"

# Wait for vLLM to be ready
echo "Waiting for vLLM to initialize (this may take 1-2 minutes)..."
for i in {1..60}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✓ vLLM server is ready"
        break
    fi
    sleep 2
    echo -n "."
done

# Check if vLLM started successfully
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo -e "${RED}Error: vLLM failed to start. Check logs/vllm.log${NC}"
    exit 1
fi

# Step 3: Start Streamlit
echo ""
echo -e "${GREEN}Step 3: Starting Streamlit demo...${NC}"

# Kill any existing Streamlit process
pkill -f "streamlit run" || true
sleep 1

# Start Streamlit in background
nohup ./scripts/start_demo.sh > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "Streamlit starting (PID: $STREAMLIT_PID)"

sleep 5

# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_EC2_IP")

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "Services running:"
echo "  • vLLM Server: http://localhost:8000"
echo "  • Streamlit Demo: http://localhost:8501"
echo ""
echo "Share this URL with your friends:"
echo -e "  ${GREEN}http://${PUBLIC_IP}:8501${NC}"
echo ""
echo "Logs:"
echo "  • vLLM: tail -f logs/vllm.log"
echo "  • Streamlit: tail -f logs/streamlit.log"
echo ""
echo "To stop all services:"
echo "  pkill -f 'vllm.entrypoints' && pkill -f 'streamlit run'"
echo ""
