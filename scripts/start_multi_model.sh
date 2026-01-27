#!/bin/bash
# Start multiple vLLM instances for model comparison
#
# This runs two vLLM servers:
# - Port 8000: Fine-tuned model
# - Port 8001: Base Mistral model
#
# Usage:
#   ./scripts/start_multi_model.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Starting Multi-Model vLLM Setup"
echo "=========================================="

# Check for merged model
if [[ ! -d "./outputs/ciq-model-merged" ]]; then
    echo -e "${YELLOW}Warning: Merged model not found. Running merge first...${NC}"
    if [[ -d "./outputs/ciq-model" ]]; then
        python -m src.training.trainer \
            --merge_adapters ./outputs/ciq-model \
            --merge_output ./outputs/ciq-model-merged
    else
        echo "Error: No trained model found. Complete training first."
        exit 1
    fi
fi

# Kill any existing vLLM processes
echo "Stopping any existing vLLM processes..."
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

# Start fine-tuned model on port 8000
echo ""
echo -e "${GREEN}Starting Fine-tuned Mistral on port 8000...${NC}"
nohup python -m vllm.entrypoints.openai.api_server \
    --model ./outputs/ciq-model-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype half \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.45 \
    > logs/vllm_finetuned.log 2>&1 &
FINETUNED_PID=$!
echo "  PID: $FINETUNED_PID"

# Wait a bit for first model to load
echo "Waiting for fine-tuned model to initialize..."
sleep 30

# Start base model on port 8001
echo ""
echo -e "${GREEN}Starting Base Mistral on port 8001...${NC}"
nohup python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype half \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.45 \
    > logs/vllm_base.log 2>&1 &
BASE_PID=$!
echo "  PID: $BASE_PID"

# Wait for models to be ready
echo ""
echo "Waiting for models to be ready..."
for i in {1..60}; do
    FINETUNED_READY=$(curl -s http://localhost:8000/v1/models 2>/dev/null && echo "yes" || echo "no")
    BASE_READY=$(curl -s http://localhost:8001/v1/models 2>/dev/null && echo "yes" || echo "no")

    if [[ "$FINETUNED_READY" == "yes" ]] && [[ "$BASE_READY" == "yes" ]]; then
        break
    fi
    echo -n "."
    sleep 5
done
echo ""

# Check status
echo ""
echo "=========================================="
echo "Model Status"
echo "=========================================="

if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Fine-tuned model ready on port 8000${NC}"
else
    echo -e "${YELLOW}✗ Fine-tuned model not ready - check logs/vllm_finetuned.log${NC}"
fi

if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Base model ready on port 8001${NC}"
else
    echo -e "${YELLOW}✗ Base model not ready - check logs/vllm_base.log${NC}"
fi

echo ""
echo "Logs:"
echo "  Fine-tuned: tail -f logs/vllm_finetuned.log"
echo "  Base: tail -f logs/vllm_base.log"
echo ""
echo "To stop: pkill -f 'vllm.entrypoints'"
echo ""
