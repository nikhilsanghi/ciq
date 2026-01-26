#!/bin/bash
# Start vLLM inference server for e-commerce LLM
# Usage: ./scripts/start_vllm.sh [model_path] [port] [gpu_memory_util]
#
# Examples:
#   ./scripts/start_vllm.sh                                    # Use defaults
#   ./scripts/start_vllm.sh models/final 8000 0.85
#   ./scripts/start_vllm.sh mistralai/Mistral-7B-Instruct-v0.3 8001 0.9

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default parameters
MODEL_PATH=${1:-"models/final"}
PORT=${2:-8000}
GPU_MEMORY_UTIL=${3:-0.85}
MAX_MODEL_LEN=${4:-4096}

# Determine project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log_info "Starting vLLM inference server..."
log_info "Project root: $PROJECT_ROOT"
log_info "Model: $MODEL_PATH"
log_info "Port: $PORT"
log_info "GPU Memory Utilization: $GPU_MEMORY_UTIL"
log_info "Max Model Length: $MAX_MODEL_LEN"

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    log_error "vLLM is not installed. Install with: pip install vllm"
    exit 1
fi

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    log_info "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
else
    log_error "No GPU detected. vLLM requires GPU for inference."
    exit 1
fi

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    log_error "Port $PORT is already in use."
    log_error "Kill the existing process or use a different port."
    exit 1
fi

# Resolve model path
FULL_MODEL_PATH="$PROJECT_ROOT/$MODEL_PATH"
if [[ -d "$FULL_MODEL_PATH" ]]; then
    MODEL_PATH="$FULL_MODEL_PATH"
    log_info "Using local model: $MODEL_PATH"
else
    log_info "Using HuggingFace model: $MODEL_PATH"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Build vLLM command
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --dtype half \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --enable-prefix-caching \
    --port $PORT \
    --host 0.0.0.0"

# Add tensor parallelism if multiple GPUs available
GPU_COUNT=$(nvidia-smi -L | wc -l)
if [[ $GPU_COUNT -gt 1 ]]; then
    log_info "Detected $GPU_COUNT GPUs. Enabling tensor parallelism."
    VLLM_CMD="$VLLM_CMD --tensor-parallel-size $GPU_COUNT"
fi

log_info "Starting vLLM server..."
log_info "OpenAI-compatible API will be available at: http://0.0.0.0:$PORT"
log_info "API documentation at: http://0.0.0.0:$PORT/docs"
echo ""
log_info "Press Ctrl+C to stop the server"
echo ""

# Run vLLM server
exec $VLLM_CMD
