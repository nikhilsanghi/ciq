#!/bin/bash
# Training script for e-commerce LLM
# Usage: ./scripts/train.sh [model_name] [experiment_name] [config_path]
#
# Examples:
#   ./scripts/train.sh                                    # Use defaults
#   ./scripts/train.sh mistralai/Mistral-7B-Instruct-v0.3 my-experiment
#   ./scripts/train.sh meta-llama/Meta-Llama-3-8B-Instruct llama-qlora configs/custom.yaml

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
MODEL_NAME=${1:-"mistralai/Mistral-7B-Instruct-v0.3"}
EXPERIMENT_NAME=${2:-"ecommerce-qlora"}
CONFIG_PATH=${3:-"configs/training_config.yaml"}

# Determine project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log_info "Starting e-commerce LLM training..."
log_info "Project root: $PROJECT_ROOT"
log_info "Model: $MODEL_NAME"
log_info "Experiment: $EXPERIMENT_NAME"
log_info "Config: $CONFIG_PATH"

# Validate config file exists
if [[ ! -f "$PROJECT_ROOT/$CONFIG_PATH" ]]; then
    log_error "Config file not found: $PROJECT_ROOT/$CONFIG_PATH"
    exit 1
fi

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    log_info "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    log_warn "No GPU detected. Training will be slow on CPU."
fi

# Create output directories
OUTPUT_DIR="$PROJECT_ROOT/models/checkpoints/$EXPERIMENT_NAME"
LOGGING_DIR="$PROJECT_ROOT/logs/$EXPERIMENT_NAME"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGGING_DIR"

log_info "Output directory: $OUTPUT_DIR"
log_info "Logging directory: $LOGGING_DIR"

# Set environment variables for optimal training
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info

# Optional: Set CUDA settings for better memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run training
log_info "Launching training process..."

python -m src.training.trainer \
    --model_name "$MODEL_NAME" \
    --config "$PROJECT_ROOT/$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$LOGGING_DIR"

TRAIN_EXIT_CODE=$?

if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    log_info "Training completed successfully!"
    log_info "Checkpoints saved to: $OUTPUT_DIR"

    # List saved checkpoints
    if [[ -d "$OUTPUT_DIR" ]]; then
        log_info "Saved artifacts:"
        ls -lh "$OUTPUT_DIR"
    fi
else
    log_error "Training failed with exit code: $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi
