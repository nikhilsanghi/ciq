#!/bin/bash
# Evaluation script for e-commerce LLM
# Usage: ./scripts/evaluate.sh [model_path] [task] [output_dir]
#
# Tasks:
#   - classification: Product categorization evaluation
#   - extraction: Attribute extraction evaluation
#   - qa: Question-answering evaluation
#   - all: Run all evaluations
#
# Examples:
#   ./scripts/evaluate.sh                                           # Use defaults
#   ./scripts/evaluate.sh models/checkpoints/ecommerce-qlora classification
#   ./scripts/evaluate.sh models/final all results/

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_task() {
    echo -e "${BLUE}[TASK]${NC} $1"
}

# Default parameters
MODEL_PATH=${1:-"models/checkpoints/ecommerce-qlora"}
TASK=${2:-"all"}  # classification, extraction, qa, or all
OUTPUT_DIR=${3:-"results"}

# Determine project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Validate task parameter
VALID_TASKS=("classification" "extraction" "qa" "all")
if [[ ! " ${VALID_TASKS[*]} " =~ " ${TASK} " ]]; then
    log_error "Invalid task: $TASK"
    log_error "Valid tasks: ${VALID_TASKS[*]}"
    exit 1
fi

log_info "Starting e-commerce LLM evaluation..."
log_info "Project root: $PROJECT_ROOT"
log_info "Model path: $MODEL_PATH"
log_info "Task: $TASK"
log_info "Output directory: $OUTPUT_DIR"

# Validate model path exists
FULL_MODEL_PATH="$PROJECT_ROOT/$MODEL_PATH"
if [[ ! -d "$FULL_MODEL_PATH" ]]; then
    log_warn "Model directory not found at: $FULL_MODEL_PATH"
    log_warn "Attempting to use as HuggingFace model ID..."
    FULL_MODEL_PATH="$MODEL_PATH"
fi

# Create output directory
FULL_OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
mkdir -p "$FULL_OUTPUT_DIR"

# Set evaluation timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Function to run single task evaluation
run_evaluation() {
    local task_name=$1
    local output_file="$FULL_OUTPUT_DIR/evaluation_${task_name}_${TIMESTAMP}.json"

    log_task "Running $task_name evaluation..."

    python -m src.evaluation.evaluate \
        --model_path "$FULL_MODEL_PATH" \
        --task "$task_name" \
        --output_path "$output_file" \
        --temperature 0

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        log_info "$task_name evaluation complete: $output_file"
        return 0
    else
        log_error "$task_name evaluation failed with exit code: $exit_code"
        return $exit_code
    fi
}

# Run evaluations based on task parameter
EVAL_EXIT_CODE=0

if [[ "$TASK" == "all" ]]; then
    log_info "Running all evaluation tasks..."

    for task in "classification" "extraction" "qa"; do
        if ! run_evaluation "$task"; then
            EVAL_EXIT_CODE=1
            log_warn "Continuing with remaining tasks despite failure..."
        fi
        echo ""
    done
else
    if ! run_evaluation "$TASK"; then
        EVAL_EXIT_CODE=1
    fi
fi

# Summary
echo ""
log_info "===== Evaluation Summary ====="
log_info "Results saved to: $FULL_OUTPUT_DIR"

if [[ -d "$FULL_OUTPUT_DIR" ]]; then
    log_info "Generated files:"
    ls -lh "$FULL_OUTPUT_DIR"/evaluation_*_${TIMESTAMP}.json 2>/dev/null || true
fi

if [[ $EVAL_EXIT_CODE -eq 0 ]]; then
    log_info "All evaluations completed successfully!"
else
    log_warn "Some evaluations failed. Check logs for details."
fi

exit $EVAL_EXIT_CODE
