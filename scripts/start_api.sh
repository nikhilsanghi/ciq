#!/bin/bash
# Start FastAPI application for e-commerce LLM
# Usage: ./scripts/start_api.sh [host] [port] [workers]
#
# Examples:
#   ./scripts/start_api.sh                    # Development mode with reload
#   ./scripts/start_api.sh 0.0.0.0 8080 1     # Development defaults
#   ./scripts/start_api.sh 0.0.0.0 8080 4     # Production with 4 workers

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
HOST=${1:-"0.0.0.0"}
PORT=${2:-8080}
WORKERS=${3:-1}
RELOAD=${RELOAD:-"true"}

# Determine project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log_info "Starting FastAPI application..."
log_info "Project root: $PROJECT_ROOT"
log_info "Host: $HOST"
log_info "Port: $PORT"
log_info "Workers: $WORKERS"

# Change to project root
cd "$PROJECT_ROOT"

# Check if uvicorn is installed
if ! python -c "import uvicorn" 2>/dev/null; then
    log_error "uvicorn is not installed. Install with: pip install uvicorn[standard]"
    exit 1
fi

# Check if FastAPI app exists
if ! python -c "from src.api.main import app" 2>/dev/null; then
    log_warn "FastAPI app not found at src.api.main:app"
    log_warn "Ensure the module exists and is properly configured."
fi

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    log_error "Port $PORT is already in use."
    log_error "Kill the existing process or use a different port."
    exit 1
fi

# Check vLLM server connectivity (optional)
VLLM_URL=${VLLM_URL:-"http://localhost:8000"}
if curl -s --connect-timeout 2 "$VLLM_URL/health" >/dev/null 2>&1; then
    log_info "vLLM server detected at: $VLLM_URL"
else
    log_warn "vLLM server not detected at: $VLLM_URL"
    log_warn "Start vLLM server for full functionality: ./scripts/start_vllm.sh"
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT"

# Build uvicorn command
if [[ "$WORKERS" -gt 1 ]]; then
    # Production mode: multiple workers, no reload
    log_info "Running in production mode with $WORKERS workers"
    UVICORN_CMD="uvicorn src.api.main:app \
        --host $HOST \
        --port $PORT \
        --workers $WORKERS \
        --access-log \
        --log-level info"
else
    # Development mode: single worker with reload
    if [[ "$RELOAD" == "true" ]]; then
        log_info "Running in development mode with auto-reload"
        UVICORN_CMD="uvicorn src.api.main:app \
            --host $HOST \
            --port $PORT \
            --reload \
            --reload-dir $PROJECT_ROOT/src \
            --access-log \
            --log-level debug"
    else
        log_info "Running in single-worker mode without reload"
        UVICORN_CMD="uvicorn src.api.main:app \
            --host $HOST \
            --port $PORT \
            --access-log \
            --log-level info"
    fi
fi

log_info "API will be available at: http://$HOST:$PORT"
log_info "API documentation at: http://$HOST:$PORT/docs"
log_info "Alternative docs at: http://$HOST:$PORT/redoc"
echo ""
log_info "Press Ctrl+C to stop the server"
echo ""

# Run uvicorn server
exec $UVICORN_CMD
