#!/bin/bash
# Start the Streamlit demo app
#
# Usage:
#   ./scripts/start_demo.sh              # Start on default port 8501
#   ./scripts/start_demo.sh --port 8080  # Start on custom port

set -e

# Configuration
PORT=8501
HOST="0.0.0.0"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "=========================================="
echo "Starting E-Commerce LLM Demo"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo ""
echo "Access the demo at:"
echo "  Local: http://localhost:$PORT"
echo "  External: http://$(curl -s ifconfig.me):$PORT"
echo ""
echo "Make sure vLLM server is running on port 8000"
echo "=========================================="
echo ""

# Install streamlit if not present
pip install streamlit requests --quiet

# Start Streamlit
streamlit run app/demo.py \
    --server.address "$HOST" \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
