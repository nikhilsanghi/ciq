#!/bin/bash
# Start the comparison demo (requires both vLLM servers running)
#
# Prerequisites:
#   ./scripts/start_multi_model.sh
#
# Usage:
#   ./scripts/start_compare_demo.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "E-Commerce LLM - Comparison Demo"
echo "=========================================="

# Check if both servers are running
FINETUNED_READY=$(curl -s http://localhost:8000/v1/models 2>/dev/null && echo "yes" || echo "no")
BASE_READY=$(curl -s http://localhost:8001/v1/models 2>/dev/null && echo "yes" || echo "no")

if [[ "$FINETUNED_READY" != "yes" ]] || [[ "$BASE_READY" != "yes" ]]; then
    echo -e "${YELLOW}Warning: Not all vLLM servers are running${NC}"
    echo ""
    echo "Server status:"
    [[ "$FINETUNED_READY" == "yes" ]] && echo -e "  ${GREEN}Fine-tuned (8000): Running${NC}" || echo "  Fine-tuned (8000): Not running"
    [[ "$BASE_READY" == "yes" ]] && echo -e "  ${GREEN}Base (8001): Running${NC}" || echo "  Base (8001): Not running"
    echo ""
    echo "Start both servers first:"
    echo "  ./scripts/start_multi_model.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create logs directory
mkdir -p logs

# Kill any existing Streamlit on port 8501
pkill -f "streamlit run app/compare_demo.py" 2>/dev/null || true
sleep 1

echo ""
echo -e "${GREEN}Starting Comparison Demo on port 8501...${NC}"

nohup streamlit run app/compare_demo.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    > logs/streamlit_compare.log 2>&1 &

STREAMLIT_PID=$!
echo "  PID: $STREAMLIT_PID"

# Wait for startup
sleep 3

echo ""
echo "=========================================="
echo -e "${GREEN}Comparison Demo Ready!${NC}"
echo "=========================================="
echo ""
echo "Access the demo at:"
echo "  http://localhost:8501 (local)"
echo "  http://<EC2-PUBLIC-IP>:8501 (remote)"
echo ""
echo "Log: tail -f logs/streamlit_compare.log"
echo "Stop: pkill -f 'streamlit run app/compare_demo.py'"
echo ""
