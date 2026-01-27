#!/bin/bash
# Configure AWS Security Group for demo access
#
# This script opens the necessary ports for:
# - Streamlit (8501)
# - vLLM API (8000)
# - SSH (22) - usually already open
#
# Usage:
#   ./scripts/setup_aws_security.sh <security-group-id>
#
# Example:
#   ./scripts/setup_aws_security.sh sg-0123456789abcdef0

set -e

if [[ -z "$1" ]]; then
    echo "Usage: ./scripts/setup_aws_security.sh <security-group-id>"
    echo ""
    echo "Find your security group ID:"
    echo "  aws ec2 describe-instances --query 'Reservations[].Instances[].SecurityGroups[].GroupId' --output text"
    exit 1
fi

SG_ID="$1"

echo "=========================================="
echo "Configuring Security Group: $SG_ID"
echo "=========================================="

# Allow Streamlit (8501)
echo "Adding rule for Streamlit (port 8501)..."
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 8501 \
    --cidr 0.0.0.0/0 \
    2>/dev/null || echo "  (Rule may already exist)"

# Allow vLLM API (8000)
echo "Adding rule for vLLM API (port 8000)..."
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0 \
    2>/dev/null || echo "  (Rule may already exist)"

# Allow alternative Streamlit port (8080)
echo "Adding rule for alternative web (port 8080)..."
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 8080 \
    --cidr 0.0.0.0/0 \
    2>/dev/null || echo "  (Rule may already exist)"

echo ""
echo "=========================================="
echo "Security Group Configuration Complete"
echo "=========================================="
echo ""
echo "Open ports:"
echo "  • 22 (SSH)"
echo "  • 8000 (vLLM API)"
echo "  • 8080 (Alternative web)"
echo "  • 8501 (Streamlit)"
echo ""
echo "Your demo will be accessible at:"
echo "  http://<EC2-Public-IP>:8501"
echo ""
