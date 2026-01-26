# AWS Deployment Guide for E-Commerce LLM

A practical guide to running your fine-tuning experiments on AWS with $200 in credits.

---

## Budget Strategy: Making $200 Go Far

With spot instances, your credits stretch significantly:

| Instance | GPU | VRAM | On-Demand | Spot Price | Hours for $200 |
|----------|-----|------|-----------|------------|----------------|
| g5.xlarge | A10G | 24GB | ~$1.00/hr | ~$0.40/hr | **500 hours** |
| g5.2xlarge | A10G | 24GB | ~$1.21/hr | ~$0.50/hr | 400 hours |
| p3.2xlarge | V100 | 16GB | ~$3.06/hr | ~$1.00/hr | 200 hours |

**Recommendation:** Use `g5.xlarge` spot instances. 500 hours = 50+ complete training cycles.

---

## Datasets Overview

Before you start, here are the datasets you'll need:

### Required Datasets (Must Have)

| Dataset | Size | Purpose | Link |
|---------|------|---------|------|
| **ECInstruct** | 116K examples | Main training data for all 3 tasks | [HuggingFace](https://huggingface.co/datasets/NingLab/ECInstruct) |
| **Alpaca** | 52K examples | Prevents catastrophic forgetting (10% mix) | [HuggingFace](https://huggingface.co/datasets/tatsu-lab/alpaca) |

### Optional Datasets (For Extended Experiments)

| Dataset | Size | Purpose | Link |
|---------|------|---------|------|
| MAVE | 3M annotations | Additional attribute extraction data | [GitHub](https://github.com/google-research-datasets/MAVE) |
| Amazon ESCI | 2.6M judgments | Search relevance training | [GitHub](https://github.com/amazon-science/esci-data) |
| AmazonQA | 923K Q&A | Product Q&A training | [GitHub](https://github.com/amazonqa/amazonqa) |

### Download Script

Run this on your local machine or EC2 instance:

```bash
# Download required datasets (~550 MB)
python scripts/download_datasets.py

# Or with optional datasets too
python scripts/download_datasets.py --include-optional
```

The datasets download from HuggingFace automatically - no manual download needed for required ones.

---

## Phase 1: Account Security (Do This First)

### 1.1 Enable MFA on Root Account

1. Sign in to AWS Console as root
2. Go to **IAM** → **Security credentials** (top right dropdown)
3. Under "Multi-factor authentication (MFA)", click **Assign MFA device**
4. Choose "Authenticator app" and scan with Google Authenticator/Authy

### 1.2 Create IAM User for Daily Work

Never use root for daily tasks. Create an IAM user:

1. Go to **IAM** → **Users** → **Create user**
2. Username: `nikhil-dev` (or your name)
3. Select "Provide user access to the AWS Management Console"
4. Create a custom password
5. **Attach policies directly:**
   - `AmazonEC2FullAccess`
   - `AmazonS3FullAccess`
   - `AmazonVPCFullAccess`
   - `ServiceQuotasFullAccess`
6. Create user and save the credentials

### 1.3 Set Up Billing Alerts

Avoid surprise charges:

1. Go to **Billing** → **Budgets** → **Create budget**
2. Choose "Cost budget"
3. Budget name: `CIQ-Learning-Budget`
4. Budget amount: `200` USD (Monthly)
5. **Add alerts at:**
   - 25% ($50) - Info
   - 50% ($100) - Warning
   - 75% ($150) - Critical
6. Add your email for notifications

---

## Phase 2: Request GPU Quota (Do Now - Takes 24-48 Hours)

AWS limits GPU access by default. Request an increase:

1. Go to **Service Quotas** → **AWS services** → **Amazon EC2**
2. Search for: `Running On-Demand G and VT instances`
3. Click on it → **Request quota increase**
4. **Region:** `us-east-1`
5. **New quota value:** `4` (g5.xlarge needs 4 vCPUs)
6. Submit and wait for approval email

**Also request spot quota:**
- Search for: `All G and VT Spot Instance Requests`
- Request: `4` vCPUs

---

## Phase 3: Install AWS CLI Locally

On your Mac:

```bash
# Install AWS CLI
brew install awscli

# Configure with your IAM user credentials
aws configure
# AWS Access Key ID: <from IAM user creation>
# AWS Secret Access Key: <from IAM user creation>
# Default region: us-east-1
# Default output format: json

# Verify it works
aws sts get-caller-identity
```

---

## Phase 4: Set Up S3 Storage

### 4.1 Create Your Bucket

```bash
# Create bucket (must be globally unique name)
aws s3 mb s3://ciq-nikhil-ecommerce --region us-east-1

# Verify
aws s3 ls
```

### 4.2 Create Folder Structure

```bash
# Create logical folders (S3 doesn't have real folders, but prefixes work)
aws s3api put-object --bucket ciq-nikhil-ecommerce --key datasets/
aws s3api put-object --bucket ciq-nikhil-ecommerce --key checkpoints/
aws s3api put-object --bucket ciq-nikhil-ecommerce --key models/
aws s3api put-object --bucket ciq-nikhil-ecommerce --key logs/
```

### 4.3 Upload Your Training Data

```bash
# From your ciq project directory
cd /Users/nikhilsanghi/Documents/Synthesis_Labs_Projects/ciq

# Upload data folder
aws s3 sync ./data s3://ciq-nikhil-ecommerce/datasets/

# Verify upload
aws s3 ls s3://ciq-nikhil-ecommerce/datasets/ --recursive
```

---

## Phase 5: Create Key Pair for SSH

```bash
# Create key pair
aws ec2 create-key-pair \
    --key-name ciq-gpu-key \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/ciq-gpu-key.pem

# Set correct permissions
chmod 400 ~/.ssh/ciq-gpu-key.pem
```

---

## Phase 6: Create Security Group

```bash
# Get your current IP
MY_IP=$(curl -s https://checkip.amazonaws.com)

# Create security group
aws ec2 create-security-group \
    --group-name ciq-gpu-sg \
    --description "Security group for CIQ GPU instances"

# Allow SSH only from your IP
aws ec2 authorize-security-group-ingress \
    --group-name ciq-gpu-sg \
    --protocol tcp \
    --port 22 \
    --cidr ${MY_IP}/32

# Allow port 8000 (vLLM) and 8080 (FastAPI) from your IP
aws ec2 authorize-security-group-ingress \
    --group-name ciq-gpu-sg \
    --protocol tcp \
    --port 8000 \
    --cidr ${MY_IP}/32

aws ec2 authorize-security-group-ingress \
    --group-name ciq-gpu-sg \
    --protocol tcp \
    --port 8080 \
    --cidr ${MY_IP}/32
```

---

## Phase 7: Launch GPU Instance

### 7.1 Find the Deep Learning AMI

```bash
# Find the latest Deep Learning AMI (Ubuntu, PyTorch)
aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text
```

Save this AMI ID (looks like `ami-0abc123...`).

### 7.2 Launch Spot Instance

```bash
# Replace ami-XXXXX with the AMI ID from above
aws ec2 run-instances \
    --image-id ami-XXXXX \
    --instance-type g5.xlarge \
    --key-name ciq-gpu-key \
    --security-groups ciq-gpu-sg \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ciq-training}]' \
    --query 'Instances[0].InstanceId' \
    --output text
```

### 7.3 Get Instance Public IP

```bash
# Wait a minute for the instance to start, then:
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=ciq-training" "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text
```

---

## Phase 8: Connect and Set Up Instance

### 8.1 SSH into Instance

```bash
ssh -i ~/.ssh/ciq-gpu-key.pem ubuntu@<PUBLIC_IP>
```

### 8.2 Verify GPU

```bash
# Check GPU is recognized
nvidia-smi

# Should show: NVIDIA A10G with 24GB memory
```

### 8.3 Set Up Project

```bash
# The Deep Learning AMI has conda pre-installed
conda create -n ciq python=3.10 -y
conda activate ciq

# Clone your project (if using GitHub)
git clone https://github.com/YOUR_USERNAME/ciq.git
cd ciq

# Or sync from S3 if not using git
# aws s3 sync s3://ciq-nikhil-ecommerce/code/ ./

# Install dependencies
pip install torch transformers peft trl bitsandbytes vllm fastapi chromadb accelerate
pip install -r requirements.txt  # if you have one

# Download training data from S3
aws s3 sync s3://ciq-nikhil-ecommerce/datasets/ ./data/

# Verify PyTorch sees the GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Phase 9: Run Training

### 9.1 Use tmux (Keeps Running After SSH Disconnect)

```bash
# Start tmux session
tmux new -s training

# Now run your training inside tmux
```

### 9.2 Start Training

```bash
# Activate environment
conda activate ciq

# Run training
python -m src.training.trainer \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --train_data ./data/train.jsonl \
    --eval_data ./data/eval.jsonl \
    --output_dir ./outputs/ciq-model \
    --lora_r 32 \
    --learning_rate 2e-4 \
    --epochs 3

# To detach from tmux (training continues): Ctrl+B, then D
# To reattach later: tmux attach -t training
```

### 9.3 Monitor Training

```bash
# In a new terminal/tmux window
watch -n 5 nvidia-smi  # GPU utilization

# Check training logs
tail -f ./outputs/ciq-model/training.log
```

### 9.4 Save Checkpoints to S3 (Do Periodically)

```bash
# Sync outputs to S3 (run in another tmux window)
aws s3 sync ./outputs/ s3://ciq-nikhil-ecommerce/checkpoints/ --exclude "*.bin"

# After training completes, sync everything including model weights
aws s3 sync ./outputs/ s3://ciq-nikhil-ecommerce/models/
```

---

## Phase 10: Run Inference

### 10.1 Start vLLM Server

```bash
tmux new -s vllm

python -m vllm.entrypoints.openai.api_server \
    --model ./outputs/ciq-model \
    --dtype half \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --port 8000

# Detach: Ctrl+B, D
```

### 10.2 Start FastAPI

```bash
tmux new -s api

cd ~/ciq
uvicorn app.main:app --host 0.0.0.0 --port 8080

# Detach: Ctrl+B, D
```

### 10.3 Test from Your Local Machine

```bash
# From your Mac
curl http://<INSTANCE_IP>:8080/health

curl -X POST http://<INSTANCE_IP>:8080/classify \
    -H "Content-Type: application/json" \
    -d '{"text": "Apple iPhone 15 Pro Max 256GB Black Titanium"}'
```

---

## Phase 11: Cost Control (CRITICAL)

### Stop Instance When Not Using

```bash
# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=ciq-training" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text)

# Stop instance (keeps data, stops billing for compute)
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Start again when needed
aws ec2 start-instances --instance-ids $INSTANCE_ID
```

### Check Spending

```bash
# Quick cost check (last 7 days)
aws ce get-cost-and-usage \
    --time-period Start=$(date -v-7d +%Y-%m-%d),End=$(date +%Y-%m-%d) \
    --granularity DAILY \
    --metrics BlendedCost \
    --query 'ResultsByTime[*].[TimePeriod.Start,Total.BlendedCost.Amount]' \
    --output table
```

### Daily Checklist

- [ ] Check AWS Billing Dashboard
- [ ] Stop GPU instance if not actively using
- [ ] Delete old checkpoints from S3 you don't need
- [ ] Verify no orphaned resources (EBS volumes, etc.)

---

## Cost Estimates

| Activity | Instance | Duration | Spot Cost |
|----------|----------|----------|-----------|
| Training (50K samples) | g5.xlarge | ~3 hours | ~$1.20 |
| Training (full ECInstruct 116K) | g5.xlarge | ~6 hours | ~$2.40 |
| Inference testing | g5.xlarge | 2 hours | ~$0.80 |
| S3 storage (10GB/month) | - | 1 month | ~$0.23 |
| **Full experiment cycle** | - | 8 hours | **~$3.20** |

**With $200, you can run 60+ complete experiment cycles.**

---

## Troubleshooting

### "InsufficientInstanceCapacity" Error

Spot instances may not be available. Try:
```bash
# Check spot pricing and availability
aws ec2 describe-spot-price-history \
    --instance-types g5.xlarge \
    --product-descriptions "Linux/UNIX" \
    --start-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
    --query 'SpotPriceHistory[*].[AvailabilityZone,SpotPrice]'

# Try a different availability zone
--placement '{"AvailabilityZone":"us-east-1b"}'
```

### SSH Connection Refused

```bash
# Check instance is running
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].State.Name'

# Check security group allows your current IP
MY_IP=$(curl -s https://checkip.amazonaws.com)
echo "Your IP: $MY_IP"
# If IP changed, update security group
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# If not working, the AMI may need driver reinstall
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot
```

### Out of GPU Memory

Reduce batch size or model context:
```bash
# Reduce batch size
--per_device_train_batch_size 2

# Or reduce context length
--max-model-len 2048
```

---

## Quick Reference Commands

```bash
# Start instance
aws ec2 start-instances --instance-ids $INSTANCE_ID

# Stop instance
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Get public IP
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# SSH connect
ssh -i ~/.ssh/ciq-gpu-key.pem ubuntu@<IP>

# Sync to S3
aws s3 sync ./outputs/ s3://ciq-nikhil-ecommerce/models/

# Download from S3
aws s3 sync s3://ciq-nikhil-ecommerce/models/ ./outputs/

# Check costs
aws ce get-cost-and-usage --time-period Start=2025-01-01,End=2025-01-31 \
    --granularity MONTHLY --metrics BlendedCost
```

---

## Next Steps After First Successful Training

1. **Compare models:** Run same training with LLaMA-3-8B (see Future Work in FOR_NIKHIL.md)
2. **Optimize:** Try different LoRA ranks, learning rates
3. **Evaluate:** Run full evaluation suite on held-out test data
4. **Production:** Consider keeping a small inference instance running

Good luck with your experiments!
