# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

E-commerce LLM system for CommerceIQ demonstrating product classification, attribute extraction, and Q&A capabilities using fine-tuned language models.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   FastAPI    │────▶│    vLLM      │────▶│  Fine-tuned  │
│   Gateway    │     │   Server     │     │  Mistral-7B  │
└──────────────┘     └──────────────┘     └──────────────┘
        │                                         │
        ▼                                         │
┌──────────────┐                                  │
│  ChromaDB    │◀─────────────────────────────────┘
│ (RAG Store)  │
└──────────────┘
```

**Core Components:**
- **FastAPI Gateway**: API endpoints for classification, extraction, and Q&A
- **vLLM Server**: High-performance inference with PagedAttention
- **Fine-tuned Model**: QLoRA-trained Mistral-7B/LLaMA-3-8B
- **ChromaDB**: Vector store for RAG-based Q&A

## Core Tasks

Three primary e-commerce tasks with task-specific prefixes in prompts:
- `[CLASSIFY]` - Hierarchical product categorization (5,595+ Google Product Taxonomy categories)
- `[EXTRACT]` - Attribute-value extraction as JSON
- `[QA]` - Product question answering with RAG

## Key Datasets

| Dataset | Purpose | Source |
|---------|---------|--------|
| MAVE | 3M attribute annotations | google-research-datasets/MAVE |
| ECInstruct | 116K multi-task examples | NingLab/ECInstruct |
| Amazon ESCI | 2.6M relevance judgments | amazon-science/esci-data |
| AmazonQA | 923K product Q&A pairs | amazonqa/amazonqa |

## Build & Run Commands

```bash
# Environment setup
pip install torch transformers peft trl bitsandbytes vllm fastapi chromadb

# Start vLLM inference server
python -m vllm.entrypoints.openai.api_server \
    --model=mistralai/Mistral-7B-Instruct-v0.3 \
    --dtype=half \
    --max-model-len=4096 \
    --gpu-memory-utilization=0.85

# Run FastAPI application
uvicorn app.main:app --reload
```

## Training Configuration

QLoRA fine-tuning parameters for 7B models (8-12GB VRAM):
- Quantization: 4-bit NF4 with double quantization
- LoRA rank: 32, alpha: 64
- Target modules: all linear layers (q/k/v/o_proj, gate/up/down_proj)
- Epochs: 3 (avoid overtraining)
- Include ~10% general instruction data to prevent catastrophic forgetting

## Evaluation Metrics

| Task | Primary Metrics |
|------|-----------------|
| Classification | Weighted F1, Hierarchical F1, Top-k accuracy |
| Extraction | Exact match, Token F1, Slot accuracy |
| Q&A | ROUGE-L, BERTScore |

## Important Constraints

- Use temperature=0 during evaluation for consistent results
- Validate JSON outputs with try-catch (LLMs can produce malformed JSON)
- For production: prioritize FP8 quantization → INT8 SmoothQuant → AWQ 4-bit
- Enable vLLM prefix caching for repeated template-based queries


For every project, write a detailed FOR_Nikhil.md file that explains the whole project in plain language. 

Explain the technical architecture, the structure of the codebase and how the various parts are connected, the technologies used, why we made these technical decisions, and lessons I can learn from it (this should include the bugs we ran into and how we fixed them, potential pitfalls and how to avoid them in the future, new technologies used, how good engineers think and work, best practices, etc). 

It should be very engaging to read; don't make it sound like boring technical documentation/textbook. Where appropriate, use analogies and anecdotes to make it more understandable and memorable.