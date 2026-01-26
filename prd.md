# Building an e-commerce LLM system: A practical implementation guide

**For a 2-day demo showcasing product classification, attribute extraction, and Q&A, you can achieve production-quality results by combining the MAVE dataset (3M attribute annotations), QLoRA fine-tuning on Mistral-7B (~3 hours on consumer GPU), and vLLM for inference.** The eCeLLM framework from ICML 2024 demonstrates that fine-tuned 7B models outperform GPT-4 by 10.7% on e-commerce tasks, making this approach highly compelling for CommerceIQ.

Key resources that will accelerate your demo: the ECInstruct dataset (116K multi-task examples), ExtractGPT's production-ready prompts for attribute extraction, and pre-built HuggingFace models like `NingLab/eCeLLM` that can serve as baselines or starting points.

## Best datasets for e-commerce LLM training

**MAVE (Google Research)** stands out as the premier attribute extraction dataset with **3 million attribute-value annotations** across 2.2 million Amazon products, 1,257 categories, and 705 unique attributes. The dataset provides span-level annotations in JSON Lines format, making it ideal for training extraction models. Download from: https://github.com/google-research-datasets/MAVE

**Amazon ESCI** offers the best query-product relevance data with **2.6 million judgments** across 130K queries and 1.66M products in three languages. Labels include Exact, Substitute, Complement, and Irrelevant—perfect for search ranking tasks. Available at: https://github.com/amazon-science/esci-data (Apache 2.0 license).

For product Q&A, **AmazonQA** provides **923K questions with 3.6M answers** linked to 14M reviews, including answerability labels. Access at: https://github.com/amazonqa/amazonqa. The **Contextual Product QA** dataset from Amazon Science adds multi-source Q&A from 6 heterogeneous sources: https://github.com/amazon-science/contextual-product-qa

| Dataset | Size | Primary Task | Download |
|---------|------|--------------|----------|
| MAVE | 3M annotations | Attribute extraction | github.com/google-research-datasets/MAVE |
| Amazon ESCI | 2.6M judgments | Search relevance | github.com/amazon-science/esci-data |
| AmazonQA | 923K questions | Product Q&A | github.com/amazonqa/amazonqa |
| ECInstruct | 116K samples | Multi-task e-commerce | huggingface.co/datasets/NingLab/ECInstruct |
| WDC Products | 98M offers | Product matching | webdatacommons.org/largescaleproductcorpus |

**For hierarchical classification**, Google Product Taxonomy provides 5,595+ categories with full paths (e.g., "Electronics > Audio > Headphones"). The Amazon Review Data (2018) from UCSD contains **15.5 million products** with metadata including hierarchical categories—access via https://nijianmo.github.io/amazon/

## QLoRA fine-tuning configuration for 7B models

**QLoRA enables fine-tuning Mistral-7B or LLaMA-3-8B on a single 24GB GPU** in approximately 2-3 hours for 50K examples. The configuration below balances quality and efficiency:

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig
import torch

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Saves ~3GB additional
)

# LoRA configuration - apply to ALL linear layers
lora_config = LoraConfig(
    r=32,              # Rank (16-32 for most tasks, 64 for complex)
    lora_alpha=64,     # Alpha = 2×r works well
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**GPU memory requirements**: QLoRA reduces 7B fine-tuning from ~70GB (full fine-tune) to **8-12GB VRAM**. With gradient checkpointing, an RTX 3060 (12GB) can handle training. Training time scales as: RTX 4090 ~2-3 hours, A100 ~1.5 hours, T4 (Colab) ~4-6 hours for 50K examples.

For multi-task training combining classification, extraction, and Q&A, research shows **dramatic improvements**: classification accuracy jumps from 0.67 to 0.96, and NER accuracy from 0.71 to 0.98 when using multi-task datasets. Structure your data with task-specific prefixes:

```json
{"instruction": "[CLASSIFY] Categorize this product", "input": "Nike Air Max 90...", "output": "Footwear > Athletic"}
{"instruction": "[EXTRACT] Extract attributes as JSON", "input": "Bounty 8-pack...", "output": "{\"brand\": \"Bounty\", \"quantity\": \"8\"}"}
{"instruction": "[QA] Answer about this product", "input": "Is this waterproof?...", "output": "Yes, rated IPX7..."}
```

## Implementing the three core tasks

### Product classification with thousands of categories

**For hierarchical classification with 1000+ categories**, a two-stage hybrid approach works best at scale. Mercari's production system classifies 3 billion items by: (1) using LLM to label a sample of millions, (2) storing embeddings in a vector database, (3) using kNN to classify remaining items. This reduces costs from ~$1M to manageable levels.

For your demo, leverage **eCeLLM** from ICML 2024, which achieved 10.7% improvement over GPT-4:
- GitHub: https://github.com/ninglab/eCeLLM  
- HuggingFace: https://huggingface.co/NingLab

Research finding: flat classification often performs equally well or better than hierarchical approaches when using LLMs with few-shot examples. GPT-4 with ~5 similar examples achieves best performance on 370+ classes.

### Attribute extraction from product text

**ExtractGPT** provides production-tested prompts achieving 85% F1-score:

```python
system_prompt = """You are an algorithm for extracting product attributes in JSON format."""

user_prompt = """Extract attributes from: "Dr. Brown's Infant Toothbrush Set, 1.4 Ounce, Blue"
Valid attributes: Brand, Color, Size, Quantity
Output as JSON. Use 'n/a' for missing attributes."""

# Expected output: {"Brand": "Dr. Brown's", "Color": "Blue", "Size": "1.4 Ounce", "Quantity": "n/a"}
```

The MAVE dataset provides span-level annotations essential for training extraction models. Each record includes character offsets (`begin`, `end`) for extracted values, enabling sequence labeling approaches. Code: https://github.com/wbsg-uni-mannheim/ExtractGPT

### E-commerce Q&A systems

Implement RAG (Retrieval-Augmented Generation) for product Q&A:

1. **Index product data** in a vector database (ChromaDB for demo simplicity)
2. **Retrieve relevant context** when questions arrive (product specs, reviews, FAQs)
3. **Generate answers** grounded in retrieved information

Graph-enhanced RAG achieves 23% improvement in factual accuracy and 89% user satisfaction by combining knowledge graph subgraphs with document retrieval.

## System architecture and production deployment

### Recommended architecture for demo

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

**vLLM deployment** (fastest path to demo):
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model=mistralai/Mistral-7B-Instruct-v0.3 \
    --dtype=half \
    --max-model-len=4096 \
    --gpu-memory-utilization=0.85
```

This creates an OpenAI-compatible API endpoint. Wrap with FastAPI for custom e-commerce logic:

```python
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

@app.post("/extract-attributes")
async def extract(product_title: str):
    response = client.chat.completions.create(
        model="mistral-7b",
        messages=[{"role": "user", "content": f"[EXTRACT] {product_title}"}]
    )
    return {"attributes": response.choices[0].message.content}
```

### Evaluation metrics to track

| Task | Primary Metrics |
|------|-----------------|
| Classification | Weighted F1, Hierarchical F1, Top-k accuracy |
| Attribute extraction | Exact match, Token F1, Slot accuracy |
| Q&A | ROUGE-L, BERTScore, Human relevance ratings |

For hierarchical classification, use **HiClass** (scikit-learn compatible): https://github.com/scikit-learn-contrib/hiclass

### Inference optimization

For production, **quantization is essential**. Priority order: FP8 (best quality-speed tradeoff on modern GPUs) → INT8 SmoothQuant → AWQ 4-bit (memory-constrained). Pre-quantized models like `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` can run inference without fine-tuning overhead.

vLLM's PagedAttention and continuous batching provide **2-4x throughput** over naive implementations. Enable prefix caching for repeated prompts (common in e-commerce with template-based queries).

## Two-day implementation roadmap

### Day 1: Data preparation and training

**Morning (4 hours)**:
- Download MAVE dataset and ECInstruct from HuggingFace
- Format 2,000 examples (500 classification, 1,000 extraction, 500 Q&A)
- Set up training environment: `pip install torch transformers peft trl bitsandbytes`

**Afternoon (4 hours)**:
- Launch QLoRA training using SFTTrainer
- Expected training time: 2-3 hours on RTX 4090 / 4-5 hours on T4
- Prepare demo product catalog (100 products with attributes)

### Day 2: Integration and demo

**Morning (4 hours)**:
- Deploy fine-tuned model with vLLM
- Build FastAPI endpoints for three tasks
- Set up ChromaDB for RAG-based Q&A

**Afternoon (4 hours)**:
- Create simple Gradio/Streamlit demo interface
- Prepare example queries demonstrating each capability
- Document evaluation metrics and results

## Critical resources and repositories

**Essential GitHub repositories**:
- eCeLLM (multi-task e-commerce LLM): https://github.com/ninglab/eCeLLM
- ExtractGPT (attribute extraction): https://github.com/wbsg-uni-mannheim/ExtractGPT
- LLaMA-Factory (easy fine-tuning UI): https://github.com/hiyouga/LLaMA-Factory
- MAVE dataset: https://github.com/google-research-datasets/MAVE

**Key papers**:
- eCeLLM: Large Language Models for E-commerce (ICML 2024)
- MAVE: A Product Dataset for Multi-source Attribute Value Extraction (WSDM 2022)
- e-Llama: Domain Adaptation of Foundation LLMs for e-Commerce (ACL 2025)

**Common pitfalls to avoid**: Don't overtrain—3 epochs is usually sufficient for QLoRA; always include ~10% general instruction data to prevent catastrophic forgetting; use temperature=0 during evaluation for consistent results; validate JSON outputs with try-catch parsing since LLMs can produce malformed JSON.

## Conclusion

The e-commerce LLM landscape has matured significantly, with **publicly available datasets covering millions of products** and fine-tuning techniques that enable consumer GPUs to train competitive models. For a CommerceIQ demo, the combination of MAVE/ECInstruct data, QLoRA fine-tuning, and vLLM deployment provides the fastest path to a working system demonstrating real business value.

The key insight from recent research: domain-adapted smaller models (7B-8B parameters) consistently outperform generic large models like GPT-4 on e-commerce tasks when properly fine-tuned. This makes the investment in custom training worthwhile even for a demo—you'll show both technical capability and understanding of when custom solutions beat API calls.