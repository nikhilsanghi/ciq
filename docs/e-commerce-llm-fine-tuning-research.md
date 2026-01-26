# E-Commerce LLM Fine-Tuning Best Practices: Comprehensive Research Report

## Executive Summary

This report synthesizes current research and best practices for fine-tuning Large Language Models (LLMs) for e-commerce applications, covering three primary tasks: product classification, attribute extraction, and product Q&A. The research draws from academic papers, industry implementations, and practical tutorials from 2024-2025.

---

## 1. Product Classification with Hierarchical Taxonomy

### 1.1 The Challenge

Product classification in e-commerce involves categorizing products into hierarchical taxonomies like the Google Product Taxonomy (5,595+ categories). The high diversity of products and granular hierarchy result in hundreds or thousands of possible categories, presenting significant challenges for both manual and automated approaches.

### 1.2 Approaches

#### Flat vs. Hierarchical Classification

| Configuration | Description | Pros | Cons |
|---------------|-------------|------|------|
| **Flat** | Predict leaf node directly | Simpler training | Harder with many classes |
| **Hierarchical** | Predict level-by-level (e.g., L2 first, then leaf) | Leverages taxonomy structure | Requires cascade design |
| **Few-shot** | Use demonstrations in prompt | Works with new categories | Higher latency |

#### Fine-Tuning vs. Prompt Engineering

Research comparing these approaches for taxonomy construction found:
- **Fine-tuning**: BERTScore F1: 70.91%, Cosine Similarity: 66.40%
- **Prompt Engineering**: BERTScore F1: 61.66%, Cosine Similarity: 60.34%

Fine-tuning yields higher accuracy and consistency but requires more resources.

#### Dual-Expert Classification Paradigm (Amazon Science)

A hybrid approach that:
1. Fine-tuned domain-specific expert recommends top-K candidate categories
2. LLM-based expert analyzes nuanced differences between candidates
3. Selects the most suitable target category

### 1.3 Best Practices

1. **Use hierarchical structure**: Predicting intermediate levels first reduces the effective number of classes at each step
2. **Combine approaches**: Domain-specific models for initial filtering + LLMs for final disambiguation
3. **Handle abbreviated descriptions**: Real-world product descriptions are often incomplete; train models to handle this
4. **Consider zero-shot capabilities**: For new categories, LLMs can classify without retraining

### 1.4 Evaluation Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Weighted F1** | Accounts for class imbalance | Standard multi-class |
| **Hierarchical F1** | Expands predictions to include parent concepts | Taxonomy-aware evaluation |
| **Top-K Accuracy** | Correct label in top K predictions | Recommendation scenarios |
| **Semantic F1** | Grants partial credit based on semantic similarity | Fine-grained evaluation |

**Hierarchical F1 Implementation**: Expand both predicted and gold standard labels to include parent concepts up to the root, then compute standard precision/recall/F1.

---

## 2. Attribute Extraction from Product Descriptions

### 2.1 Task Definition

Extract key-value pairs from product descriptions, such as:
- Input: "Apple iPhone 15 Pro Max 256GB Space Black Titanium"
- Output: `{"brand": "Apple", "model": "iPhone 15 Pro Max", "storage": "256GB", "color": "Space Black", "material": "Titanium"}`

### 2.2 Key Dataset: MAVE

**MAVE (Multi-source Attribute Value Extraction)**:
- 3 million attribute-value annotations
- 1,257 unique categories
- 2.2 million cleaned Amazon product profiles
- Largest product attribute extraction dataset by number of examples

**Access**: https://github.com/google-research-datasets/MAVE

### 2.3 Approaches

#### Traditional NER Models
- **Pros**: Computationally efficient, low resource requirements
- **Cons**: Lack flexibility for diverse product descriptions
- **Best for**: Stable taxonomies with clearly defined attribute sets

#### LLM-Based Extraction

**GPT-4 Performance on MAVE**:
- Outperforms traditional methods (SU-OpenTag, AVEQA, MAVEQA) by 10%
- Achieves F1-score of 91%
- Particularly strong at string wrangling and name expansion

**Few-Shot Learning**:
- F1 scores above 90% achievable with just 100 training examples per attribute
- F1 increases consistently with more training examples

#### Zero-Shot with HyperPAVE
- Multi-label zero-shot model for unseen attribute values
- Uses inductive link prediction + fine-tuned BERT encoder
- Significantly outperforms classification models and generative LLMs in zero-shot scenarios

### 2.4 Best Practices

1. **Validate JSON outputs**: LLMs can produce malformed JSON; always use try-catch
2. **Use structured prompts**: Include attribute definitions and examples
3. **Extract both explicit and implicit attributes**: Mistral-based models show strong contextual understanding
4. **Consider multi-source extraction**: Product title, description, and images can all provide attributes

### 2.5 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match** | Predicted value exactly matches ground truth |
| **Token F1** | F1 score at token level (handles partial matches) |
| **Slot Accuracy** | Percentage of attributes correctly extracted |
| **Attribute-level F1** | F1 computed per attribute type |

---

## 3. Product Q&A with RAG

### 3.1 Why RAG for E-Commerce?

The e-commerce domain is highly dynamic:
- Prices, availability, and product attributes change frequently
- Traditional LLMs trained on static datasets become outdated quickly
- RAG combines retrieval from current data with language generation

### 3.2 Key Dataset: AmazonQA

**AmazonQA**:
- 923K questions, 3.6M answers, 14M reviews
- 156K products
- Includes answerability annotations
- Two question types: "yesno" (boolean) and "descriptive"

**Access**: https://github.com/amazonqa/amazonqa

### 3.3 RAG Architecture Best Practices

#### Retrieval Layer

1. **Hybrid Search**: Combine BM25 (keyword) + Dense Retrieval (semantic)
   - Better coverage than either approach alone
   - Sentence transformers for dense retrieval

2. **Re-ranking**: Score search results to ensure top results are most relevant

3. **Knowledge Graph Integration**: Neo4j + LLM provides:
   - More relevant answers vs. vector search alone
   - Domain-specific, factual knowledge
   - Explainability of results
   - Role-based access control

#### Generation Layer

1. **Context Window Management**: Limit retrieved context to most relevant snippets
2. **Source Attribution**: Include references to source products/reviews
3. **Answer Synthesis**: Combine information from multiple reviews

### 3.4 Evaluation Metrics

| Metric | Description | Best For |
|--------|-------------|----------|
| **ROUGE-L** | Longest common subsequence overlap | Text summarization |
| **BERTScore** | Semantic similarity via BERT embeddings | Semantic accuracy |
| **Retrieval Recall@K** | % of relevant docs in top K | Retrieval quality |
| **Answer Relevance** | LLM-judged relevance score | End-to-end quality |

**BERTScore vs ROUGE-L**:
- BERTScore: 0.93 Pearson correlation with human judgment
- ROUGE: 0.78 Pearson correlation
- BERTScore captures paraphrasing; ROUGE requires exact matches

### 3.5 Production Considerations

1. **Guardrails**: Implement LLM monitoring for accuracy and compliance (see DoorDash example)
2. **Continuous Fine-Tuning**: Use production data to fine-tune smaller models (2B parameters)
3. **Fallback Mechanisms**: Handle unanswerable questions gracefully

---

## 4. Recommended Datasets

### 4.1 Primary Datasets

| Dataset | Size | Tasks | Source |
|---------|------|-------|--------|
| **MAVE** | 3M annotations, 2.2M products | Attribute extraction | google-research-datasets/MAVE |
| **ECInstruct** | 116K samples, 10 tasks | Multi-task instruction tuning | NingLab/ECInstruct |
| **Amazon ESCI** | 2.6M relevance judgments | Query-product matching | amazon-science/esci-data |
| **AmazonQA** | 923K questions | Product Q&A | amazonqa/amazonqa |

### 4.2 ECInstruct Details

The first open-source, large-scale benchmark for e-commerce instruction tuning:
- **4 Task Categories**: Product Understanding, User Understanding, Query-Product Matching, Product Q&A
- **10 Tasks**: Including classification, extraction, and generation
- **Evaluation**: Both in-domain (IND) and out-of-domain (OOD) test sets

**eCeLLM Results**: 10.7% improvement over best baselines including GPT-4 Turbo

### 4.3 Amazon ESCI Details

ESCI Relevance Labels:
- **Exact (E)**: Fully satisfies query
- **Substitute (S)**: Functional substitute
- **Complement (C)**: Complementary product
- **Irrelevant (I)**: Does not satisfy query

---

## 5. Fine-Tuning Best Practices

### 5.1 QLoRA Configuration for 7B Models

Recommended settings for 8-12GB VRAM:

```yaml
# Quantization
quantization: 4-bit NF4
double_quantization: true

# LoRA Parameters
lora_rank: 32
lora_alpha: 64
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Training
epochs: 3
learning_rate: 2e-4
batch_size: 4
gradient_accumulation_steps: 4
```

### 5.2 Preventing Catastrophic Forgetting

**Key Findings**:
- Forgetting intensifies as model scale increases (1B to 7B range)
- LoRA alone does NOT mitigate forgetting in continual learning
- Cannot be avoided through early stopping

**Prevention Techniques**:

| Technique | Description | Effectiveness |
|-----------|-------------|---------------|
| **Data Mixing** | Include ~10% general instruction data | High |
| **PEFT/Adapters** | Train small parameter sets while freezing base | High |
| **EWC (Elastic Weight Consolidation)** | Penalize changes to important weights | Medium |
| **Rehearsal/Replay** | Include samples from previous tasks | High |
| **Sharpness-Aware Minimization** | Flatten loss landscape | Medium |

**Best Practice**: Combine PEFT with a small amount of rehearsal data (~10% general instructions).

### 5.3 Training Data Preparation

1. **Task-Specific Prefixes**: Use `[CLASSIFY]`, `[EXTRACT]`, `[QA]` prefixes
2. **Structured Output Format**: Train with consistent JSON schemas
3. **Include Edge Cases**: Abbreviated descriptions, missing attributes
4. **Balance Categories**: Oversample rare categories or use weighted loss

### 5.4 Inference Optimization

| Technique | Memory Reduction | Speed Improvement |
|-----------|------------------|-------------------|
| **FP8 Quantization** | 50% | Minimal loss |
| **INT8 SmoothQuant** | 50% | Good |
| **AWQ 4-bit** | 75% | Moderate loss |
| **GGUF (CPU)** | 90% RAM | 18x throughput |

**Recommendation**: For production, prioritize FP8 > INT8 SmoothQuant > AWQ 4-bit

---

## 6. Small Specialized Models vs. Large LLMs

### 6.1 Performance Comparison

| Model Type | Example | Accuracy | Cost | Latency |
|------------|---------|----------|------|---------|
| **Small Specialized** | Fine-tuned Llama-3-8B | F1: 0.76-0.88 | Low | Fast |
| **Large LLM** | GPT-4, Claude | F1: 0.78-0.90 | High | Slow |
| **Specialized 1B** | Domain-tuned | 99% (intent) | Very Low | Very Fast |

**Key Finding**: Small language models (8B) achieve F1 within 0.02 of LLMs that are 100-300x larger.

### 6.2 When to Use Each

**Use Small Specialized Models When**:
- Task is well-defined and narrow
- Latency and cost are critical
- Privacy requires on-premise deployment
- High volume of requests

**Use Large LLMs When**:
- Zero-shot or few-shot scenarios
- Multistep reasoning required
- Broad domain knowledge needed
- Handling distribution shifts

### 6.3 Recommended Approach

**Hybrid Strategy**:
1. Small specialized model for initial filtering/classification
2. LLM for complex disambiguation or edge cases
3. Continuous fine-tuning of small models using LLM outputs

---

## 7. Common Pitfalls and Solutions

### 7.1 Classification Pitfalls

| Pitfall | Solution |
|---------|----------|
| Too many classes | Use hierarchical prediction |
| Class imbalance | Weighted loss, oversampling |
| Abbreviated descriptions | Train on realistic, messy data |
| New categories | Zero-shot LLM backup |

### 7.2 Extraction Pitfalls

| Pitfall | Solution |
|---------|----------|
| Malformed JSON output | Validate with try-catch, retry |
| Missing attributes | Train to output null/empty |
| Implicit attributes | Use context-aware models (Mistral) |
| Conflicting values | Prioritize by source reliability |

### 7.3 Q&A Pitfalls

| Pitfall | Solution |
|---------|----------|
| Hallucinations | RAG grounding, guardrails |
| Outdated information | Real-time retrieval |
| Unanswerable questions | Answerability classifier |
| Context overflow | Smart context selection, summarization |

---

## 8. Recommended Tools and Libraries

### 8.1 Training

- **Hugging Face Transformers + PEFT**: LoRA/QLoRA implementation
- **Axolotl**: Streamlined fine-tuning framework
- **TRL (Transformer Reinforcement Learning)**: SFT trainer
- **bitsandbytes**: 4-bit quantization

### 8.2 Inference

- **vLLM**: High-performance inference with PagedAttention
- **GGUF**: CPU-friendly quantized format
- **Text Generation Inference (TGI)**: Production-ready serving

### 8.3 RAG

- **ChromaDB**: Lightweight vector store
- **Neo4j**: Knowledge graph integration
- **LangChain/LlamaIndex**: RAG orchestration

### 8.4 Evaluation

- **evaluate (Hugging Face)**: ROUGE, BERTScore
- **evidently**: RAG evaluation framework
- **ragas**: RAG-specific metrics

---

## 9. Quick Start Recommendations

### For Beginners

1. Start with **ECInstruct** dataset and **eCeLLM** as baseline
2. Use **QLoRA** with Mistral-7B-Instruct for fine-tuning
3. Evaluate with **weighted F1** (classification), **token F1** (extraction), **ROUGE-L + BERTScore** (Q&A)
4. Include **10% general instruction data** to prevent forgetting

### For Production

1. Fine-tune **7-8B model** on domain-specific data
2. Deploy with **vLLM** and **FP8 quantization**
3. Implement **RAG** for product Q&A with hybrid search
4. Add **guardrails** for hallucination detection
5. Use **LLM cascade**: Small model first, large model for edge cases

---

## Sources

### Product Classification
- [LLMs for product classification in e-commerce - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2949719125000184)
- [E-commerce product categorization with LLM-based dual-expert classification paradigm - Amazon Science](https://www.amazon.science/publications/e-commerce-product-categorization-with-llm-based-dual-expert-classification-paradigm)
- [LLM-Based Robust Product Classification in Commerce and Compliance - arXiv](https://arxiv.org/abs/2408.05874)
- [Automated Taxonomy Construction Using LLMs - MDPI](https://www.mdpi.com/2673-4117/6/11/283)

### Attribute Extraction
- [MAVE Dataset - GitHub](https://github.com/google-research-datasets/MAVE)
- [MAVE: A Product Dataset for Multi-source Attribute Value Extraction - arXiv](https://arxiv.org/pdf/2112.08663)
- [Using LLMs for Extraction and Normalization of Product Attribute Values - arXiv](https://arxiv.org/pdf/2403.02130)
- [Multi-Label Zero-Shot Product Attribute-Value Extraction - arXiv](https://arxiv.org/html/2402.08802v1)

### ECInstruct and eCeLLM
- [eCeLLM Project Page](https://ninglab.github.io/eCeLLM/)
- [eCeLLM: Generalizing LLMs for E-commerce - arXiv](https://arxiv.org/abs/2402.08831)
- [EcomGPT: Instruction-tuning LLMs for E-commerce - arXiv](https://arxiv.org/abs/2308.06966)

### Amazon ESCI Dataset
- [Shopping Queries Dataset - GitHub](https://github.com/amazon-science/esci-data)
- [Shopping Queries Dataset Paper - arXiv](https://arxiv.org/abs/2206.06588)

### AmazonQA Dataset
- [AmazonQA - GitHub](https://github.com/amazonqa/amazonqa)
- [AmazonQA: A Review-Based Question Answering Task - arXiv](https://arxiv.org/abs/1908.04364)

### RAG for E-Commerce
- [RAG Evaluation Guide - Evidently AI](https://www.evidentlyai.com/llm-guide/rag-evaluation)
- [Advanced RAG Techniques - Neo4j](https://neo4j.com/blog/genai/advanced-rag-techniques/)
- [Contextually Aware E-Commerce Product QA using RAG - arXiv](https://arxiv.org/html/2508.01990v1)

### Fine-Tuning Tutorials
- [Fine-Tuning Mistral 7B with QLoRA Using Axolotl - MarkTechPost](https://www.marktechpost.com/2025/02/09/tutorial-to-fine-tuning-mistral-7b-with-qlora-using-axolotl-for-efficient-llm-training/)
- [LLM Fine-tuning Complete Guide 2025 - TensorBlue](https://tensorblue.com/blog/llm-fine-tuning-complete-guide-tutorial-2025)
- [From Quantization to Inference: QLoRA with Mistral 7B - Towards AI](https://towardsai.net/p/machine-learning/from-quantization-to-inference-beginners-guide-for-practical-fine-tuning-qlora-with-mistral-7b)

### Catastrophic Forgetting
- [Catastrophic Forgetting in LLMs During Continual Fine-tuning - arXiv](https://arxiv.org/abs/2308.08747)
- [Mitigating Catastrophic Forgetting in LLM Tuning - APXML](https://apxml.com/courses/fine-tuning-adapting-large-language-models/chapter-5-advanced-fine-tuning-strategies/mitigating-catastrophic-forgetting)

### Evaluation Metrics
- [BERTScore for LLM Evaluation - Comet](https://www.comet.com/site/blog/bertscore-for-llm-evaluation/)
- [Hierarchical Confusion Matrix - Oxford Academic](https://academic.oup.com/jrsssc/article/72/5/1394/7217007)
- [BERTScore Explained - Galileo AI](https://galileo.ai/blog/bert-score-explained-guide)

### Model Size Comparisons
- [Performance Trade-offs of Optimizing Small Language Models for E-Commerce - arXiv](https://arxiv.org/html/2510.21970v1)
- [SLM vs LLM: How to Pick the Right Model Size - Label Your Data](https://labelyourdata.com/articles/llm-fine-tuning/slm-vs-llm)
