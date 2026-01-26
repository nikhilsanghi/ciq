# The E-Commerce LLM Project: A Learning Journey

> *"Teaching an AI to be an e-commerce expert is like training a new retail employee who has read every product manual ever written, but has never actually worked in a store."*

Welcome to your deep dive into building a production-ready e-commerce AI system. This isn't your typical technical documentation—think of it as a conversation with a senior engineer who's been through the trenches and wants to share what actually matters.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Architecture: How the Pieces Fit Together](#2-architecture-how-the-pieces-fit-together)
3. [The QLoRA Magic Trick](#3-the-qlora-magic-trick)
4. [The Three E-Commerce Tasks](#4-the-three-e-commerce-tasks)
5. [The Data Story](#5-the-data-story)
6. [Critical Lessons & Pitfalls](#6-critical-lessons--pitfalls)
7. [How Good Engineers Think](#7-how-good-engineers-think)
8. [Technologies Deep Dive](#8-technologies-deep-dive)
9. [Interview Gold](#9-interview-gold)
10. [What's Next](#10-whats-next)

---

## 1. The Big Picture

### What We Built

We built an AI system that can do three things really well for e-commerce:

1. **Classify products** into categories (out of 5,595+ Google Product Taxonomy categories!)
2. **Extract attributes** from product descriptions (like "color: blue, size: medium")
3. **Answer questions** about products (like a knowledgeable sales associate)

### Why This Matters

Here's the thing about e-commerce: there are *millions* of products, and humans can't manually categorize them all. A single misclassified product means:
- Lost sales (customer can't find it)
- Wasted ad spend (showing shoes to people searching for phones)
- Poor search results (garbage in, garbage out)

Companies like Amazon, Walmart, and Shopify spend enormous resources on this problem. We built a system that tackles it with a fine-tuned LLM.

### The Journey in One Paragraph

We took a pre-trained language model (Mistral-7B), taught it e-commerce expertise using 116,000 training examples, deployed it with a high-performance inference server (vLLM), and wrapped it in a clean API. The cool part? We did this on consumer-grade hardware by using a clever technique called QLoRA.

---

## 2. Architecture: How the Pieces Fit Together

### The Restaurant Kitchen Analogy

Imagine our system is a high-end restaurant:

```
Customer (User Request)
        ↓
    🧑‍🍳 Waiter (FastAPI)
    Takes the order, knows the menu
        ↓
    👨‍🍳 Kitchen Manager (vLLM)
    Coordinates multiple orders efficiently
        ↓
    🍳 The Chef (Fine-tuned Mistral-7B)
    Actually prepares the food
        ↓
    📚 Recipe Book (ChromaDB)
    Reference material for complex dishes
```

Let me break this down:

### FastAPI: The Waiter

The waiter doesn't cook, but they're essential. They:
- Take orders in a format humans understand (JSON requests)
- Know what's on the menu (validate inputs)
- Handle multiple tables (async requests)
- Deliver food to the right table (return responses)

```
POST /classify  →  "What category is this product?"
POST /extract   →  "What attributes does this have?"
POST /qa        →  "Answer this question about the product"
```

### vLLM: The Kitchen Manager

Here's where it gets interesting. A naive approach would be: one chef, one dish at a time. But that's incredibly slow.

vLLM is like having a genius kitchen manager who:
- **Batches orders** - "Hey chef, cook 5 steaks together, not one by one"
- **Manages workspace efficiently** - Doesn't let half-finished dishes hog counter space (PagedAttention)
- **Prioritizes intelligently** - Quick appetizers don't wait for slow-cooking entrees (continuous batching)

**The result?** 24x faster than the naive approach. Not a typo. Twenty-four times.

### The Chef: Our Fine-Tuned Model

The chef (Mistral-7B) came to us already knowing how to cook (pre-trained on internet text). Our job was to teach them *our specific cuisine* (e-commerce tasks).

We didn't rebuild the chef from scratch—that would be insanely expensive. Instead, we gave them specialized training through QLoRA (more on this magic later).

### ChromaDB: The Recipe Book

Sometimes the chef needs to look something up. For our Q&A task, we use RAG (Retrieval-Augmented Generation):

1. Customer asks: "Is this laptop waterproof?"
2. We search our recipe book (vector database) for relevant info
3. We give the chef context: "Here's what the product specs say..."
4. Chef gives an informed answer

Without RAG, the chef might hallucinate. With RAG, they have grounded facts.

### The Request Flow (What Actually Happens)

```
1. User sends: POST /classify {"product_title": "Nike Air Max 90 Running Shoes"}

2. FastAPI receives request, validates it

3. Builds prompt: "[CLASSIFY] Categorize this product into Google
   Product Taxonomy categories. Product: Nike Air Max 90 Running Shoes"

4. Sends to vLLM server (OpenAI-compatible API)

5. vLLM batches with other requests, sends to model

6. Model returns: "Apparel & Accessories > Shoes > Athletic Shoes"

7. FastAPI parses response, returns: {"category": "Apparel & Accessories > Shoes > Athletic Shoes"}
```

Total time: ~200ms. Pretty good for understanding a product.

---

## 3. The QLoRA Magic Trick

### The Problem: You Can't Afford Full Fine-Tuning

Let me paint you a picture.

Mistral-7B has 7 billion parameters. Each parameter is a number. In full precision (FP32), each number takes 4 bytes. So:

```
7,000,000,000 parameters × 4 bytes = 28 GB just to LOAD the model
```

But wait, we need to train it! Training requires:
- Gradients (another 28 GB)
- Optimizer states (56 GB for Adam)
- Activations (varies, but ~30 GB)

**Total: ~140 GB of VRAM**

The best consumer GPU (RTX 4090) has 24 GB. Even cloud A100s have only 80 GB. Houston, we have a problem.

### The LoRA Insight: Don't Renovate the Mansion, Add a Shed

Here's the key insight that won a Best Paper award:

> *"Pre-trained models have low intrinsic dimensionality. You can achieve the same adaptation by training much smaller matrices."*

Translation: **You don't need to change all 7 billion parameters. You can get 95% of the benefit by changing 0.1% of them.**

How? Instead of updating the original weight matrix W, we add two small matrices:

```
Original: y = Wx
LoRA:     y = Wx + BAx

Where:
- W is frozen (7B params, don't touch)
- B is tiny (rank 32 × hidden size)
- A is tiny (hidden size × rank 32)
```

It's like... imagine renovating a mansion. Full fine-tuning is gutting every room. LoRA is adding a small extension that routes through the existing structure.

### The QLoRA Twist: Compress the Mansion Too

LoRA helps with training memory, but we still need to *load* the 28 GB model.

QLoRA says: "What if we compress the model while loading it?"

```
FP32: 4 bytes per parameter → 28 GB
FP16: 2 bytes per parameter → 14 GB
INT8: 1 byte per parameter  → 7 GB
NF4:  0.5 bytes per parameter → 3.5 GB
```

We use NF4 (Normal Float 4-bit). Why "normal float"? Because neural network weights follow a normal distribution, and NF4 is optimized for that distribution. It's better than regular INT4.

### The Final Math

```
QLoRA Memory Budget:
- Base model (4-bit):        ~3.5 GB
- LoRA adapters:             ~0.5 GB
- Optimizer (8-bit AdamW):   ~0.5 GB
- Gradients:                 ~0.5 GB
- Activations (checkpointed): ~2 GB
─────────────────────────────────────
Total:                       ~7-8 GB ✓
```

From 140 GB to 8 GB. That's the magic trick.

### Our Specific Configuration

```python
# In src/training/config.py

quantization = {
    "load_in_4bit": True,              # Shrink the mansion
    "bnb_4bit_quant_type": "nf4",      # Use the good compression
    "bnb_4bit_compute_dtype": "bf16",  # Math in bfloat16
    "bnb_4bit_use_double_quant": True  # Extra savings
}

lora = {
    "r": 32,                # Rank of adapter matrices
    "lora_alpha": 64,       # Scaling (usually 2x rank)
    "target_modules": [     # Where to add adapters
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # FFN
    ]
}
```

---

## 4. The Three E-Commerce Tasks

We teach our model three distinct skills, each with a special prefix:

### [CLASSIFY]: The Department Expert

**Analogy**: Training someone to know every aisle in a superstore.

```
Input:  "Apple MacBook Pro 14-inch M3 Pro chip 18GB RAM"
Output: "Electronics > Computers > Laptops"
```

The challenge? There are 5,595+ categories in Google's Product Taxonomy. And they're hierarchical—not just "Electronics" but "Electronics > Computers > Laptops."

We use **Hierarchical F1** scoring, which gives partial credit:
- If truth is "A > B > C" and we predict "A > B > D", we get 2/3 credit
- Better than binary right/wrong for 5-level hierarchies

### [EXTRACT]: The Label Reader

**Analogy**: Teaching someone to systematically read every product label and extract key facts.

```
Input:  "Sony WH-1000XM5 Wireless Noise Canceling Headphones, Black,
         30-hour battery, USB-C charging"
Output: {
    "brand": "Sony",
    "model": "WH-1000XM5",
    "type": "Wireless Noise Canceling Headphones",
    "color": "Black",
    "battery_life": "30 hours",
    "charging": "USB-C"
}
```

**The tricky part**: LLMs sometimes produce malformed JSON. We have extensive fallback parsing:

```python
# In src/evaluation/metrics.py

def parse_json_output(text: str) -> dict:
    # Try 1: Direct JSON parse
    # Try 2: Extract from markdown code blocks
    # Try 3: Regex extraction
    # Try 4: Fix common issues (single quotes, trailing commas)
    # Try 5: Give up gracefully
```

### [QA]: The Sales Associate

**Analogy**: A knowledgeable employee who can answer customer questions.

```
Input:  Product: "Patagonia Better Sweater Fleece Jacket"
        Question: "Is this warm enough for skiing?"

Output: "The Better Sweater is a mid-weight fleece rated for 40-60°F.
         It works great as a layering piece under a ski shell, but
         isn't warm enough as your only jacket for skiing in cold
         conditions."
```

This task uses RAG (Retrieval-Augmented Generation):
1. Search ChromaDB for relevant product info
2. Include retrieved context in the prompt
3. Generate a grounded answer

### Why Task Prefixes Matter

You might wonder: why `[CLASSIFY]`, `[EXTRACT]`, `[QA]`?

It's **multi-task learning**. One model, three skills. The prefix tells the model which "mode" to operate in.

Without prefixes, the model might classify when asked to extract. Prefixes are like saying "Put on your classification hat" vs "Put on your extraction hat."

The research shows 2-5% improvement from clear task prefixes.

---

## 5. The Data Story

### Our Training Curriculum: ECInstruct

We use the ECInstruct dataset: 116,000 e-commerce instruction examples.

```python
from datasets import load_dataset
dataset = load_dataset("NingLab/ECInstruct")
```

Each example looks like:
```json
{
    "instruction": "[CLASSIFY] Categorize this product...",
    "input": "Title: Nike Air Max 90\nDescription: Classic running shoe...",
    "output": "Apparel & Accessories > Shoes > Athletic Shoes"
}
```

### The Catastrophic Forgetting Problem

Here's something that caught me off guard the first time I saw it.

When you fine-tune a model intensively on one domain, it can *forget* everything else. It's called **catastrophic forgetting**.

**Analogy**: Imagine a medical student who studies cardiology so intensely that they forget how to make coffee, hold a conversation, or tie their shoes. That's catastrophic forgetting.

After aggressive e-commerce fine-tuning, your model might:
- Forget how to have a normal conversation
- Lose general knowledge
- Become weirdly stilted in responses

### The 10% Solution

The fix is elegant: **mix in 10% general instruction data**.

```python
# In src/data/preprocess.py

def mix_datasets(ecommerce_data, general_data, general_ratio=0.1):
    """
    90% e-commerce + 10% general = no forgetting
    """
```

We use the Alpaca dataset (52K general instruction examples) as our "don't forget how to be human" data.

Think of it as: "Study cardiology for 9 hours, then spend 1 hour on general topics." The medical student stays a functional human.

### Data Format: Instruction Tuning

We format everything for instruction tuning:

```
<s>[INST] [CLASSIFY] Categorize this product into Google Product
Taxonomy categories.

Product: Nike Air Max 90 Running Shoes for Men, White/Black [/INST]
Apparel & Accessories > Shoes > Athletic Shoes</s>
```

This format (Mistral's format) trains the model to:
1. Read instructions
2. Process input
3. Generate appropriate output

---

## 6. Critical Lessons & Pitfalls

Let me share the gotchas that cost engineers hours of debugging.

### Pitfall #1: The JSON Parsing Nightmare

**The Problem**: You ask the model to output JSON, and it outputs:

```
Here's the extracted information:

```json
{"brand": "Nike", "color": "Blue"}
```

Let me know if you need anything else!
```

That's not valid JSON. That's a conversational response containing JSON.

**The Solution**: Robust parsing with multiple fallbacks.

```python
def parse_json_output(text: str) -> dict:
    # Strip markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]

    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass

    # Fix common issues (single quotes, trailing commas)
    fixed = text.replace("'", '"')
    fixed = re.sub(r',\s*}', '}', fixed)
    # ... more fixes
```

**Lesson**: Never trust LLM output format. Always have fallbacks.

### Pitfall #2: Padding Side Matters (A Lot)

**The Problem**: Your model works in training but generates garbage during inference.

**The Cause**: Padding side.

```python
# WRONG for training
tokenizer.padding_side = "left"  # Used for generation
# Batch: ["<pad><pad>Hello", "<pad>Hi there"]

# CORRECT for training
tokenizer.padding_side = "right"  # Used for training
# Batch: ["Hello<pad><pad>", "Hi there<pad>"]
```

Why? In causal language models, the model predicts left-to-right. If you pad on the left during training, the model learns weird patterns.

**But**: For generation, you want left padding so all sequences start at position 0.

```python
# In src/training/trainer.py
tokenizer.padding_side = "right"  # Training

# In src/inference/model.py
tokenizer.padding_side = "left"   # Generation
```

### Pitfall #3: Temperature = 0 for Evaluation

**The Problem**: Your evaluation metrics are inconsistent. Same input, different outputs.

**The Cause**: Temperature > 0 adds randomness.

```python
# WRONG for evaluation
response = model.generate(temperature=0.7)  # Random!

# CORRECT for evaluation
response = model.generate(temperature=0.0)  # Deterministic
```

**When to use temperature > 0**: Creative tasks, diversity in outputs.
**When to use temperature = 0**: Evaluation, production consistency.

### Pitfall #4: Spot Instances Die (Save Checkpoints!)

**The Problem**: You're 80% through training on a spot instance. AWS terminates it. You lose everything.

**The Solution**: Save checkpoints frequently.

```python
training_args = TrainingArguments(
    save_steps=500,           # Save every 500 steps
    save_total_limit=3,       # Keep last 3 checkpoints
    resume_from_checkpoint=True,  # Resume if interrupted
)
```

**Real talk**: I've lost 12-hour training runs to spot termination. Now I checkpoint religiously.

### Pitfall #5: Flash Attention is Free Speed

**The Problem**: Your training is slow.

**The Solution**: Enable Flash Attention 2.

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",  # 2-3x speedup!
)
```

**The catch**: Requires Ampere or newer GPU (RTX 30xx, A100, H100).

Flash Attention is a mathematically equivalent but computationally optimized attention implementation. Same results, much faster.

### Pitfall #6: Gradient Checkpointing Trade-offs

**The Problem**: Out of memory during training.

**The Solution**: Gradient checkpointing.

```python
model.gradient_checkpointing_enable()
```

**How it works**: Instead of storing all activations, recompute them during backprop.

**Trade-off**:
- Memory: -50%
- Compute: +30%

Worth it when memory-constrained. The extra compute is cheaper than a bigger GPU.

---

## 7. How Good Engineers Think

This project taught me patterns that separate good engineers from great ones.

### Pattern 1: Start Simple, Add Complexity Only When Needed

Bad approach:
> "Let me build a distributed training system with custom CUDA kernels from day one."

Good approach:
> "Let me get a single training run working on one GPU. Then optimize."

We started with:
1. Basic HuggingFace training
2. Added QLoRA when we hit memory limits
3. Added vLLM when we needed throughput
4. Added RAG when Q&A needed grounding

Each addition solved a real problem we actually encountered.

### Pattern 2: Measure Before Optimizing

> "Premature optimization is the root of all evil." — Donald Knuth

Before optimizing inference speed, we:
1. Measured baseline: 10 tokens/sec with HuggingFace
2. Identified bottleneck: single-request processing
3. Added vLLM: 240 tokens/sec

We didn't guess. We measured, identified, fixed.

### Pattern 3: Design for Failure

What if vLLM is down? Our API handles it gracefully:

```python
@app.post("/classify")
async def classify(request: ClassifyRequest):
    try:
        result = await call_vllm(prompt)
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=503,
            detail="Model server timeout - please retry"
        )
```

The user gets a clear error, not a cryptic stack trace.

### Pattern 4: The "10% Rule"

When specializing a model, keep 10% general capability. When writing code, spend 10% on error handling. When training, save checkpoints every 10% progress.

The 10% is a buffer against reality not matching expectations.

### Pattern 5: Make It Obvious, Not Clever

Compare:

```python
# Clever (bad)
r = lambda p: (x:=p.split('>')[-1],h(x) if '>' in p else f(x))[1]

# Obvious (good)
def route_prediction(prediction: str) -> str:
    """Route to hierarchy or flat classifier based on prediction format."""
    if ">" in prediction:
        return handle_hierarchical(prediction)
    return handle_flat(prediction)
```

Future you (and your teammates) will thank present you.

---

## 8. Technologies Deep Dive

Let me explain *why* we chose each technology, not just what it does.

### Mistral-7B vs GPT-4 vs LLaMA

| Model | Size | Cost | Control | Latency |
|-------|------|------|---------|---------|
| GPT-4 | Unknown | $$$$ | None | 2-3s |
| LLaMA-3-8B | 8B | $ | Full | 0.2s |
| **Mistral-7B** | 7B | $ | Full | 0.2s |

We chose Mistral-7B because:
1. **Open weights** - We can fine-tune it
2. **Instruction-tuned variant** - Works well out of box
3. **Efficient architecture** - Grouped query attention
4. **7B is the sweet spot** - Good quality, fits on one GPU

GPT-4 is great, but:
- Can't fine-tune (only prompt)
- Expensive at scale
- No privacy guarantees

### vLLM vs HuggingFace generate()

```
HuggingFace: 10 tokens/sec (1 request at a time)
vLLM:        240 tokens/sec (continuous batching)
```

Why the 24x difference?

**HuggingFace** processes one request completely before starting another. If you have 10 requests, you wait for all 10 sequentially.

**vLLM** uses:
- **Continuous batching**: Start new requests as others finish
- **PagedAttention**: Efficient memory management (like OS virtual memory)
- **Prefix caching**: Reuse common prompt prefixes

For production, vLLM is the obvious choice.

### ChromaDB vs Pinecone vs FAISS

| Database | Hosted | Scalability | Cost | Simplicity |
|----------|--------|-------------|------|------------|
| Pinecone | Yes | Massive | $$$$ | Easy |
| FAISS | No | Large | Free | Medium |
| **ChromaDB** | No | Medium | Free | Very Easy |

We chose ChromaDB because:
1. **Python-native** - `pip install chromadb` and go
2. **Persistent storage** - Survives restarts
3. **Good enough for demo scale** - ~1M documents
4. **Zero infrastructure** - No separate database server

For production with billions of vectors? Probably Pinecone or managed FAISS. For learning and demos? ChromaDB is perfect.

### FastAPI vs Flask vs Django

| Framework | Async | Auto-docs | Validation | Speed |
|-----------|-------|-----------|------------|-------|
| Flask | No* | No | No | Medium |
| Django | Partial | No | Django forms | Medium |
| **FastAPI** | Yes | Yes | Pydantic | Fast |

*Flask can do async with extensions, but it's not native.

We chose FastAPI because:
1. **Async-native** - Important for calling vLLM
2. **Auto-generated OpenAPI docs** - Free documentation
3. **Pydantic validation** - Request/response validation built-in
4. **Modern Python** - Type hints, dataclasses

Flask would work. FastAPI just works better for ML APIs.

---

## 9. Interview Gold

Here are the concepts that come up in interviews, with memorable explanations.

### "Explain QLoRA in one sentence"

> "QLoRA compresses a large model to 4-bit precision and adds tiny trainable adapters, reducing memory requirements by 10-20x while maintaining 95%+ quality."

### "When would you fine-tune vs use prompt engineering?"

**Fine-tune when:**
- Consistent output format matters (JSON, categories)
- High volume (fine-tuning amortizes cost over millions of requests)
- Domain-specific terminology (medical, legal, e-commerce)
- Privacy requirements (can't send data to OpenAI)

**Prompt engineering when:**
- Quick prototyping (test idea in an hour)
- Using closed models (GPT-4, Claude)
- Low volume, varied tasks
- Rapidly changing requirements

### "Why BERTScore over ROUGE for Q&A?"

ROUGE counts word overlap. BERTScore measures semantic similarity.

```
Reference: "The laptop has good battery life"
Generated: "This computer lasts long on a single charge"

ROUGE: Low (few word matches)
BERTScore: High (same meaning)
```

For Q&A where paraphrasing is acceptable, BERTScore is better.

### "How would you handle LLM hallucinations?"

1. **RAG** - Ground responses in retrieved facts
2. **Temperature = 0** - Reduce randomness
3. **Structured outputs** - Constrain to valid options
4. **Confidence scores** - Know when model is uncertain
5. **Human review** - For high-stakes decisions

### "Explain catastrophic forgetting"

> "When fine-tuning aggressively on one domain, the model forgets general capabilities. It's like a doctor who studies cardiology so intensely they forget how to have normal conversations. The fix is mixing in 10% general data during fine-tuning."

### "Why is inference different from training?"

| Aspect | Training | Inference |
|--------|----------|-----------|
| Gradients | Yes | No |
| Batch size | Fixed | Variable |
| Padding | Right | Left |
| Mode | model.train() | model.eval() |
| Dropout | Active | Disabled |

The model behaves differently in each mode.

---

## 10. What's Next

### Production Considerations

If you're taking this to production:

1. **Monitoring** - Track latency, error rates, model drift
2. **A/B Testing** - Compare fine-tuned vs base model
3. **Guardrails** - Prevent harmful outputs
4. **Caching** - Cache common queries
5. **Rate limiting** - Protect against abuse

### Scaling Strategies

As you grow:

| Scale | Strategy |
|-------|----------|
| 10 QPS | Single GPU, vLLM |
| 100 QPS | Multi-GPU tensor parallelism |
| 1000 QPS | Multiple replicas behind load balancer |
| 10000 QPS | Distilled smaller model + caching |

### Further Learning

**Papers to Read:**
1. LoRA (Hu et al., 2021) - The original paper
2. QLoRA (Dettmers et al., 2023) - Quantized version
3. vLLM (Kwon et al., 2023) - PagedAttention
4. RLHF (Ouyang et al., 2022) - Human preference alignment

**Courses:**
- Hugging Face's NLP Course (free)
- Stanford CS324: Large Language Models
- DeepLearning.AI's LLM Specialization

**Repos to Study:**
- `huggingface/transformers` - The foundation
- `huggingface/peft` - LoRA implementation
- `vllm-project/vllm` - High-performance inference

---

## Future Work

Here are experiments and improvements planned for future iterations:

### Model Comparison: LLaMA-3-8B vs Mistral-7B

The codebase already supports both models (check `configs/training_config.yaml` - LLaMA-3-8B is commented out). A proper comparison would:

1. **Train both models** on the same dataset with identical hyperparameters
2. **Benchmark on all three tasks:**
   - Classification: Weighted F1, Top-k accuracy
   - Extraction: Exact match, Token F1
   - Q&A: ROUGE-L, BERTScore
3. **Compare practical metrics:**
   - Training time on same hardware
   - Inference throughput (tokens/sec)
   - Memory usage during training and inference

**To run the comparison:**
```bash
# Train Mistral (current default)
python -m src.training.trainer --output_dir ./outputs/mistral-ecommerce

# Train LLaMA-3 (uncomment in config or use CLI)
python -m src.training.trainer \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir ./outputs/llama3-ecommerce

# Evaluate both
python -m src.evaluation.evaluate --model_path ./outputs/mistral-ecommerce
python -m src.evaluation.evaluate --model_path ./outputs/llama3-ecommerce
```

### Quantization Strategy Improvements

Current setup uses 4-bit NF4 for memory efficiency. Future experiments:

- **INT8 base model + FP16 trainables**: Better quality, needs 16+ GB VRAM
- **GPTQ vs AWQ vs GGUF**: Compare quantization methods for inference
- **FP8 on Hopper GPUs**: Best of both worlds (if you have H100 access)

### Production Hardening

- Add monitoring with Prometheus/Grafana
- Implement request batching in FastAPI
- Add model A/B testing infrastructure
- Set up automated retraining pipeline

---

## Final Thoughts

Building this system taught me that LLM engineering is less about magic and more about understanding constraints:

- **Memory constraints** → QLoRA
- **Latency constraints** → vLLM
- **Quality constraints** → Fine-tuning + RAG
- **Cost constraints** → Open models + efficient inference

The best engineers don't fight constraints. They embrace them and find creative solutions within them.

Good luck on your journey. The field is moving fast, but the fundamentals—understanding trade-offs, measuring before optimizing, designing for failure—those stay constant.

---

*Built with curiosity and lots of GPU hours. May your loss curves always decrease.*
