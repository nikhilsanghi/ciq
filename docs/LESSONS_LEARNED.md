# Lessons Learned: E-Commerce LLM Fine-tuning (v1 Attempt)

**Date:** January 27, 2026
**Duration:** ~8 hours
**Outcome:** Failed - Model produces garbage output

---

## What We Tried

1. Fine-tuned Mistral-7B-Instruct-v0.3 using QLoRA on EC2 g5.xlarge
2. Used ECInstruct dataset (50K samples, 1 epoch)
3. Training metrics looked good: loss=0.61, accuracy=92.3%
4. Merged LoRA adapters with base model
5. Deployed with vLLM + Streamlit demo

---

## What Went Wrong

### 1. Dataset Mismatch (Critical)
**The #1 mistake that wasted 5 hours of training.**

We chose ECInstruct thinking it had classification/extraction/Q&A tasks. It actually contains:
- Query-product relevance ranking
- Product similarity comparison
- Purchase prediction
- Document yes/no Q&A

But our demo expected:
- Product classification into Google Taxonomy
- Attribute extraction as JSON
- Product Q&A with natural answers

**Lesson:** ALWAYS run `head -5 training_data.jsonl` and inspect the actual format BEFORE training.

### 2. No Output Validation
We saw good metrics (92% accuracy) and assumed success. Never tested actual model outputs until deployment.

**Lesson:** Test actual model outputs during training, not just loss/accuracy metrics.

### 3. Tokenizer Mismatch
The merge process failed to copy tokenizer files. We manually copied from base model, which may have caused issues.

**Lesson:** Validate tokenizer compatibility. Check that `tokenizer_config.json` has correct `tokenizer_class`.

### 4. Insufficient Training
Only 1 epoch on 50K samples. May have caused instability or catastrophic forgetting.

**Lesson:** Use at least 2-3 epochs. Include ~10% general instruction data to prevent forgetting.

### 5. No Intermediate Testing
We never tested:
- The LoRA adapter before merging
- The merged model before deployment
- Output quality at any checkpoint

**Lesson:** Test at every stage: after training, before merge, after merge.

---

## The Broken Output

Instead of answering questions, the model produced:
```
onomyonomy [CLASSIFY] Classify the the following product into the the Taxonomy.
onomy [CLASSIFY] Classify theify theify [ifyIFYIFYify Product Taxify]
```

This is **degenerate output** - repetitive nonsense indicating the model is fundamentally broken, not just misformatted.

---

## Checklist for Next Attempt

### Before Training
- [ ] Download dataset
- [ ] Run `head -20` to inspect actual format
- [ ] Verify task types match your goals
- [ ] Test prompt format on base model first
- [ ] Create train/val split

### During Training
- [ ] Use validation set
- [ ] Log sample outputs every N steps
- [ ] Train for 2-3 epochs minimum
- [ ] Include 10% general instruction data

### After Training
- [ ] Test LoRA adapter (before merge) with sample prompts
- [ ] Validate tokenizer files exist and are correct
- [ ] Test merged model with sample prompts
- [ ] Compare outputs to base model

### Deployment
- [ ] Test on exact same prompts as training format
- [ ] Verify model produces coherent text
- [ ] Only then update demo UI

---

## Datasets to Use Instead

| Task | Dataset | Why |
|------|---------|-----|
| Classification | Google Product Taxonomy | Actual product categories |
| Extraction | MAVE | 3M attribute annotations |
| Q&A | AmazonQA | 923K real product Q&A pairs |

---

## Time Breakdown

| Activity | Time | Outcome |
|----------|------|---------|
| Setup EC2 + environment | 1 hr | Success |
| Download ECInstruct | 30 min | Success |
| Training | 3.5 hr | Success (metrics) |
| Merge + Deploy | 1 hr | Partial |
| Debug broken model | 2 hr | Failed |
| **Total** | **8 hr** | **Wasted** |

---

## Key Quote

> "92% accuracy means nothing if you trained on the wrong task."

---

## Next Steps

1. Test base Mistral-7B-Instruct on target tasks (no fine-tuning)
2. Download correct datasets (MAVE, AmazonQA, Taxonomy)
3. Inspect data format thoroughly
4. Test prompts on base model
5. Train with validation and output monitoring
6. Test at every checkpoint
