"""
Evaluation metrics for e-commerce LLM tasks.

Includes classification, extraction, and Q&A metrics with support for:
- Hierarchical taxonomy classification
- JSON attribute extraction with partial credit
- Semantic similarity for Q&A evaluation
"""

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Classification Metrics
# =============================================================================


def compute_classification_metrics(
    predictions: List[str],
    references: List[str],
    average: str = "weighted",
    top_k_values: Optional[List[int]] = None,
    delimiter: str = " > ",
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics for product categorization.

    Args:
        predictions: List of predicted category strings
        references: List of ground truth category strings
        average: Averaging strategy for F1/precision/recall ('weighted', 'macro', 'micro')
        top_k_values: List of k values for top-k accuracy (if predictions contain multiple)
        delimiter: Delimiter used in hierarchical categories (e.g., "Apparel > Shoes > Athletic")

    Returns:
        Dictionary containing:
        - weighted_f1: F1 score with class weights (handles imbalance)
        - macro_f1: Unweighted average F1 across classes
        - precision: Precision score
        - recall: Recall score
        - accuracy: Exact match accuracy
        - hierarchical_f1: Partial credit for hierarchical matches
        - level_accuracy: Accuracy at each taxonomy level
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    if not predictions:
        logger.warning("Empty predictions list provided")
        return {
            "weighted_f1": 0.0,
            "macro_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "hierarchical_f1": 0.0,
        }

    # Normalize predictions and references
    predictions_norm = [_normalize_category(p) for p in predictions]
    references_norm = [_normalize_category(r) for r in references]

    # Get unique labels for sklearn metrics
    all_labels = list(set(predictions_norm + references_norm))

    metrics = {}

    # Standard classification metrics
    try:
        metrics["weighted_f1"] = f1_score(
            references_norm,
            predictions_norm,
            labels=all_labels,
            average="weighted",
            zero_division=0,
        )
        metrics["macro_f1"] = f1_score(
            references_norm,
            predictions_norm,
            labels=all_labels,
            average="macro",
            zero_division=0,
        )
        metrics["precision"] = precision_score(
            references_norm,
            predictions_norm,
            labels=all_labels,
            average=average,
            zero_division=0,
        )
        metrics["recall"] = recall_score(
            references_norm,
            predictions_norm,
            labels=all_labels,
            average=average,
            zero_division=0,
        )
        metrics["accuracy"] = accuracy_score(references_norm, predictions_norm)
    except Exception as e:
        logger.error(f"Error computing sklearn metrics: {e}")
        metrics.update(
            {
                "weighted_f1": 0.0,
                "macro_f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
            }
        )

    # Hierarchical F1 for taxonomy categories
    metrics["hierarchical_f1"] = hierarchical_f1(predictions, references, delimiter)

    # Level-wise accuracy for hierarchical categories
    level_acc = compute_level_accuracy(predictions, references, delimiter)
    metrics["level_accuracy"] = level_acc

    return metrics


def hierarchical_f1(
    predictions: List[str],
    references: List[str],
    delimiter: str = " > ",
) -> float:
    """
    Compute hierarchical F1 score with partial credit for partial category matches.

    For taxonomy categories like "Apparel > Shoes > Athletic", this gives partial
    credit when the prediction matches some levels of the hierarchy.

    Example:
        Reference: "Apparel > Shoes > Athletic"
        Prediction: "Apparel > Shoes > Running"
        -> Matches 2 out of 3 levels = 66.7% credit

    Args:
        predictions: List of predicted category strings
        references: List of ground truth category strings
        delimiter: Delimiter separating hierarchy levels

    Returns:
        Hierarchical F1 score (0.0 to 1.0)
    """
    if not predictions or not references:
        return 0.0

    total_precision = 0.0
    total_recall = 0.0
    valid_samples = 0

    for pred, ref in zip(predictions, references):
        pred_levels = _split_hierarchy(pred, delimiter)
        ref_levels = _split_hierarchy(ref, delimiter)

        if not ref_levels:
            continue

        valid_samples += 1

        # Count matching levels at each position
        matches = 0
        for i, (p_level, r_level) in enumerate(zip(pred_levels, ref_levels)):
            if _normalize_category(p_level) == _normalize_category(r_level):
                matches += 1
            else:
                # Stop at first mismatch for strict hierarchical matching
                break

        # Precision: matches / predicted levels
        if pred_levels:
            total_precision += matches / len(pred_levels)

        # Recall: matches / reference levels
        total_recall += matches / len(ref_levels)

    if valid_samples == 0:
        return 0.0

    avg_precision = total_precision / valid_samples
    avg_recall = total_recall / valid_samples

    # F1 score
    if avg_precision + avg_recall == 0:
        return 0.0

    return 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)


def compute_level_accuracy(
    predictions: List[str],
    references: List[str],
    delimiter: str = " > ",
    max_levels: int = 5,
) -> Dict[str, float]:
    """
    Compute accuracy at each level of the taxonomy hierarchy.

    Args:
        predictions: List of predicted category strings
        references: List of ground truth category strings
        delimiter: Delimiter separating hierarchy levels
        max_levels: Maximum number of levels to evaluate

    Returns:
        Dictionary with accuracy at each level (level_1, level_2, etc.)
    """
    level_correct = defaultdict(int)
    level_total = defaultdict(int)

    for pred, ref in zip(predictions, references):
        pred_levels = _split_hierarchy(pred, delimiter)
        ref_levels = _split_hierarchy(ref, delimiter)

        for i in range(min(len(ref_levels), max_levels)):
            level_total[i + 1] += 1
            if i < len(pred_levels):
                if _normalize_category(pred_levels[i]) == _normalize_category(
                    ref_levels[i]
                ):
                    level_correct[i + 1] += 1

    return {
        f"level_{i}": level_correct[i] / level_total[i] if level_total[i] > 0 else 0.0
        for i in range(1, max_levels + 1)
        if level_total[i] > 0
    }


def compute_top_k_accuracy(
    predictions: List[List[str]],
    references: List[str],
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """
    Compute top-k accuracy for classification tasks.

    Args:
        predictions: List of prediction lists (ranked by confidence)
        references: List of ground truth labels
        k_values: List of k values to compute accuracy for

    Returns:
        Dictionary with top-k accuracy for each k value
    """
    results = {}

    for k in k_values:
        correct = 0
        for preds, ref in zip(predictions, references):
            top_k_preds = preds[:k] if len(preds) >= k else preds
            top_k_normalized = [_normalize_category(p) for p in top_k_preds]
            if _normalize_category(ref) in top_k_normalized:
                correct += 1

        results[f"top_{k}_accuracy"] = correct / len(references) if references else 0.0

    return results


def _normalize_category(category: str) -> str:
    """Normalize category string for comparison."""
    if not category:
        return ""
    # Lowercase, strip whitespace, normalize multiple spaces
    normalized = category.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _split_hierarchy(category: str, delimiter: str = " > ") -> List[str]:
    """Split hierarchical category into levels."""
    if not category:
        return []
    return [level.strip() for level in category.split(delimiter) if level.strip()]


# =============================================================================
# Extraction Metrics
# =============================================================================


def compute_extraction_metrics(
    predictions: List[Union[dict, str]],
    references: List[Union[dict, str]],
) -> Dict[str, float]:
    """
    Compute extraction metrics for attribute-value extraction tasks.

    Args:
        predictions: List of predicted extractions (dict or JSON string)
        references: List of ground truth extractions (dict or JSON string)

    Returns:
        Dictionary containing:
        - exact_match: Proportion of predictions that exactly match reference
        - token_f1: Average token-level F1 score for values
        - slot_accuracy: Per-attribute accuracy
        - slot_precision: Precision for predicted attributes
        - slot_recall: Recall for reference attributes
        - value_accuracy: Accuracy of values for correctly predicted attributes
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    if not predictions:
        return {
            "exact_match": 0.0,
            "token_f1": 0.0,
            "slot_accuracy": 0.0,
            "slot_precision": 0.0,
            "slot_recall": 0.0,
            "value_accuracy": 0.0,
        }

    exact_matches = 0
    token_f1_scores = []
    slot_precisions = []
    slot_recalls = []
    value_accuracies = []

    for pred, ref in zip(predictions, references):
        # Parse JSON if needed
        pred_dict = parse_json_output(pred) if isinstance(pred, str) else pred
        ref_dict = parse_json_output(ref) if isinstance(ref, str) else ref

        # Handle parsing failures
        if pred_dict is None:
            pred_dict = {}
        if ref_dict is None:
            ref_dict = {}

        # Exact match
        if _dicts_equal(pred_dict, ref_dict):
            exact_matches += 1

        # Slot-level metrics
        slot_metrics = _compute_slot_metrics(pred_dict, ref_dict)
        slot_precisions.append(slot_metrics["precision"])
        slot_recalls.append(slot_metrics["recall"])
        value_accuracies.append(slot_metrics["value_accuracy"])

        # Token F1 for values
        token_f1 = _compute_value_token_f1(pred_dict, ref_dict)
        token_f1_scores.append(token_f1)

    n = len(predictions)
    return {
        "exact_match": exact_matches / n,
        "token_f1": np.mean(token_f1_scores) if token_f1_scores else 0.0,
        "slot_accuracy": (np.mean(slot_precisions) + np.mean(slot_recalls)) / 2,
        "slot_precision": np.mean(slot_precisions) if slot_precisions else 0.0,
        "slot_recall": np.mean(slot_recalls) if slot_recalls else 0.0,
        "value_accuracy": np.mean(value_accuracies) if value_accuracies else 0.0,
    }


def parse_json_output(text: str) -> Optional[dict]:
    """
    Safely parse LLM JSON output with fallback handling.

    Handles common LLM output issues:
    - Markdown code blocks (```json ... ```)
    - Leading/trailing text
    - Single quotes instead of double quotes
    - Trailing commas

    Args:
        text: Raw text output from LLM

    Returns:
        Parsed dictionary or None if parsing fails
    """
    if not text or not isinstance(text, str):
        return None

    # If already a dict, return as-is
    if isinstance(text, dict):
        return text

    original_text = text

    try:
        # Try direct parsing first
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract JSON from markdown code blocks
    code_block_patterns = [
        r"```json\s*\n?(.*?)\n?```",
        r"```\s*\n?(.*?)\n?```",
        r"`(.*?)`",
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                text = match.group(1).strip()
                break

    # Find JSON-like content with braces
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            text = brace_match.group()

    # Try fixing common issues
    fixes = [
        # Replace single quotes with double quotes
        (r"'([^']*)':", r'"\1":'),
        (r":\s*'([^']*)'", r': "\1"'),
        # Remove trailing commas
        (r",\s*}", "}"),
        (r",\s*]", "]"),
        # Fix unquoted keys
        (r"(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1 "\2":'),
    ]

    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.debug(f"Failed to parse JSON: {original_text[:100]}...")
        return None


def _dicts_equal(d1: dict, d2: dict) -> bool:
    """Check if two dictionaries are equal (case-insensitive values)."""
    if set(d1.keys()) != set(d2.keys()):
        return False

    for key in d1:
        v1 = str(d1[key]).lower().strip() if d1[key] is not None else ""
        v2 = str(d2[key]).lower().strip() if d2[key] is not None else ""
        if v1 != v2:
            return False

    return True


def _compute_slot_metrics(pred: dict, ref: dict) -> Dict[str, float]:
    """Compute slot-level precision, recall, and value accuracy."""
    pred_keys = set(pred.keys())
    ref_keys = set(ref.keys())

    if not ref_keys:
        return {"precision": 1.0 if not pred_keys else 0.0, "recall": 1.0, "value_accuracy": 1.0}

    # Precision: correctly predicted attributes / all predicted attributes
    correct_keys = pred_keys & ref_keys
    precision = len(correct_keys) / len(pred_keys) if pred_keys else 0.0

    # Recall: correctly predicted attributes / all reference attributes
    recall = len(correct_keys) / len(ref_keys)

    # Value accuracy: for matching keys, how many values are correct
    if correct_keys:
        correct_values = sum(
            1
            for k in correct_keys
            if str(pred.get(k, "")).lower().strip()
            == str(ref.get(k, "")).lower().strip()
        )
        value_accuracy = correct_values / len(correct_keys)
    else:
        value_accuracy = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "value_accuracy": value_accuracy,
    }


def _compute_value_token_f1(pred: dict, ref: dict) -> float:
    """Compute average token-level F1 for attribute values."""
    if not ref:
        return 1.0 if not pred else 0.0

    f1_scores = []

    for key in ref:
        ref_value = str(ref[key]).lower().strip() if ref[key] is not None else ""
        pred_value = (
            str(pred.get(key, "")).lower().strip() if pred.get(key) is not None else ""
        )

        ref_tokens = set(ref_value.split())
        pred_tokens = set(pred_value.split())

        if not ref_tokens:
            f1_scores.append(1.0 if not pred_tokens else 0.0)
            continue

        if not pred_tokens:
            f1_scores.append(0.0)
            continue

        common = ref_tokens & pred_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return np.mean(f1_scores) if f1_scores else 0.0


# =============================================================================
# Q&A Metrics
# =============================================================================


def compute_qa_metrics(
    predictions: List[str],
    references: List[str],
    use_bertscore: bool = True,
) -> Dict[str, float]:
    """
    Compute Q&A evaluation metrics.

    Args:
        predictions: List of predicted answers
        references: List of ground truth answers
        use_bertscore: Whether to compute BERTScore (requires additional dependencies)

    Returns:
        Dictionary containing:
        - rouge_1: ROUGE-1 F1 score (unigram overlap)
        - rouge_2: ROUGE-2 F1 score (bigram overlap)
        - rouge_l: ROUGE-L F1 score (longest common subsequence)
        - bleu: BLEU score
        - bertscore_precision: BERTScore precision (if enabled)
        - bertscore_recall: BERTScore recall (if enabled)
        - bertscore_f1: BERTScore F1 (if enabled)
        - exact_match: Exact string match rate
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    if not predictions:
        return {
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "rouge_l": 0.0,
            "bleu": 0.0,
            "exact_match": 0.0,
        }

    metrics = {}

    # Exact match
    exact_matches = sum(
        1
        for p, r in zip(predictions, references)
        if p.lower().strip() == r.lower().strip()
    )
    metrics["exact_match"] = exact_matches / len(predictions)

    # ROUGE scores
    try:
        rouge = evaluate.load("rouge")
        rouge_results = rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=True,
        )
        metrics["rouge_1"] = rouge_results.get("rouge1", 0.0)
        metrics["rouge_2"] = rouge_results.get("rouge2", 0.0)
        metrics["rouge_l"] = rouge_results.get("rougeL", 0.0)
    except Exception as e:
        logger.warning(f"Failed to compute ROUGE scores: {e}")
        # Fallback to manual ROUGE-L computation
        rouge_l_scores = [
            _compute_rouge_l(p, r) for p, r in zip(predictions, references)
        ]
        metrics["rouge_1"] = 0.0
        metrics["rouge_2"] = 0.0
        metrics["rouge_l"] = np.mean(rouge_l_scores) if rouge_l_scores else 0.0

    # BLEU score
    try:
        bleu = evaluate.load("bleu")
        # BLEU expects references as list of lists
        refs_formatted = [[r] for r in references]
        bleu_results = bleu.compute(
            predictions=predictions,
            references=refs_formatted,
        )
        metrics["bleu"] = bleu_results.get("bleu", 0.0)
    except Exception as e:
        logger.warning(f"Failed to compute BLEU score: {e}")
        metrics["bleu"] = 0.0

    # BERTScore for semantic similarity
    if use_bertscore:
        try:
            bertscore = evaluate.load("bertscore")
            bert_results = bertscore.compute(
                predictions=predictions,
                references=references,
                lang="en",
                model_type="microsoft/deberta-xlarge-mnli",
            )
            metrics["bertscore_precision"] = np.mean(bert_results["precision"])
            metrics["bertscore_recall"] = np.mean(bert_results["recall"])
            metrics["bertscore_f1"] = np.mean(bert_results["f1"])
        except Exception as e:
            logger.warning(f"Failed to compute BERTScore: {e}")
            # BERTScore not available, skip

    return metrics


def _compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L score (longest common subsequence) manually.

    Args:
        prediction: Predicted text
        reference: Reference text

    Returns:
        ROUGE-L F1 score
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    # Compute LCS length using dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    precision = lcs_length / m if m > 0 else 0.0
    recall = lcs_length / n if n > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# =============================================================================
# Aggregate Metrics
# =============================================================================


def compute_all_metrics(
    task: str,
    predictions: List[Any],
    references: List[Any],
    **kwargs,
) -> Dict[str, float]:
    """
    Compute all relevant metrics for a given task.

    Args:
        task: One of "classification", "extraction", "qa"
        predictions: List of predictions
        references: List of ground truth values
        **kwargs: Additional arguments passed to specific metric functions

    Returns:
        Dictionary of computed metrics
    """
    task = task.lower().strip()

    if task == "classification":
        return compute_classification_metrics(predictions, references, **kwargs)
    elif task == "extraction":
        return compute_extraction_metrics(predictions, references)
    elif task in ("qa", "question_answering", "qna"):
        return compute_qa_metrics(predictions, references, **kwargs)
    else:
        raise ValueError(
            f"Unknown task: {task}. Must be one of: classification, extraction, qa"
        )
