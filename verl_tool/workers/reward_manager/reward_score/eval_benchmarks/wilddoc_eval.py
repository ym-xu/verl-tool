"""
Official WildDoc evaluation metrics.
- WildDoc-DocVQA: ANLS (same as DocVQA)
- WildDoc-ChartQA: Relaxed Accuracy
- WildDoc-TableVQA: subset-specific (fintabnet, vtabfact, vwtq)

Reference: https://github.com/bytedance/WildDoc
"""
import re
from typing import List, Union
try:
    from .docvqa_anls import anls_score
except ImportError:
    from docvqa_anls import anls_score


def wilddoc_docvqa_score(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    """WildDoc-DocVQA uses standard ANLS (same as DocVQA)."""
    return anls_score(prediction, ground_truths)


def wilddoc_chartqa_score(prediction: str, ground_truth: str) -> float:
    """
    WildDoc-ChartQA uses Relaxed Accuracy.
    Correct if prediction is within 5% of ground truth (for numbers)
    or exact match (for text).

    Following the ChartQA relaxed accuracy definition.
    """
    prediction = prediction.strip().lower()
    ground_truth = ground_truth.strip().lower()

    if prediction == ground_truth:
        return 1.0

    # Try numeric comparison with 5% tolerance
    try:
        pred_num = _extract_number(prediction)
        gt_num = _extract_number(ground_truth)
        if pred_num is not None and gt_num is not None:
            if gt_num == 0:
                return 1.0 if pred_num == 0 else 0.0
            if abs(pred_num - gt_num) / abs(gt_num) <= 0.05:
                return 1.0
    except (ValueError, ZeroDivisionError):
        pass

    return 0.0


def _extract_number(text: str) -> float:
    """Extract a number from text, handling commas and percentages."""
    text = text.strip().rstrip('%').replace(',', '').strip()
    try:
        return float(text)
    except ValueError:
        # Try to find a number in the text
        match = re.search(r'[-+]?\d*\.?\d+', text)
        if match:
            return float(match.group())
        return None


def wilddoc_tablevqa_fintabnet_score(prediction: str, ground_truth: str) -> float:
    """
    WildDoc-TableVQA fintabnetqa evaluation.
    Uses exact matching after normalization, with fuzzy fallback.
    """
    pred = _fintabnet_normalize(prediction)
    gt = _fintabnet_normalize(ground_truth)
    if pred == gt:
        return 1.0
    # Fuzzy: check if gt is substring of pred or vice versa
    if gt in pred or pred in gt:
        return 1.0
    return 0.0


def _fintabnet_normalize(text: str) -> str:
    """Normalize text for fintabnet evaluation."""
    text = text.strip().lower()
    text = text.replace(",", "").replace("$", "").replace("%", "")
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def wilddoc_tablevqa_vtabfact_score(prediction: str, ground_truth: str) -> float:
    """
    WildDoc-TableVQA vtabfact evaluation.
    Binary true/false matching.
    """
    pred = prediction.strip().lower()
    gt = ground_truth.strip().lower()

    pred_bool = None
    if 'true' in pred or 'yes' in pred:
        pred_bool = True
    elif 'false' in pred or 'no' in pred:
        pred_bool = False

    gt_bool = None
    if gt in ('true', 'yes', '1'):
        gt_bool = True
    elif gt in ('false', 'no', '0'):
        gt_bool = False

    if pred_bool is not None and gt_bool is not None and pred_bool == gt_bool:
        return 1.0
    return 0.0


def wilddoc_tablevqa_vwtq_score(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    """
    WildDoc-TableVQA vwtq/vwtq_syn evaluation.
    Uses denotation checking after TSV unescaping.
    """
    if isinstance(ground_truths, str):
        ground_truths = _tsv_unescape_list(ground_truths)

    pred_values = _to_value_list(prediction)
    gt_values = _to_value_list_from_list(ground_truths)

    if not pred_values and not gt_values:
        return 1.0
    if not pred_values or not gt_values:
        return 0.0

    # Check denotation: same set of values
    if set(pred_values) == set(gt_values):
        return 1.0
    return 0.0


def _tsv_unescape_list(text: str) -> List[str]:
    """Unescape TSV-formatted answer list."""
    text = text.strip()
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]
    items = [item.strip().strip("'\"") for item in text.split(',')]
    return [item for item in items if item]


def _to_value_list(text: str) -> List[str]:
    """Convert text to a list of normalized values."""
    text = text.strip().lower()
    # Try to parse as a list
    if ',' in text:
        return [v.strip() for v in text.split(',') if v.strip()]
    return [text] if text else []


def _to_value_list_from_list(items: List[str]) -> List[str]:
    """Normalize a list of ground truth values."""
    return [item.strip().lower() for item in items if item.strip()]
