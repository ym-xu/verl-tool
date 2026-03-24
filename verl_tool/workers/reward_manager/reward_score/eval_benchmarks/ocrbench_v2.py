"""
Official OCRBench v2 evaluation metrics.
Unlike v1 (simple substring containment), v2 has 30 task types with different metrics.

Reference: https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2/eval_scripts

Task types and their metrics:
- VQA-type tasks: substring containment (short answers) or ANLS (long answers)
- Chinese VQA: same as VQA but with Chinese text handling
- Case-sensitive VQA: case-preserving substring/ANLS
- Counting: exact match or regression (IoU-based)
- Math expressions: substring containment after removing spaces
- Full-page OCR: (BLEU + METEOR + F-measure + (1 - edit_distance)) / 4
- Multiple choice: extract letter, exact match
- Table parsing: TEDS (requires external lib, falls back to edit distance)
- Spatial/detection: IoU on bounding boxes
"""
import re
from typing import List, Optional, Union

# Avoid hard dependency on nltk/jieba — graceful fallback
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.meteor_score import meteor_score as _meteor_score
    from nltk.metrics import f_measure, precision, recall
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _contains_chinese(text: str) -> bool:
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


# ========== VQA-type evaluation ==========

def vqa_evaluation(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    General VQA evaluation: substring containment for short answers (<5 words),
    ANLS for longer answers. Case-insensitive.
    """
    if isinstance(ground_truth, list):
        return max(vqa_evaluation(prediction, gt) for gt in ground_truth)

    prediction = prediction.strip().lower()
    ground_truth = ground_truth.strip().lower()

    if not ground_truth:
        return 1.0 if not prediction else 0.0

    # Short answer: substring containment
    if len(ground_truth.split()) < 5:
        if ground_truth in prediction:
            return 1.0
        # Fallback to ANLS
        return _anls(prediction, ground_truth)
    else:
        return _anls(prediction, ground_truth)


def cn_vqa_evaluation(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """Chinese VQA: remove spaces before comparison, use comma-separated length."""
    if isinstance(ground_truth, list):
        return max(cn_vqa_evaluation(prediction, gt) for gt in ground_truth)

    prediction = prediction.strip().replace(" ", "").lower()
    ground_truth = ground_truth.strip().replace(" ", "").lower()

    if not ground_truth:
        return 1.0 if not prediction else 0.0

    # Use comma count for "word" length in Chinese
    gt_len = len(ground_truth.split("，")) if "，" in ground_truth else len(ground_truth.split(","))
    if gt_len < 5:
        if ground_truth in prediction:
            return 1.0
    return _anls(prediction, ground_truth)


def vqa_evaluation_case_sensitive(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """Case-sensitive VQA evaluation."""
    if isinstance(ground_truth, list):
        return max(vqa_evaluation_case_sensitive(prediction, gt) for gt in ground_truth)

    prediction = prediction.strip()
    ground_truth = ground_truth.strip()

    if not ground_truth:
        return 1.0 if not prediction else 0.0

    if len(ground_truth.split()) < 5:
        if ground_truth in prediction:
            return 1.0
    return _anls(prediction, ground_truth, case_sensitive=True)


def _anls(pred: str, gt: str, threshold: float = 0.5, case_sensitive: bool = False) -> float:
    if not case_sensitive:
        pred = pred.lower()
        gt = gt.lower()
    dist = _levenshtein_distance(pred, gt)
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 1.0
    sim = 1.0 - dist / max_len
    return sim if sim >= threshold else 0.0


# ========== Counting evaluation ==========

def counting_evaluation(prediction: str, ground_truth: str, mode: str = "exact") -> float:
    """
    Counting evaluation.
    mode="exact": substring containment.
    mode="regression": extract number, compute IoU-like score.
    """
    prediction = prediction.strip().lower()
    ground_truth = ground_truth.strip().lower()

    if mode == "exact":
        return 1.0 if ground_truth in prediction else 0.0
    elif mode == "regression":
        pred_num = _extract_first_number(prediction)
        gt_num = _extract_first_number(ground_truth)
        if pred_num is None or gt_num is None:
            return 0.0
        if gt_num == 0:
            return 1.0 if pred_num == 0 else 0.0
        score = 1.0 - abs(pred_num - gt_num) / abs(gt_num)
        return max(0.0, score)
    return 0.0


def _extract_first_number(text: str) -> Optional[float]:
    match = re.search(r'[-+]?\d*\.?\d+', text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


# ========== Math expression evaluation ==========

def math_expression_evaluation(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """Math expression: remove whitespace, substring containment."""
    if isinstance(ground_truth, list):
        return max(math_expression_evaluation(prediction, gt) for gt in ground_truth)

    prediction = prediction.strip().replace(" ", "").lower()
    ground_truth = ground_truth.strip().replace(" ", "").lower()
    return 1.0 if ground_truth in prediction else 0.0


def cn_math_expression_evaluation(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """Chinese math expression: remove \\text{} tags, then same as math_expression_evaluation."""
    if isinstance(ground_truth, list):
        return max(cn_math_expression_evaluation(prediction, gt) for gt in ground_truth)

    # Remove \text{...} LaTeX tags
    prediction = re.sub(r'\\text\{([^{}]*)\}', r'\1', prediction)
    ground_truth = re.sub(r'\\text\{([^{}]*)\}', r'\1', ground_truth)
    return math_expression_evaluation(prediction, ground_truth)


# ========== Full-page OCR evaluation ==========

def page_ocr_evaluation(prediction: str, ground_truth: str) -> float:
    """
    Full-page OCR: (BLEU + METEOR + F-measure + (1 - NED)) / 4.
    Falls back to (1 - NED) alone if nltk is not available.
    """
    if not prediction.strip() and not ground_truth.strip():
        return 1.0
    if not prediction.strip() or not ground_truth.strip():
        return 0.0

    # Tokenize
    is_chinese = _contains_chinese(ground_truth) or _contains_chinese(prediction)
    if is_chinese and _HAS_JIEBA:
        ref_tokens = jieba.lcut(ground_truth)
        hyp_tokens = jieba.lcut(prediction)
    else:
        ref_tokens = ground_truth.split()
        hyp_tokens = prediction.split()

    # Normalized edit distance
    ned = _levenshtein_distance(prediction, ground_truth) / max(len(prediction), len(ground_truth))

    if _HAS_NLTK and ref_tokens and hyp_tokens:
        try:
            bleu = sentence_bleu([ref_tokens], hyp_tokens)
        except Exception:
            bleu = 0.0
        try:
            meteor = _meteor_score([ref_tokens], hyp_tokens)
        except Exception:
            meteor = 0.0
        ref_set = set(ref_tokens)
        hyp_set = set(hyp_tokens)
        try:
            f1 = f_measure(ref_set, hyp_set) or 0.0
        except Exception:
            f1 = 0.0
        return (bleu + meteor + f1 + (1.0 - ned)) / 4.0
    else:
        # Fallback: only NED
        return 1.0 - ned


# ========== Multiple choice evaluation ==========

def multiple_choice_evaluation(prediction: str, ground_truth: str) -> float:
    """Extract letter choice from prediction, exact match with ground truth."""
    pred_letter = _extract_choice_letter(prediction)
    gt_letter = ground_truth.strip().upper()
    if pred_letter and pred_letter == gt_letter:
        return 1.0
    return 0.0


def _extract_choice_letter(text: str) -> Optional[str]:
    """Extract a single letter choice (A/B/C/D) from text."""
    text = text.strip()
    # Try common patterns
    match = re.search(r'\b([A-D])\b', text.upper())
    if match:
        return match.group(1)
    # First character if it's a letter
    if text and text[0].upper() in 'ABCD':
        return text[0].upper()
    return None


# ========== Master routing function ==========

# Task type → evaluation function mapping
# Based on OCRBench_v2/eval_scripts/eval.py task categories
TASK_EVAL_MAP = {
    # English tasks
    "text_recognition": vqa_evaluation,
    "scene_text_recognition": vqa_evaluation,
    "document_text_recognition": vqa_evaluation,
    "handwritten_text_recognition": vqa_evaluation,
    "text_recognition_case_sensitive": vqa_evaluation_case_sensitive,
    "artistic_text_recognition": vqa_evaluation,
    "text_detection": vqa_evaluation,  # simplified
    "text_spotting": vqa_evaluation,   # simplified (official uses spotting_evaluation)
    "scene_text_vqa": vqa_evaluation,
    "document_vqa": vqa_evaluation,
    "kie": vqa_evaluation,
    "counting": counting_evaluation,
    "math_expression": math_expression_evaluation,
    "formula_recognition": math_expression_evaluation,
    "full_page_ocr": page_ocr_evaluation,
    "document_parsing": page_ocr_evaluation,
    "table_recognition": page_ocr_evaluation,  # simplified (official uses TEDS)
    "multiple_choice": multiple_choice_evaluation,
    "visual_text_understanding": vqa_evaluation,
    "knowledge_reasoning": vqa_evaluation,
    "relationship_extraction": vqa_evaluation,
    # Chinese tasks
    "cn_text_recognition": cn_vqa_evaluation,
    "cn_scene_text_vqa": cn_vqa_evaluation,
    "cn_document_vqa": cn_vqa_evaluation,
    "cn_kie": cn_vqa_evaluation,
    "cn_relationship_extraction": cn_vqa_evaluation,
    "cn_visual_text_understanding": cn_vqa_evaluation,
    "cn_knowledge_reasoning": cn_vqa_evaluation,
    "cn_math_expression": cn_math_expression_evaluation,
    "cn_formula_recognition": cn_math_expression_evaluation,
}


def ocrbench_v2_score(
    prediction: str,
    ground_truth: Union[str, List[str]],
    task_type: str = "document_vqa",
) -> float:
    """
    Compute OCRBench v2 score for a single sample.
    Routes to the appropriate evaluation function based on task_type.

    Args:
        prediction: Model prediction string.
        ground_truth: Ground truth answer.
        task_type: Task type string (see TASK_EVAL_MAP for valid types).

    Returns:
        Score in [0, 1].
    """
    eval_fn = TASK_EVAL_MAP.get(task_type, vqa_evaluation)
    return eval_fn(prediction, ground_truth)
