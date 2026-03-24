"""
Official VISA (Paper-VISA, Wiki-VISA) evaluation metrics.
Two metrics: answer quality (token subsequence match) and source attribution (IoU).

Reference: https://github.com/castorini/visa/blob/main/eval.py
"""
import re
import unicodedata
from typing import List, Optional


def _normalize(text: str) -> str:
    """Unicode NFD normalization, matching official VISA eval."""
    return unicodedata.normalize("NFD", text)


def _simple_tokenize(text: str) -> List[str]:
    """
    Simple whitespace + punctuation tokenizer (uncased).
    Approximates SpaCy tokenizer used in official code.
    Splits on whitespace and strips punctuation from tokens.
    """
    text = text.lower()
    # Split on whitespace and punctuation boundaries
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def visa_has_answer(text: str, answers: List[str], regex: bool = False) -> bool:
    """
    Check if any answer appears as a token subsequence in text.
    Follows official VISA eval.py has_answers() function.

    Args:
        text: Model prediction text.
        answers: List of ground truth answer strings.
        regex: If True, use regex matching instead of token subsequence.

    Returns:
        True if any answer is found in text.
    """
    text = _normalize(text)

    if regex:
        for ans in answers:
            ans = _normalize(ans)
            if _regex_match(text, ans):
                return True
    else:
        text_tokens = _simple_tokenize(text)
        for ans in answers:
            ans = _normalize(ans)
            ans_tokens = _simple_tokenize(ans)
            if len(ans_tokens) == 0:
                continue
            # Check contiguous subsequence
            for i in range(len(text_tokens) - len(ans_tokens) + 1):
                if ans_tokens == text_tokens[i: i + len(ans_tokens)]:
                    return True
    return False


def _regex_match(text: str, pattern: str) -> bool:
    """Regex match following official VISA implementation."""
    try:
        compiled = re.compile(pattern, flags=re.IGNORECASE | re.UNICODE | re.MULTILINE)
    except re.error:
        return False
    return compiled.search(text) is not None


def visa_iou(bbox_a: List[float], bbox_b: List[float], threshold: float = 0.5) -> bool:
    """
    Check if IoU between two bounding boxes meets threshold.
    Follows official VISA eval.py enough_iou() function.

    Args:
        bbox_a: [x1, y1, x2, y2] predicted bbox.
        bbox_b: [x1, y1, x2, y2] ground truth bbox.
        threshold: IoU threshold (default 0.5).

    Returns:
        True if IoU >= threshold.
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    i_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u_area = a_area + b_area - i_area

    if u_area <= 0:
        return False
    return (i_area / u_area) >= threshold


def visa_answer_score(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute VISA answer quality score.

    Returns:
        1.0 if answer found, 0.0 otherwise.
    """
    if visa_has_answer(prediction, ground_truths):
        return 1.0
    return 0.0
