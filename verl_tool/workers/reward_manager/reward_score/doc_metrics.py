"""Document understanding evaluation metrics: ANLS, IoU, NED similarity."""
import re
from typing import List, Optional, Union


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} in model response."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def compute_anls(prediction: str, ground_truths: Union[str, List[str]], threshold: float = 0.5) -> float:
    """
    Compute ANLS (Average Normalized Levenshtein Similarity) for VQA.
    Standard metric for DocVQA, InfoVQA, etc.

    Returns max ANLS across all acceptable ground truth answers.
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    if not prediction:
        return 0.0

    prediction = prediction.strip().lower()
    max_score = 0.0
    for gt in ground_truths:
        gt = gt.strip().lower()
        if not gt and not prediction:
            max_score = 1.0
            continue
        if not gt or not prediction:
            continue
        ned = _normalized_edit_distance(prediction, gt)
        score = 1.0 - ned if ned < threshold else 0.0
        max_score = max(max_score, score)
    return max_score


def compute_iou(pred_bbox: List[float], gt_bbox: List[float]) -> float:
    """
    Compute IoU (Intersection over Union) between two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format (normalized 0-1 or pixel coords).
    """
    if len(pred_bbox) != 4 or len(gt_bbox) != 4:
        return 0.0

    x1 = max(pred_bbox[0], gt_bbox[0])
    y1 = max(pred_bbox[1], gt_bbox[1])
    x2 = min(pred_bbox[2], gt_bbox[2])
    y2 = min(pred_bbox[3], gt_bbox[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area_pred = max(0, pred_bbox[2] - pred_bbox[0]) * max(0, pred_bbox[3] - pred_bbox[1])
    area_gt = max(0, gt_bbox[2] - gt_bbox[0]) * max(0, gt_bbox[3] - gt_bbox[1])
    union = area_pred + area_gt - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def compute_ned_similarity(prediction: str, ground_truth: str) -> float:
    """
    Compute normalized edit distance similarity for OCR.
    Returns 1 - NED where NED = edit_distance / max(len(pred), len(gt)).
    """
    prediction = prediction.strip()
    ground_truth = ground_truth.strip()
    if not prediction and not ground_truth:
        return 1.0
    if not prediction or not ground_truth:
        return 0.0
    ned = _normalized_edit_distance(prediction, ground_truth)
    return 1.0 - ned


def _normalized_edit_distance(s1: str, s2: str) -> float:
    """Compute normalized edit distance between two strings."""
    dist = _edit_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return dist / max_len


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    # Use single-row DP for space efficiency
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


def parse_bbox_from_text(text: str) -> Optional[List[float]]:
    """
    Parse bounding box coordinates from text.
    Handles formats like: [0.1, 0.2, 0.8, 0.9] or 0.1, 0.2, 0.8, 0.9
    """
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if len(numbers) >= 4:
        try:
            bbox = [float(x) for x in numbers[:4]]
            return bbox
        except ValueError:
            return None
    return None
