"""
Official ANLS (Average Normalized Levenshtein Similarity) metric.
Used by: DocVQA, InfographicsVQA, WildDoc-DocVQA.

Reference: https://github.com/rubenpt91/MP-DocVQA-Framework/blob/master/metrics.py
           https://rrc.cvc.uab.es/?ch=17
"""
from typing import List, Union


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
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


def anls_score(prediction: str, ground_truths: Union[str, List[str]], threshold: float = 0.5) -> float:
    """
    Compute ANLS for a single prediction against one or more ground truths.
    Exactly follows the official DocVQA evaluation protocol.

    Args:
        prediction: Model prediction string.
        ground_truths: Single string or list of acceptable answer strings.
        threshold: ANLS threshold tau (default 0.5).

    Returns:
        ANLS score in [0, 1].
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    prediction = prediction.strip().lower()
    if len(prediction) == 0:
        return 0.0

    max_similarity = 0.0
    for gt in ground_truths:
        gt = gt.strip().lower()
        if len(gt) == 0 and len(prediction) == 0:
            max_similarity = 1.0
            continue
        if len(gt) == 0 or len(prediction) == 0:
            continue
        edit_dist = _levenshtein_distance(gt, prediction)
        max_len = max(len(gt), len(prediction))
        similarity = 1.0 - edit_dist / max_len
        max_similarity = max(max_similarity, similarity)

    return max_similarity if max_similarity >= threshold else 0.0
