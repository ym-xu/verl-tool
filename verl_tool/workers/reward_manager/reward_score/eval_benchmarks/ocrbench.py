"""
Official OCRBench (v1) evaluation metric.
Uses substring containment matching with case-insensitive preprocessing.

Reference: https://github.com/Yuliang-Liu/MultimodalOCR/blob/main/OCRBench/scripts/monkey.py
"""


def ocrbench_score(prediction: str, ground_truth: str, is_hme100k: bool = False) -> int:
    """
    Compute OCRBench score for a single sample.
    Binary: 1 if ground truth is a substring of prediction, else 0.

    Args:
        prediction: Model prediction string.
        ground_truth: Ground truth answer string.
        is_hme100k: If True, uses HME100k (handwritten math) preprocessing
                     which preserves case and strips all spaces.

    Returns:
        1 if correct, 0 if incorrect.
    """
    if is_hme100k:
        # HME100k: preserve case, remove all spaces
        answer = ground_truth.strip().replace("\n", " ").replace(" ", "")
        predict = prediction.strip().replace("\n", " ").replace(" ", "")
    else:
        # Standard: case-insensitive
        answer = ground_truth.lower().strip().replace("\n", " ")
        predict = prediction.lower().strip().replace("\n", " ")

    if answer in predict:
        return 1
    return 0
