"""
Official TextVQA / VQA Accuracy metric.
Follows the VQA Challenge evaluation protocol exactly.

Reference: https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py
           https://visualqa.org/evaluation.html
"""
import re
from typing import List


# --- Official preprocessing from VQA eval ---

_contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebod'd", "## somebodyd": "somebody'd",
    "somebody'dve": "somebody'd've", "somebodyd've": "somebody'd've",
    "someone'd": "someone'd", "someoned": "someone'd",
    "someone'dve": "someone'd've", "someoned've": "someone'd've",
    "somethingd": "something'd", "something'dve": "something'd've",
    "somethingd've": "something'd've", "thatd": "that'd",
    "thatd've": "that'd've", "that'dve": "that'd've", "thats": "that's",
    "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're",
    "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
    "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd": "y'all'd", "y'alld": "y'all'd", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}

_manual_map = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}

_articles = ["a", "an", "the"]

_period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
_comma_strip = re.compile(r"(\d)(\,)(\d)")
_punct = [
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+",
    "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!",
]


def _process_punctuation(inText: str) -> str:
    """Official VQA punctuation processing."""
    outText = inText
    for p in _punct:
        if (p + " " in inText or " " + p in inText) or (
            re.search(_comma_strip, inText) is not None
        ):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = _period_strip.sub("", outText, count=1)
    return outText


def _process_digit_article(inText: str) -> str:
    """Official VQA digit and article processing."""
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = _manual_map.get(word, word)
        if word not in _articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in _contractions:
            outText[wordId] = _contractions[word]
    return " ".join(outText)


def _preprocess_answer(answer: str) -> str:
    """Full official VQA answer preprocessing pipeline."""
    answer = answer.replace("\n", " ").replace("\t", " ").strip()
    answer = _process_punctuation(answer)
    answer = _process_digit_article(answer)
    return answer


def textvqa_accuracy_score(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute official TextVQA / VQA accuracy for a single prediction.
    Uses the 10-annotator soft accuracy formula.

    Args:
        prediction: Model prediction string.
        ground_truths: List of human-annotated answers (typically 10).

    Returns:
        Accuracy score in [0, 1].
    """
    if not ground_truths:
        return 0.0

    prediction = _preprocess_answer(prediction)
    processed_gts = [_preprocess_answer(gt) for gt in ground_truths]

    gt_accuracies = []
    for i, gt in enumerate(processed_gts):
        other_gts = processed_gts[:i] + processed_gts[i + 1:]
        matching = sum(1 for other in other_gts if other == prediction)
        acc = min(1.0, matching / 3.0)
        gt_accuracies.append(acc)

    return sum(gt_accuracies) / len(gt_accuracies)
