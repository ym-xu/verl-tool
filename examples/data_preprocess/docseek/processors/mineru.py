"""
MinerU GND/OCR processor.
Reads MinerU JSON output, applies rule-based filtering,
generates GND and OCR training samples.

Key fixes:
- bbox already in 0-1000 (converted by previous project), no double conversion
- Deduplication within same document
- No caption/footnote GND/OCR (parent bbox doesn't match caption location)
- InfoVQA: only text OCR (infographics have irregular table/image elements)
- Stricter text quality filters

MinerU element format:
{
    "type": "text" | "table" | "image" | "equation" | "discarded",
    "text": "...",
    "bbox": [x1, y1, x2, y2],  # already in 0-1000 coordinates
    "text_level": 1,            # optional, 1 = title/heading
    ...
}
"""
import os
import json
import datasets
from typing import Dict, List, Optional, Set, Tuple
from .base import DatasetProcessor
try:
    from ..utils import make_sample, format_bbox_str
except ImportError:
    from utils import make_sample, format_bbox_str


def is_garbled(text: str, threshold: float = 0.3) -> bool:
    """Check if text is garbled (too many non-alphanumeric characters)."""
    if not text:
        return True
    alnum = sum(c.isalnum() or c.isspace() for c in text)
    return alnum / len(text) < threshold


def has_latex_noise(text: str) -> bool:
    """Check if text has excessive LaTeX artifacts."""
    if not text:
        return True
    latex_chars = sum(1 for c in text if c in '\\{}$^_')
    return latex_chars / len(text) > 0.3


def has_word_merge(text: str, max_word_len: int = 20) -> bool:
    """Detect MinerU OCR word merging (missing spaces at line breaks).
    Normal English words rarely exceed 20 chars. Merged words like
    'includingsmokinghabits' (22 chars) indicate broken OCR.
    """
    if not text:
        return False
    return any(len(w) > max_word_len for w in text.split())


class MinerUProcessor(DatasetProcessor):
    """Process MinerU JSON outputs into GND and OCR training samples.

    Important: MinerU bbox is already in 0-1000 coordinates (converted upstream).
    We do NOT re-normalize — just use bbox directly.
    """

    def __init__(self, name: str, config: dict, output_dir: str):
        super().__init__(name, config, output_dir)
        self.mineru_dirs = config["mineru_dirs"]
        self.image_dirs = config["image_dirs"]
        self.tasks = config.get("tasks", ["gnd", "ocr"])
        self.filter_pass_rate = config.get("filter_pass_rate", False)
        self.gnd_scores_file = config.get("gnd_scores_file", None)
        self.ocr_scores_file = config.get("ocr_scores_file", None)
        # Per-source rules
        self.source_rules = config.get("source_rules", {})
        self._stats = {
            "name": name, "total_files": 0, "total_elements": 0,
            "after_filter": 0, "gnd_count": 0, "ocr_count": 0,
            "dedup_removed": 0,
            "skipped": {"no_image": 0, "garbled": 0, "too_short": 0,
                        "too_long": 0, "discarded_type": 0, "no_bbox": 0,
                        "latex_noise": 0, "word_merge": 0, "duplicate": 0,
                        "source_rule": 0},
        }

    def load(self) -> datasets.Dataset:
        return None

    def _load_scores(self, scores_file: Optional[str]) -> Dict[int, float]:
        """Load pass_rate scores keyed by index."""
        scores = {}
        if not scores_file or not os.path.exists(scores_file):
            return scores
        with open(scores_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                scores[item["index"]] = item["pass_rate"]
        print(f"  Loaded {len(scores)} scores from {os.path.basename(scores_file)}")
        return scores

    def process(self) -> Dict[str, datasets.Dataset]:
        gnd_samples = []
        ocr_samples = []

        gnd_scores = self._load_scores(self.gnd_scores_file) if self.filter_pass_rate else {}
        ocr_scores = self._load_scores(self.ocr_scores_file) if self.filter_pass_rate else {}

        for source_name, mineru_dir in self.mineru_dirs.items():
            image_dir = self.image_dirs.get(source_name)
            if not os.path.isdir(mineru_dir):
                print(f"  [SKIP] MinerU dir not found: {mineru_dir}")
                continue

            files = [f for f in os.listdir(mineru_dir) if f.endswith(".json")]
            self._stats["total_files"] += len(files)
            print(f"  Processing {source_name}: {len(files)} MinerU files")

            # Get source-specific rules
            rules = self.source_rules.get(source_name, {})
            # InfoVQA: only text OCR, no table/image GND (infographics are irregular)
            gnd_types = rules.get("gnd_types", ["table", "image", "equation"])
            ocr_types = rules.get("ocr_types", ["text", "equation"])

            for fname in files:
                doc_id = fname.replace(".json", "")

                image_path = self._find_image(doc_id, image_dir)
                if image_path is None:
                    self._stats["skipped"]["no_image"] += 1
                    continue

                try:
                    elements = json.load(open(os.path.join(mineru_dir, fname)))
                except Exception:
                    continue

                # Dedup within document: track seen texts and bboxes
                seen_texts: Set[str] = set()
                seen_bboxes: Set[str] = set()

                for elem in elements:
                    self._stats["total_elements"] += 1

                    result = self._process_element(
                        elem, doc_id, source_name, image_path,
                        gnd_types, ocr_types, seen_texts, seen_bboxes
                    )
                    if result is None:
                        continue

                    task_type, sample = result
                    if task_type == "gnd" and "gnd" in self.tasks:
                        gnd_samples.append(sample)
                    elif task_type == "ocr" and "ocr" in self.tasks:
                        ocr_samples.append(sample)

        # Apply pass rate filter if scores are available
        if self.filter_pass_rate and gnd_scores:
            before = len(gnd_samples)
            gnd_samples = [s for i, s in enumerate(gnd_samples)
                           if gnd_scores.get(i, 0.0) < 1.0]
            print(f"  GND pass_rate filter: {before} → {len(gnd_samples)}")

        if self.filter_pass_rate and ocr_scores:
            before = len(ocr_samples)
            ocr_samples = [s for i, s in enumerate(ocr_samples)
                           if ocr_scores.get(i, 0.0) < 1.0]
            print(f"  OCR pass_rate filter: {before} → {len(ocr_samples)}")

        results = {}
        self._stats["gnd_count"] = len(gnd_samples)
        self._stats["ocr_count"] = len(ocr_samples)
        self._stats["after_filter"] = len(gnd_samples) + len(ocr_samples)
        print(f"  MinerU final: GND={len(gnd_samples)}, OCR={len(ocr_samples)}")
        print(f"  Skipped: {self._stats['skipped']}")

        if gnd_samples:
            results["gnd"] = datasets.Dataset.from_list(gnd_samples)
        if ocr_samples:
            results["ocr"] = datasets.Dataset.from_list(ocr_samples)

        return results

    def _find_image(self, doc_id: str, image_dir: str) -> Optional[str]:
        """Find image file matching MinerU doc_id."""
        if not image_dir or not os.path.isdir(image_dir):
            return None
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            path = os.path.join(image_dir, doc_id + ext)
            if os.path.exists(path):
                return path
        return None

    def _process_element(self, elem, doc_id, source, image_path,
                         gnd_types, ocr_types, seen_texts, seen_bboxes):
        """Process a single MinerU element. No bbox re-normalization needed."""
        etype = elem.get("type", "")
        text = elem.get("text", "").strip()
        bbox = elem.get("bbox")

        # Skip discarded / empty type
        if etype in ("discarded", ""):
            self._stats["skipped"]["discarded_type"] += 1
            return None

        if not bbox or len(bbox) != 4:
            self._stats["skipped"]["no_bbox"] += 1
            return None

        # bbox is already 0-1000, use directly
        bbox_key = f"{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}"

        # Dedup: skip if we've seen this exact bbox in this document
        if bbox_key in seen_bboxes:
            self._stats["skipped"]["duplicate"] += 1
            return None
        seen_bboxes.add(bbox_key)

        bbox_str = format_bbox_str([int(b) for b in bbox])

        text_level = elem.get("text_level", None)
        is_title = text_level == 1

        # --- GND: locate elements ---
        if etype in gnd_types or is_title:
            question = self._make_gnd_question(etype, text, is_title)
            if question:
                idx = hash(f"{doc_id}_{etype}_{bbox_key}") % (10**8)
                sample = make_sample(
                    data_source=f"mineru_{source}_gnd",
                    question=question,
                    image_path=image_path,
                    ground_truth=bbox_str,
                    task_type="gnd",
                    dataset=f"mineru_{source}",
                    split="train",
                    index=idx,
                    extra_fields={"element_type": etype, "doc_id": doc_id},
                )
                return ("gnd", sample)

        # --- OCR: read text ---
        if etype in ocr_types:
            if not self._check_text_quality(text):
                return None

            # Dedup: skip if we've seen this exact text in this document
            text_key = text[:100].lower().strip()
            if text_key in seen_texts:
                self._stats["skipped"]["duplicate"] += 1
                return None
            seen_texts.add(text_key)

            question = self._make_ocr_question(etype, bbox_str, is_title)
            idx = hash(f"{doc_id}_ocr_{bbox_key}") % (10**8)
            sample = make_sample(
                data_source=f"mineru_{source}_ocr",
                question=question,
                image_path=image_path,
                ground_truth=text,
                task_type="ocr",
                dataset=f"mineru_{source}",
                split="train",
                index=idx,
                extra_fields={"element_type": etype, "doc_id": doc_id},
            )
            return ("ocr", sample)

        return None

    def _check_text_quality(self, text: str) -> bool:
        """Stricter text quality filters for OCR ground truth."""
        if not text:
            self._stats["skipped"]["too_short"] += 1
            return False
        if len(text) < 5:
            self._stats["skipped"]["too_short"] += 1
            return False
        if len(text) > 150:
            self._stats["skipped"]["too_long"] += 1
            return False
        if is_garbled(text):
            self._stats["skipped"]["garbled"] += 1
            return False
        if has_latex_noise(text):
            self._stats["skipped"]["latex_noise"] += 1
            return False
        # Detect MinerU word merge (OCR: max 20 chars per word)
        if has_word_merge(text, max_word_len=20):
            self._stats["skipped"]["word_merge"] += 1
            return False
        # Must have at least 2 real words
        words = [w for w in text.split() if len(w) > 1]
        if len(words) < 2:
            self._stats["skipped"]["too_short"] += 1
            return False
        return True

    def _make_gnd_question(self, etype, text, is_title):
        """Generate a natural GND question."""
        if etype == "table":
            return "Where is the table in this document?"
        elif etype == "image":
            return "Where is the figure in this document?"
        elif etype == "equation":
            return "Where is the equation in this document?"
        elif is_title and text:
            short_text = text[:60].strip()
            # Filter merged title text (GND titles: max 15 chars per word)
            if has_word_merge(short_text, max_word_len=15):
                self._stats["skipped"]["word_merge"] += 1
                return None
            return f'Where is the section titled "{short_text}" in this document?'
        return None

    def _make_ocr_question(self, etype, bbox_str, is_title):
        """Generate a natural OCR question."""
        if is_title:
            return f"Read the heading text at region {bbox_str} in the document."
        elif etype == "equation":
            return f"Read the equation at region {bbox_str} in the document."
        else:
            return f"Read the text at region {bbox_str} in the document."

    def get_stats(self) -> dict:
        return self._stats
