"""
MinerU GND/OCR processor.
Reads MinerU JSON output, applies rule-based filtering,
generates GND and OCR training samples.

MinerU element format:
{
    "type": "text" | "table" | "image" | "equation" | "discarded",
    "text": "...",
    "bbox": [x1, y1, x2, y2],  # pixel coordinates
    "page_idx": 0,
    "text_level": 1,  # optional, 1 = title/heading
    "table_caption": ["..."],   # optional
    "image_caption": ["..."],   # optional
    "table_footnote": ["..."],  # optional
    "image_footnote": ["..."],  # optional
    "table_body": "...",        # optional
}
"""
import os
import json
import re
import datasets
from typing import Dict, List, Optional, Tuple
from .base import DatasetProcessor
try:
    from ..utils import make_sample, normalize_bbox_pixel_to_01, format_bbox_str
except ImportError:
    from utils import make_sample, normalize_bbox_pixel_to_01, format_bbox_str


def is_garbled(text: str, threshold: float = 0.3) -> bool:
    """Check if text is garbled (too many non-alphanumeric characters)."""
    if not text:
        return True
    alnum = sum(c.isalnum() or c.isspace() for c in text)
    return alnum / len(text) < threshold


def get_image_size(image_path: str) -> Optional[Tuple[int, int]]:
    """Get image width, height without loading full image."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


class MinerUProcessor(DatasetProcessor):
    """Process MinerU JSON outputs into GND and OCR training samples."""

    def __init__(self, name: str, config: dict, output_dir: str):
        super().__init__(name, config, output_dir)
        self.mineru_dirs = config["mineru_dirs"]  # {"docvqa": path, "infovqa": path}
        self.image_dirs = config["image_dirs"]    # {"docvqa": path, "infovqa": path}
        self.tasks = config.get("tasks", ["gnd", "ocr"])
        self._stats = {
            "name": name, "total_files": 0, "total_elements": 0,
            "after_filter": 0, "gnd_count": 0, "ocr_count": 0,
            "skipped": {"no_image": 0, "garbled": 0, "too_short": 0,
                        "too_long": 0, "discarded_type": 0, "no_bbox": 0},
        }

    def load(self) -> datasets.Dataset:
        return None  # not used

    def process(self) -> Dict[str, datasets.Dataset]:
        gnd_samples = []
        ocr_samples = []

        for source_name, mineru_dir in self.mineru_dirs.items():
            image_dir = self.image_dirs.get(source_name)
            if not os.path.isdir(mineru_dir):
                print(f"  [SKIP] MinerU dir not found: {mineru_dir}")
                continue

            files = [f for f in os.listdir(mineru_dir) if f.endswith(".json")]
            self._stats["total_files"] += len(files)
            print(f"  Processing {source_name}: {len(files)} MinerU files")

            for fname in files:
                doc_id = fname.replace(".json", "")
                json_path = os.path.join(mineru_dir, fname)

                # Find corresponding image
                image_path = self._find_image(doc_id, image_dir)
                if image_path is None:
                    self._stats["skipped"]["no_image"] += 1
                    continue

                img_size = get_image_size(image_path)
                if img_size is None:
                    continue
                img_w, img_h = img_size

                # Load and process elements
                try:
                    elements = json.load(open(json_path))
                except Exception:
                    continue

                for elem in elements:
                    self._stats["total_elements"] += 1
                    result = self._process_element(
                        elem, doc_id, source_name, image_path, img_w, img_h
                    )
                    if result is None:
                        continue

                    task_type, sample = result
                    if task_type == "gnd" and "gnd" in self.tasks:
                        gnd_samples.append(sample)
                    elif task_type == "ocr" and "ocr" in self.tasks:
                        ocr_samples.append(sample)

                # Also extract from captions and footnotes
                for elem in elements:
                    extras = self._extract_caption_footnote_samples(
                        elem, doc_id, source_name, image_path, img_w, img_h
                    )
                    for task_type, sample in extras:
                        if task_type == "gnd" and "gnd" in self.tasks:
                            gnd_samples.append(sample)
                        elif task_type == "ocr" and "ocr" in self.tasks:
                            ocr_samples.append(sample)

        results = {}
        self._stats["gnd_count"] = len(gnd_samples)
        self._stats["ocr_count"] = len(ocr_samples)
        self._stats["after_filter"] = len(gnd_samples) + len(ocr_samples)
        print(f"  MinerU total: GND={len(gnd_samples)}, OCR={len(ocr_samples)}")

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

    def _process_element(self, elem, doc_id, source, image_path, img_w, img_h):
        """Process a single MinerU element into a training sample."""
        etype = elem.get("type", "")
        text = elem.get("text", "").strip()
        bbox = elem.get("bbox")

        # Skip discarded elements
        if etype in ("discarded", ""):
            self._stats["skipped"]["discarded_type"] += 1
            return None

        if not bbox or len(bbox) != 4:
            self._stats["skipped"]["no_bbox"] += 1
            return None

        # Text quality checks (only for OCR, GND doesn't need text content)
        text_ok = self._check_text_quality(text)

        # Normalize bbox to [0, 1]
        bbox_norm = normalize_bbox_pixel_to_01(bbox, img_w, img_h)
        bbox_str = format_bbox_str(bbox_norm)

        # Decide task based on element type
        text_level = elem.get("text_level", None)
        is_title = text_level == 1

        # GND: locate elements (tables, images, equations, titles)
        if etype in ("table", "image", "equation") or is_title:
            question = self._make_gnd_question(etype, text, is_title, doc_id)
            if question:
                idx = hash(f"{doc_id}_{etype}_{bbox}") % (10**8)
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

        # OCR: read text from elements (text with sufficient quality)
        if etype in ("text", "equation") and text_ok:
            question = self._make_ocr_question(etype, bbox_str, is_title, doc_id)
            idx = hash(f"{doc_id}_ocr_{bbox}") % (10**8)
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

    def _extract_caption_footnote_samples(self, elem, doc_id, source, image_path, img_w, img_h):
        """Extract GND/OCR samples from caption and footnote fields."""
        results = []
        bbox = elem.get("bbox")
        if not bbox or len(bbox) != 4:
            return results

        bbox_norm = normalize_bbox_pixel_to_01(bbox, img_w, img_h)
        bbox_str = format_bbox_str(bbox_norm)

        # Process captions and footnotes
        for field, label in [
            ("table_caption", "table caption"),
            ("image_caption", "figure caption"),
            ("table_footnote", "table footnote"),
            ("image_footnote", "figure footnote"),
        ]:
            values = elem.get(field)
            if not values:
                continue
            if isinstance(values, str):
                values = [values]

            for text in values:
                text = text.strip()
                if not self._check_text_quality(text):
                    continue

                # GND: "Where is the {label}?"
                question = f'Where is the {label} that reads "{text[:50]}"?'
                idx = hash(f"{doc_id}_{field}_{text[:30]}") % (10**8)
                gnd_sample = make_sample(
                    data_source=f"mineru_{source}_gnd",
                    question=question,
                    image_path=image_path,
                    ground_truth=bbox_str,
                    task_type="gnd",
                    dataset=f"mineru_{source}",
                    split="train",
                    index=idx,
                    extra_fields={"element_type": field, "doc_id": doc_id},
                )
                results.append(("gnd", gnd_sample))

                # OCR: "Read the {label}"
                ocr_question = f"Read the {label} near the region {bbox_str} in the document."
                ocr_sample = make_sample(
                    data_source=f"mineru_{source}_ocr",
                    question=ocr_question,
                    image_path=image_path,
                    ground_truth=text,
                    task_type="ocr",
                    dataset=f"mineru_{source}",
                    split="train",
                    index=idx + 1,
                    extra_fields={"element_type": field, "doc_id": doc_id},
                )
                results.append(("ocr", ocr_sample))

        return results

    def _check_text_quality(self, text: str) -> bool:
        """Check if text passes quality filters."""
        if not text:
            self._stats["skipped"]["too_short"] += 1
            return False
        if len(text) < 3:
            self._stats["skipped"]["too_short"] += 1
            return False
        if len(text) > 200:
            self._stats["skipped"]["too_long"] += 1
            return False
        if is_garbled(text):
            self._stats["skipped"]["garbled"] += 1
            return False
        return True

    def _make_gnd_question(self, etype, text, is_title, doc_id):
        """Generate a natural GND question."""
        if etype == "table":
            return "Where is the table in this document?"
        elif etype == "image":
            return "Where is the figure in this document?"
        elif etype == "equation":
            return "Where is the equation in this document?"
        elif is_title and text:
            short_text = text[:60].strip()
            return f'Where is the section titled "{short_text}" in this document?'
        return None

    def _make_ocr_question(self, etype, bbox_str, is_title, doc_id):
        """Generate a natural OCR question."""
        if is_title:
            return f"Read the heading text at region {bbox_str} in the document."
        elif etype == "equation":
            return f"Read the equation at region {bbox_str} in the document."
        else:
            return f"Read the text at region {bbox_str} in the document."

    def get_stats(self) -> dict:
        return self._stats
