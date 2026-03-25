"""
VISA (Paper-VISA / Wiki-VISA) processor.
Reads from local JSONL files with QA data + local images.
No HuggingFace download needed.
"""
import os
import json
import datasets
from typing import Dict
from .base import DatasetProcessor
try:
    from ..utils import make_sample, normalize_bbox_pixel_to_01, format_bbox_str
except ImportError:
    from utils import make_sample, normalize_bbox_pixel_to_01, format_bbox_str


class VISAProcessor(DatasetProcessor):

    def __init__(self, name: str, config: dict, output_dir: str):
        super().__init__(name, config, output_dir)
        self.qa_file = config["qa_file"]
        self.data_root = config.get("data_root", "")
        self.max_samples = config.get("max_samples", None)
        self.tasks = config.get("tasks", ["vqa"])
        self.filter_pass_rate = config.get("filter_pass_rate", False)
        self.scores_file = config.get("scores_file", None)
        self._stats = {"name": name, "raw_count": 0, "after_filter": 0,
                       "vqa_count": 0, "gnd_count": 0, "skipped_no_image": 0}

    def _load_scores(self) -> Dict[str, float]:
        """Load pass_rate scores."""
        scores = {}
        if not self.scores_file or not os.path.exists(self.scores_file):
            return scores
        with open(self.scores_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                scores[item["id"]] = item["pass_rate"]
        print(f"  Loaded {len(scores)} scores from {os.path.basename(self.scores_file)}")
        return scores

    def _load_qa(self):
        """Load QA data from local JSONL file."""
        records = []
        with open(self.qa_file, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        return records

    def load(self) -> datasets.Dataset:
        return None

    def process(self) -> Dict[str, datasets.Dataset]:
        qa_data = self._load_qa()
        self._stats["raw_count"] = len(qa_data)

        scores = self._load_scores() if self.filter_pass_rate else {}
        results = {}

        vqa_samples = []
        gnd_samples = []

        for i, item in enumerate(qa_data):
            if self.max_samples and i >= self.max_samples:
                break

            # Resolve image path (relative to data_root)
            image_rel = item["image"]
            image_path = os.path.join(self.data_root, image_rel)
            if not os.path.exists(image_path):
                self._stats["skipped_no_image"] += 1
                continue

            # Hard case filter
            if self.filter_pass_rate and scores:
                record_id = item["id"]
                pass_rate = scores.get(record_id, None)
                if pass_rate is not None and pass_rate >= 1.0:
                    continue

            answers = item.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]

            # VQA sample
            if "vqa" in self.tasks:
                sample = make_sample(
                    data_source=self.name,
                    question=item["question"],
                    image_path=image_path,
                    ground_truth=answers,
                    task_type="vqa",
                    dataset=self.name,
                    split="train",
                    index=i,
                )
                vqa_samples.append(sample)

            # GND sample (if bbox available)
            if "gnd" in self.tasks:
                bbox = item.get("bbox")
                if bbox and len(bbox) == 4:
                    from PIL import Image
                    try:
                        img = Image.open(image_path)
                        w, h = img.size
                        img.close()
                    except Exception:
                        continue
                    bbox_norm = normalize_bbox_pixel_to_01(bbox, w, h)
                    bbox_str = format_bbox_str(bbox_norm)

                    gnd_question = f'Where in the document is the evidence for: {item["question"]}'
                    gnd_sample = make_sample(
                        data_source=self.name,
                        question=gnd_question,
                        image_path=image_path,
                        ground_truth=bbox_str,
                        task_type="gnd",
                        dataset=self.name,
                        split="train",
                        index=i,
                    )
                    gnd_samples.append(gnd_sample)

        self._stats["after_filter"] = len(vqa_samples)
        self._stats["vqa_count"] = len(vqa_samples)
        self._stats["gnd_count"] = len(gnd_samples)
        print(f"  {self.name}: {self._stats['raw_count']} → VQA={len(vqa_samples)}, GND={len(gnd_samples)}")

        if vqa_samples:
            results["vqa"] = datasets.Dataset.from_list(vqa_samples)
        if gnd_samples:
            results["gnd"] = datasets.Dataset.from_list(gnd_samples)

        return results

    def get_stats(self) -> dict:
        return self._stats
