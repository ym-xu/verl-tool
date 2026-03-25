"""
VISA (Paper-VISA / Wiki-VISA) processor.
Uses local image directory + HuggingFace QA data.
Images are matched by filename from local dir.
"""
import os
import json
import datasets
from typing import Dict
from .base import DatasetProcessor
from ..utils import make_sample


class VISAProcessor(DatasetProcessor):

    def __init__(self, name: str, config: dict, output_dir: str):
        super().__init__(name, config, output_dir)
        self.image_dir = config["image_dir"]
        self.hf_path = config["hf_path"]
        self.split = config.get("split", "train")
        self.max_samples = config.get("max_samples", None)
        self.tasks = config.get("tasks", ["vqa"])
        self.filter_pass_rate = config.get("filter_pass_rate", False)
        self.scores_file = config.get("scores_file", None)
        self._stats = {"name": name, "raw_count": 0, "after_filter": 0, "vqa_count": 0}

    def _load_scores(self) -> Dict[str, float]:
        """Load pass_rate scores from VISA scores file."""
        scores = {}
        if not self.scores_file or not os.path.exists(self.scores_file):
            return scores
        with open(self.scores_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                scores[item["id"]] = item["pass_rate"]
        print(f"  Loaded {len(scores)} scores from {os.path.basename(self.scores_file)}")
        return scores

    def _get_image_set(self):
        """Get set of available image filenames in local dir."""
        if not os.path.isdir(self.image_dir):
            print(f"  WARNING: image_dir not found: {self.image_dir}")
            return set()
        return set(os.listdir(self.image_dir))

    def load(self) -> datasets.Dataset:
        ds = datasets.load_dataset(self.hf_path, split=self.split)
        if self.max_samples:
            ds = ds.select(range(min(self.max_samples, len(ds))))
        self._stats["raw_count"] = len(ds)
        return ds

    def process(self) -> Dict[str, datasets.Dataset]:
        raw_ds = self.load()
        scores = self._load_scores() if self.filter_pass_rate else {}
        available_images = self._get_image_set()
        results = {}

        if "vqa" in self.tasks:
            samples = []
            for i in range(len(raw_ds)):
                example = raw_ds[i]

                # Match image from HF dataset to local file
                # VISA images have an 'id' field we can use
                vid = example.get("id", f"{i:06d}")

                # Hard case filter: only keep pass_rate < 1.0
                if self.filter_pass_rate and scores:
                    # VISA score IDs are like "visa_paper_000002"
                    prefix = "visa_paper" if "paper" in self.name else "visa_wiki"
                    score_id = f"{prefix}_{i:06d}"
                    pass_rate = scores.get(score_id, None)
                    if pass_rate is not None and pass_rate >= 1.0:
                        continue

                # Save image from HF dataset to local path if not already there
                # The local image_dir should already have images from the previous project
                # Try to find by index pattern
                image_path = self._find_image(example, i, available_images)
                if image_path is None:
                    continue

                sample = make_sample(
                    data_source=self.name,
                    question=example["question"],
                    image_path=image_path,
                    ground_truth=example["short_answer"],
                    task_type="vqa",
                    dataset=self.name,
                    split=self.split,
                    index=i,
                )
                samples.append(sample)

            self._stats["after_filter"] = len(samples)
            self._stats["vqa_count"] = len(samples)
            print(f"  {self.name} VQA: {self._stats['raw_count']} → {len(samples)} after filtering")

            if samples:
                results["vqa"] = datasets.Dataset.from_list(samples)

        return results

    def _find_image(self, example, idx, available_images):
        """Find image file in local directory."""
        # The local dir contains images directly (e.g., PMC1064093_00000.jpg)
        # Try listing all files and matching by index
        # Since images are pre-filtered (pass_rate < 1.0 only),
        # we need to save from HF dataset
        image = example.get("image", None)
        if image is None:
            return None

        # Use a deterministic filename based on dataset and index
        ext = ".jpg" if "paper" in self.name else ".png"
        fname = f"{self.name}_{idx:06d}{ext}"
        image_path = os.path.join(self.image_dir, fname)

        if fname in available_images:
            return image_path

        # Save from HF PIL Image
        os.makedirs(self.image_dir, exist_ok=True)
        image.save(image_path)
        available_images.add(fname)
        return image_path

    def get_stats(self) -> dict:
        return self._stats
