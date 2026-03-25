"""InfographicsVQA processor. Reads from local files + scores for hard case filtering."""
import os
import json
import datasets
from typing import Dict
from .base import DatasetProcessor
try:
    from ..utils import make_sample
except ImportError:
    from utils import make_sample


class InfoVQAProcessor(DatasetProcessor):

    def __init__(self, name: str, config: dict, output_dir: str):
        super().__init__(name, config, output_dir)
        self.image_dir = config["image_dir"]
        self.qa_file = config["qa_file"]
        self.max_samples = config.get("max_samples", None)
        self.filter_pass_rate = config.get("filter_pass_rate", False)
        self.scores_file = config.get("scores_file", None)
        self.scores_id_prefix = config.get("scores_id_prefix", None)
        self._stats = {"name": name, "raw_count": 0, "after_filter": 0, "vqa_count": 0}

    def _load_scores(self) -> Dict[str, float]:
        """Load pass_rate scores, filtered by id prefix."""
        scores = {}
        if not self.scores_file or not os.path.exists(self.scores_file):
            return scores
        with open(self.scores_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                sid = item["id"]
                if self.scores_id_prefix and not sid.startswith(self.scores_id_prefix):
                    continue
                scores[sid] = item["pass_rate"]
        print(f"  Loaded {len(scores)} scores (prefix={self.scores_id_prefix})")
        return scores

    def _load_qa(self):
        """Load QA data from local JSON file."""
        with open(self.qa_file, 'r') as f:
            data = json.load(f)
        return data["data"]

    def load(self) -> datasets.Dataset:
        return None

    def process(self) -> Dict[str, datasets.Dataset]:
        qa_data = self._load_qa()
        self._stats["raw_count"] = len(qa_data)

        scores = self._load_scores() if self.filter_pass_rate else {}

        samples = []
        for i, item in enumerate(qa_data):
            if self.max_samples and i >= self.max_samples:
                break

            # InfoVQA uses image_local_name field
            image_name = item["image_local_name"]
            image_path = os.path.join(self.image_dir, image_name)
            if not os.path.exists(image_path):
                continue

            # Hard case filter
            if self.filter_pass_rate and scores:
                score_id = f"{self.scores_id_prefix}_{i:05d}"
                pass_rate = scores.get(score_id, None)
                if pass_rate is not None and pass_rate >= 1.0:
                    continue

            answers = item.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]

            sample = make_sample(
                data_source="infovqa",
                question=item["question"],
                image_path=image_path,
                ground_truth=answers,
                task_type="vqa",
                dataset="infovqa",
                split="train",
                index=i,
            )
            samples.append(sample)

        self._stats["after_filter"] = len(samples)
        self._stats["vqa_count"] = len(samples)
        print(f"  {self.name}: {self._stats['raw_count']} → {len(samples)} after filtering")

        if not samples:
            return {"vqa": datasets.Dataset.from_dict({})}

        vqa_ds = datasets.Dataset.from_list(samples)
        return {"vqa": vqa_ds}

    def get_stats(self) -> dict:
        return self._stats
