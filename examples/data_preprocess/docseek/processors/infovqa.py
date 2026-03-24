"""InfographicsVQA processor. Produces VQA samples only."""
import os
import json
import datasets
from typing import Dict
from .base import DatasetProcessor
from ..utils import save_image, make_sample


class InfoVQAProcessor(DatasetProcessor):

    def __init__(self, name: str, config: dict, output_dir: str):
        super().__init__(name, config, output_dir)
        self.hf_path = config["hf_path"]
        self.split = config.get("split", "train")
        self.max_samples = config.get("max_samples", None)
        self.hard_case_only = config.get("hard_case_only", False)
        self.hard_case_threshold = config.get("hard_case_threshold", 0.8)
        self.hard_case_scores_file = config.get("hard_case_scores_file", None)
        self.image_dir = os.path.join(output_dir, "images", name)
        self._hard_scores = None
        self._stats = {"name": name, "raw_count": 0, "after_hard_filter": 0, "vqa_count": 0}

    def _load_hard_scores(self):
        if self._hard_scores is not None:
            return
        if self.hard_case_scores_file and os.path.exists(self.hard_case_scores_file):
            self._hard_scores = {}
            with open(self.hard_case_scores_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    self._hard_scores[item["index"]] = item["score"]
        else:
            self._hard_scores = {}

    def load(self) -> datasets.Dataset:
        split_name = "train" if self.split == "train" else "validation"
        ds = datasets.load_dataset(self.hf_path, split=split_name)
        if self.max_samples:
            ds = ds.select(range(min(self.max_samples, len(ds))))
        self._stats["raw_count"] = len(ds)
        return ds

    def process(self) -> Dict[str, datasets.Dataset]:
        raw_ds = self.load()

        if self.hard_case_only:
            self._load_hard_scores()
            if self._hard_scores:
                raw_ds = raw_ds.filter(
                    lambda x, idx: self._hard_scores.get(idx, 1.0) < self.hard_case_threshold,
                    with_indices=True,
                    num_proc=8,
                )
                self._stats["after_hard_filter"] = len(raw_ds)
            else:
                self._stats["after_hard_filter"] = len(raw_ds)

        vqa_ds = raw_ds.map(
            self._make_vqa_sample,
            with_indices=True,
            remove_columns=raw_ds.column_names,
            num_proc=16,
            desc=f"{self.name} VQA",
        )
        self._stats["vqa_count"] = len(vqa_ds)
        return {"vqa": vqa_ds}

    def _make_vqa_sample(self, example, idx):
        image_path = os.path.join(self.image_dir, f"{self.split}_{idx}.png")
        save_image(example["image"], image_path)

        answers = example.get("answers", [example.get("answer", "")])
        if isinstance(answers, str):
            answers = [answers]

        return make_sample(
            data_source=self.name,
            question=example["question"],
            image_path=image_path,
            ground_truth=answers,
            task_type="vqa",
            dataset=self.name,
            split=self.split,
            index=idx,
        )

    def get_stats(self) -> dict:
        return self._stats
