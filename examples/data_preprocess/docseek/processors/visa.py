"""
VISA (Paper-VISA / Wiki-VISA) processor.
Produces VQA + GND samples from VISA data.
Does NOT produce OCR (would allow reward hacking).
"""
import os
import datasets
from typing import Dict
from .base import DatasetProcessor
from ..utils import save_image, normalize_bbox_pixel_to_01, format_bbox_str, make_sample


class VISAProcessor(DatasetProcessor):

    def __init__(self, name: str, config: dict, output_dir: str):
        super().__init__(name, config, output_dir)
        self.hf_path = config["hf_path"]
        self.split = config.get("split", "train")
        self.max_samples = config.get("max_samples", None)
        self.tasks = config.get("tasks", ["vqa", "gnd"])
        self.image_dir = os.path.join(output_dir, "images", name)
        self._stats = {"name": name, "raw_count": 0, "vqa_count": 0, "gnd_count": 0}

    def load(self) -> datasets.Dataset:
        ds = datasets.load_dataset(self.hf_path, split=self.split)
        if self.max_samples:
            ds = ds.select(range(min(self.max_samples, len(ds))))
        self._stats["raw_count"] = len(ds)
        return ds

    def process(self) -> Dict[str, datasets.Dataset]:
        raw_ds = self.load()
        results = {}

        if "vqa" in self.tasks:
            vqa_ds = raw_ds.map(
                self._make_vqa_sample,
                with_indices=True,
                remove_columns=raw_ds.column_names,
                num_proc=16,
                desc=f"{self.name} VQA",
            )
            results["vqa"] = vqa_ds
            self._stats["vqa_count"] = len(vqa_ds)

        if "gnd" in self.tasks:
            # Filter out samples without valid bounding_box
            gnd_raw = raw_ds.filter(
                lambda x: x.get("bounding_box") is not None and len(x["bounding_box"]) == 4,
                num_proc=8,
            )
            gnd_ds = gnd_raw.map(
                self._make_gnd_sample,
                with_indices=True,
                remove_columns=gnd_raw.column_names,
                num_proc=16,
                desc=f"{self.name} GND",
            )
            results["gnd"] = gnd_ds
            self._stats["gnd_count"] = len(gnd_ds)

        return results

    def _make_vqa_sample(self, example, idx):
        image_path = os.path.join(self.image_dir, f"{self.split}_{idx}.png")
        save_image(example["image"], image_path)
        return make_sample(
            data_source=self.name,
            question=example["question"],
            image_path=image_path,
            ground_truth=example["short_answer"],
            task_type="vqa",
            dataset=self.name,
            split=self.split,
            index=idx,
        )

    def _make_gnd_sample(self, example, idx):
        image_path = os.path.join(self.image_dir, f"{self.split}_{idx}.png")
        save_image(example["image"], image_path)

        # Get image size for bbox normalization
        w, h = example["image"].size
        bbox_norm = normalize_bbox_pixel_to_01(example["bounding_box"], w, h)
        bbox_str = format_bbox_str(bbox_norm)

        question = f"Where in the document is the evidence for: {example['question']}"

        return make_sample(
            data_source=self.name,
            question=question,
            image_path=image_path,
            ground_truth=bbox_str,
            task_type="gnd",
            dataset=self.name,
            split=self.split,
            index=idx,
        )

    def get_stats(self) -> dict:
        return self._stats
