"""
DocSeek Training Data Preparation Pipeline.

Config-driven, supports iterative expansion.
Reads config.yaml, loads each data source via processors,
applies ratio sampling, and outputs train/val parquets.

Usage:
    python prepare_train.py
    python prepare_train.py --config custom_config.yaml
    python prepare_train.py --config config.yaml --override sources.docvqa.enabled=false
"""
import fire
import os
import datasets
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

from utils import load_config, save_stats
from processors import VISAProcessor, DocVQAProcessor, InfoVQAProcessor


PROCESSOR_MAP = {
    "paper_visa": VISAProcessor,
    "wiki_visa": VISAProcessor,
    "docvqa": DocVQAProcessor,
    "infovqa": InfoVQAProcessor,
    # "mineru_gnd_ocr": MinerUProcessor,  # Phase B
}


def sample_by_ratio(task_datasets: Dict[str, List[datasets.Dataset]],
                    task_ratio: dict,
                    vqa_source_ratio: dict,
                    seed: int = 42) -> datasets.Dataset:
    """
    Sample datasets according to task_ratio and vqa_source_ratio.
    Returns merged dataset.
    """
    all_samples = []
    stats = {}

    # Determine total dataset size (use all available data, apply ratios as weights)
    total_available = sum(
        sum(len(ds) for ds in ds_list)
        for ds_list in task_datasets.values()
    )

    for task_type, ds_list in task_datasets.items():
        task_weight = task_ratio.get(task_type, 0)
        if task_weight <= 0 or not ds_list:
            continue

        task_total = sum(len(ds) for ds in ds_list)
        target_count = int(total_available * task_weight)
        # Don't upsample: cap at available
        target_count = min(target_count, task_total)

        if task_type == "vqa" and vqa_source_ratio:
            # Sub-sample VQA by source
            for ds in ds_list:
                if len(ds) == 0:
                    continue
                source_name = ds[0]["data_source"]
                source_weight = vqa_source_ratio.get(source_name, 0)
                if source_weight <= 0:
                    continue
                source_target = int(target_count * source_weight)
                source_target = min(source_target, len(ds))
                sampled = ds.shuffle(seed=seed).select(range(source_target))
                all_samples.append(sampled)
                stats[f"{task_type}/{source_name}"] = source_target
        else:
            # For GND/OCR: merge all sources, sample proportionally
            merged_task = datasets.concatenate_datasets(ds_list)
            target_count = min(target_count, len(merged_task))
            sampled = merged_task.shuffle(seed=seed).select(range(target_count))
            all_samples.append(sampled)
            stats[task_type] = target_count

    if not all_samples:
        raise ValueError("No samples after ratio sampling. Check config.")

    merged = datasets.concatenate_datasets(all_samples)
    merged = merged.shuffle(seed=seed)

    print(f"\nSampling stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  TOTAL: {len(merged)}")

    return merged


def main(config: str = "config.yaml"):
    """Run the data preparation pipeline."""
    # Load config
    config_path = Path(config)
    if not config_path.exists():
        config_path = Path(__file__).parent / config
    cfg = load_config(str(config_path))

    output_dir = cfg["output"]["dir"]
    val_size = cfg["output"]["val_size"]
    seed = cfg["output"]["seed"]
    version = cfg["output"].get("version", None)

    os.makedirs(output_dir, exist_ok=True)

    # Process each enabled source
    task_datasets: Dict[str, List[datasets.Dataset]] = defaultdict(list)
    all_stats = {}

    for source_name, source_cfg in cfg["sources"].items():
        if not source_cfg.get("enabled", False):
            print(f"[SKIP] {source_name} (disabled)")
            continue

        processor_cls = PROCESSOR_MAP.get(source_name)
        if processor_cls is None:
            print(f"[SKIP] {source_name} (no processor)")
            continue

        print(f"\n[PROCESSING] {source_name}...")
        processor = processor_cls(
            name=source_name,
            config=source_cfg,
            output_dir=output_dir,
        )
        result = processor.process()
        all_stats[source_name] = processor.get_stats()

        for task_type, ds in result.items():
            task_datasets[task_type].append(ds)
            print(f"  {task_type}: {len(ds)} samples")

    if not any(task_datasets.values()):
        print("\nNo data produced. Check config sources.")
        return

    # Sample by ratio
    print("\n[SAMPLING] Applying task and source ratios...")
    merged = sample_by_ratio(
        task_datasets,
        cfg["task_ratio"],
        cfg.get("vqa_source_ratio", {}),
        seed=seed,
    )

    # Split train/val
    if val_size > 0 and len(merged) > val_size:
        splits = merged.train_test_split(test_size=val_size, seed=seed)
        train_ds = splits["train"]
        val_ds = splits["test"]
    else:
        train_ds = merged
        val_ds = None

    print(f"\nTrain: {len(train_ds)} samples")
    if val_ds:
        print(f"Val: {len(val_ds)} samples")

    # Save
    out_dir = Path(output_dir) / version if version else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds.to_parquet(str(out_dir / "train.parquet"))
    if val_ds:
        val_ds.to_parquet(str(out_dir / "val.parquet"))

    print(f"\nSaved to {out_dir}/")

    # Save stats
    all_stats["final"] = {
        "train_count": len(train_ds),
        "val_count": len(val_ds) if val_ds else 0,
        "task_distribution": dict(train_ds.to_pandas()["extra_info"].apply(lambda x: x["task_type"]).value_counts()),
        "source_distribution": dict(train_ds.to_pandas()["data_source"].value_counts()),
    }
    save_stats(all_stats, str(out_dir / "data_stats.json"))

    # Print sample
    print(f"\nSample (train[0]):")
    sample = train_ds[0]
    print(f"  data_source: {sample['data_source']}")
    print(f"  task_type: {sample['extra_info']['task_type']}")
    print(f"  question: {sample['prompt'][1]['content'][:100]}...")
    print(f"  ground_truth: {sample['reward_model']['ground_truth']}")


if __name__ == "__main__":
    fire.Fire(main)
