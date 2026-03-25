"""
DocSeek Training Data Preparation Pipeline.

Config-driven, supports iterative expansion.
Reads config.yaml, loads each data source via processors,
applies ratio sampling, and outputs train/val parquets.

Usage:
    python prepare_train.py
    python prepare_train.py --config custom_config.yaml
    python prepare_train.py --datasets_to_include=docvqa,infovqa,paper_visa,wiki_visa
"""
import fire
import os
import re
import datasets
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

from utils import load_config, save_stats
from processors import VISAProcessor, DocVQAProcessor, InfoVQAProcessor, MinerUProcessor


PROCESSOR_MAP = {
    "paper_visa": VISAProcessor,
    "wiki_visa": VISAProcessor,
    "docvqa": DocVQAProcessor,
    "infovqa": InfoVQAProcessor,
    "mineru_gnd_ocr": MinerUProcessor,
}


def resolve_config_vars(cfg: dict, data_root: str) -> dict:
    """Recursively replace ${data_root} in config values."""
    if isinstance(cfg, dict):
        return {k: resolve_config_vars(v, data_root) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [resolve_config_vars(v, data_root) for v in cfg]
    elif isinstance(cfg, str):
        return cfg.replace("${data_root}", data_root)
    return cfg


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
        target_count = min(target_count, task_total)

        if task_type == "vqa" and vqa_source_ratio:
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


def main(
    config: str = "config.yaml",
    datasets_to_include: str = None,
):
    """Run the data preparation pipeline.

    Args:
        config: Path to config YAML file.
        datasets_to_include: Comma-separated list of sources to include.
            If None, uses 'enabled' flag in config.
            E.g., "docvqa,infovqa,paper_visa,wiki_visa,mineru_gnd_ocr"
    """
    config_path = Path(config)
    if not config_path.exists():
        config_path = Path(__file__).parent / config
    cfg = load_config(str(config_path))

    # Resolve ${data_root} in all paths
    data_root = cfg.get("data_root", "")
    cfg = resolve_config_vars(cfg, data_root)

    output_dir = cfg["output"]["dir"]
    val_size = cfg["output"]["val_size"]
    seed = cfg["output"]["seed"]
    version = cfg["output"].get("version", None)

    os.makedirs(output_dir, exist_ok=True)

    # Determine which sources to process
    include_set = None
    if datasets_to_include:
        if isinstance(datasets_to_include, (list, tuple)):
            include_set = set(datasets_to_include)
        else:
            include_set = set(s.strip() for s in str(datasets_to_include).split(","))

    # Process each enabled source
    task_datasets: Dict[str, List[datasets.Dataset]] = defaultdict(list)
    all_stats = {}

    for source_name, source_cfg in cfg["sources"].items():
        if include_set:
            if source_name not in include_set:
                print(f"[SKIP] {source_name} (not in --datasets_to_include)")
                continue
        elif not source_cfg.get("enabled", False):
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
            if len(ds) > 0:
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
    }
    save_stats(all_stats, str(out_dir / "data_stats.json"))

    # Print sample
    print(f"\nSample (train[0]):")
    sample = train_ds[0]
    print(f"  data_source: {sample['data_source']}")
    print(f"  task_type: {sample['extra_info']['task_type']}")
    user_msg = sample['prompt'][1]['content'] if len(sample['prompt']) > 1 else ""
    print(f"  question: {user_msg[:100]}...")
    print(f"  ground_truth: {sample['reward_model']['ground_truth']}")


if __name__ == "__main__":
    fire.Fire(main)
