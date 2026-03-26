"""
Upload filtered DocSeek VQA training data to HuggingFace (private dataset).

This script:
1. Reads the training parquet
2. Copies only the referenced images into a self-contained directory
3. Updates image paths to be relative
4. Uploads to HuggingFace as a private dataset

Usage:
    # First generate the data:
    python prepare_train.py --datasets_to_include=paper_visa,wiki_visa,docvqa,infovqa

    # Then upload:
    python upload_to_hf.py --parquet data/docseek/v3_vqa_only/train.parquet \
                           --repo_id ym-xu/docseek-vqa-train \
                           --private
"""
import fire
import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def main(
    parquet: str = "data/docseek/v3_vqa_only/train.parquet",
    val_parquet: str = None,
    repo_id: str = "ym-xu/docseek-vqa-train",
    private: bool = True,
    staging_dir: str = "data/docseek/hf_upload_staging",
):
    """
    Package and upload DocSeek data to HuggingFace.

    Args:
        parquet: Path to training parquet file
        val_parquet: Path to validation parquet (auto-detected if not specified)
        repo_id: HuggingFace repo ID
        private: Whether to create a private dataset
        staging_dir: Temporary directory for staging files
    """
    from huggingface_hub import HfApi

    parquet_path = Path(parquet)
    if not parquet_path.exists():
        print(f"Error: {parquet_path} not found")
        return

    # Auto-detect val parquet
    if val_parquet is None:
        val_path = parquet_path.parent / "val.parquet"
        if val_path.exists():
            val_parquet = str(val_path)

    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    images_dir = staging / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"[1/4] Reading parquet: {parquet_path}")
    df_train = pd.read_parquet(parquet_path)
    print(f"  Train samples: {len(df_train)}")

    df_val = None
    if val_parquet and Path(val_parquet).exists():
        df_val = pd.read_parquet(val_parquet)
        print(f"  Val samples: {len(df_val)}")

    # Collect all image paths
    all_dfs = [("train", df_train)]
    if df_val is not None:
        all_dfs.append(("val", df_val))

    print(f"\n[2/4] Copying images to staging directory...")
    image_map = {}  # old_path -> new_relative_path
    copied = 0
    skipped = 0

    for split_name, df in all_dfs:
        for idx in tqdm(range(len(df)), desc=f"Processing {split_name}"):
            row = df.iloc[idx]
            images = row.get("images", [])

            # Handle numpy array
            if hasattr(images, 'tolist'):
                images = images.tolist()
            if not isinstance(images, list):
                continue

            for img_entry in images:
                if isinstance(img_entry, dict):
                    old_path = img_entry.get("image", "")
                elif isinstance(img_entry, str):
                    old_path = img_entry
                else:
                    continue

                if not old_path or old_path in image_map:
                    continue

                if not os.path.exists(old_path):
                    skipped += 1
                    continue

                # Create relative path: images/{source}/{filename}
                # Determine source from path
                fname = os.path.basename(old_path)
                if "paper_visa" in old_path or "VISA/paper" in old_path:
                    sub = "paper_visa"
                elif "wiki_visa" in old_path or "VISA/wiki" in old_path:
                    sub = "wiki_visa"
                elif "DocVQA" in old_path or "docvqa" in old_path:
                    sub = "docvqa"
                elif "InfographicsVQA" in old_path or "infovqa" in old_path:
                    sub = "infovqa"
                else:
                    sub = "other"

                rel_path = f"images/{sub}/{fname}"
                dest = staging / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)

                if not dest.exists():
                    shutil.copy2(old_path, dest)
                    copied += 1

                image_map[old_path] = rel_path

    print(f"  Copied {copied} images, skipped {skipped}")

    # Update parquet with relative paths
    print(f"\n[3/4] Updating image paths in parquet...")
    for split_name, df in all_dfs:
        new_images_col = []
        new_extra_col = []
        for idx in range(len(df)):
            row = df.iloc[idx]

            # Update images field
            images = row.get("images", [])
            if hasattr(images, 'tolist'):
                images = images.tolist()
            new_images = []
            for img_entry in (images if isinstance(images, list) else []):
                if isinstance(img_entry, dict):
                    old = img_entry.get("image", "")
                    new_images.append({"image": image_map.get(old, old)})
                else:
                    new_images.append(img_entry)
            new_images_col.append(new_images)

            # Update extra_info.images field
            extra = row.get("extra_info", {})
            if isinstance(extra, dict):
                extra = dict(extra)
                ei = extra.get("images", [])
                if hasattr(ei, 'tolist'):
                    ei = ei.tolist()
                if isinstance(ei, list):
                    extra["images"] = [image_map.get(p, p) for p in ei]
            new_extra_col.append(extra)

        df["images"] = new_images_col
        df["extra_info"] = new_extra_col

    # Save updated parquets to staging
    train_out = staging / "train.parquet"
    df_train.to_parquet(str(train_out))
    print(f"  Saved {train_out}")

    if df_val is not None:
        val_out = staging / "val.parquet"
        df_val.to_parquet(str(val_out))
        print(f"  Saved {val_out}")

    # Also copy config snapshot if exists
    config_snapshot = parquet_path.parent / "config_snapshot.yaml"
    if config_snapshot.exists():
        shutil.copy2(config_snapshot, staging / "config_snapshot.yaml")

    # Upload to HuggingFace
    print(f"\n[4/4] Uploading to {repo_id}...")
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"  Repo creation: {e}")

    # Upload entire staging directory
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload DocSeek VQA training data (0 < pass_rate < 1.0)",
    )

    print(f"\n  Done! Dataset at: https://huggingface.co/datasets/{repo_id}")

    # Print summary
    total_size = sum(
        f.stat().st_size for f in staging.rglob("*") if f.is_file()
    ) / (1024 ** 3)
    print(f"  Total size: {total_size:.2f} GB")
    print(f"  Train: {len(df_train)} samples")
    if df_val is not None:
        print(f"  Val: {len(df_val)} samples")


if __name__ == "__main__":
    fire.Fire(main)
