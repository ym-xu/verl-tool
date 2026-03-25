"""
Test MinerU GND/OCR samples with Qwen3-VL-8B to measure pass rate.
Only samples that the model can't reliably solve (pass_rate < 1.0) are kept.

Run on GPU server:
    # Test OCR samples
    python test_mineru_passrate.py --task ocr --max_samples 100  # quick test
    python test_mineru_passrate.py --task ocr                     # full run

    # Test GND samples
    python test_mineru_passrate.py --task gnd --max_samples 100
    python test_mineru_passrate.py --task gnd

    # Custom model/resolution
    python test_mineru_passrate.py --task ocr --model Qwen/Qwen3-VL-8B \
        --max_pixels 401408 --num_samples 8

Output: data/docseek/mineru_scores/{task}_scores.jsonl
    Each line: {"index": 0, "pass_rate": 0.375, "scores": [...], "element_type": "text", ...}

Then set in config.yaml:
    mineru_gnd_ocr:
      ocr_scores_file: "data/docseek/mineru_scores/ocr_scores.jsonl"
      gnd_scores_file: "data/docseek/mineru_scores/gnd_scores.jsonl"
      filter_pass_rate: true
"""
import fire
import os
import json
import re
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))
from verl_tool.workers.reward_manager.reward_score.doc_metrics import (
    compute_anls, compute_iou
)


SYSTEM_PROMPT = """You are a helpful assistant specialized in document understanding."""

OCR_PROMPT = "Look at this document image and read the text at region {bbox} in the document.\n\nPut your final answer within \\boxed{{}}."

GND_PROMPT = "Look at this document image and locate: {question}\n\nProvide the bounding box as [x1, y1, x2, y2] with normalized coordinates between 0 and 1. Put your final answer within \\boxed{{[x1,y1,x2,y2]}}."


def extract_boxed(text: str) -> str:
    """Extract answer from \\boxed{...}."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else text.strip()


def normalize_bbox_pixel_to_01(bbox, w, h):
    x1, y1, x2, y2 = bbox
    return [x1 / w, y1 / h, x2 / w, y2 / h]


def format_bbox_str(bbox):
    return f"[{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]"


def is_garbled(text, threshold=0.3):
    if not text:
        return True
    alnum = sum(c.isalnum() or c.isspace() for c in text)
    return alnum / len(text) < threshold


def build_samples(task, mineru_dirs, image_dirs, max_samples=None):
    """Build GND/OCR samples from MinerU output."""
    from PIL import Image

    samples = []
    for source_name, mineru_dir in mineru_dirs.items():
        image_dir = image_dirs.get(source_name)
        if not os.path.isdir(mineru_dir):
            continue

        files = sorted(os.listdir(mineru_dir))
        print(f"  Scanning {source_name}: {len(files)} files")

        for fname in tqdm(files, desc=f"Building {task} from {source_name}"):
            if max_samples and len(samples) >= max_samples:
                break

            doc_id = fname.replace(".json", "")

            # Find image
            img_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                p = os.path.join(image_dir, doc_id + ext)
                if os.path.exists(p):
                    img_path = p
                    break
            if not img_path:
                continue

            try:
                img = Image.open(img_path)
                w, h = img.size
                img.close()
            except Exception:
                continue

            try:
                elements = json.load(open(os.path.join(mineru_dir, fname)))
            except Exception:
                continue

            for elem in elements:
                if max_samples and len(samples) >= max_samples:
                    break

                etype = elem.get("type", "")
                text = elem.get("text", "").strip()
                bbox = elem.get("bbox")
                text_level = elem.get("text_level")
                is_title = text_level == 1

                if etype == "discarded" or not bbox or len(bbox) != 4:
                    continue

                bbox_norm = normalize_bbox_pixel_to_01(bbox, w, h)
                bbox_str = format_bbox_str(bbox_norm)

                if task == "ocr":
                    if etype not in ("text", "equation"):
                        continue
                    if not text or len(text) < 3 or len(text) > 200:
                        continue
                    if is_garbled(text):
                        continue

                    prompt = OCR_PROMPT.format(bbox=bbox_str)
                    samples.append({
                        "image_path": img_path,
                        "prompt": prompt,
                        "ground_truth": text,
                        "element_type": "title" if is_title else etype,
                        "source": source_name,
                        "doc_id": doc_id,
                        "bbox_str": bbox_str,
                    })

                elif task == "gnd":
                    # Main element GND
                    question = None
                    elem_label = etype
                    if etype == "table":
                        question = "Where is the table in this document?"
                    elif etype == "image":
                        question = "Where is the figure in this document?"
                    elif etype == "equation":
                        question = "Where is the equation in this document?"
                    elif is_title and text:
                        question = f'Where is the section titled "{text[:60]}" in this document?'
                        elem_label = "title"

                    if question:
                        samples.append({
                            "image_path": img_path,
                            "prompt": GND_PROMPT.format(question=question),
                            "ground_truth": bbox_str,
                            "element_type": elem_label,
                            "source": source_name,
                            "doc_id": doc_id,
                        })

                # Caption/footnote samples
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
                    for val in values:
                        val = val.strip()
                        if not val or len(val) < 3 or len(val) > 200 or is_garbled(val):
                            continue
                        if max_samples and len(samples) >= max_samples:
                            break

                        if task == "ocr":
                            prompt = OCR_PROMPT.format(bbox=bbox_str)
                            samples.append({
                                "image_path": img_path,
                                "prompt": prompt,
                                "ground_truth": val,
                                "element_type": field,
                                "source": source_name,
                                "doc_id": doc_id,
                                "bbox_str": bbox_str,
                            })
                        elif task == "gnd":
                            question = f'Where is the {label} that reads "{val[:50]}"?'
                            samples.append({
                                "image_path": img_path,
                                "prompt": GND_PROMPT.format(question=question),
                                "ground_truth": bbox_str,
                                "element_type": field,
                                "source": source_name,
                                "doc_id": doc_id,
                            })

    print(f"  Built {len(samples)} {task} samples")
    return samples


def run_inference(samples, task, model_name, num_samples, max_pixels, min_pixels,
                  output_file, batch_size=1):
    """Run model inference and compute pass rate."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    print(f"Loading model: {model_name}")
    print(f"Image resolution: max_pixels={max_pixels}, min_pixels={min_pixels}")
    processor = AutoProcessor.from_pretrained(
        model_name, max_pixels=max_pixels, min_pixels=min_pixels
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = []
    with open(output_file, 'w') as f:
        for i, sample in enumerate(tqdm(samples, desc=f"Scoring {task}")):
            image = Image.open(sample["image_path"])

            scores_list = []
            for trial in range(num_samples):
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": sample["prompt"]},
                    ]}
                ]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt"
                ).to(model.device)

                # Use temperature > 0 for diversity in pass@k
                with torch.no_grad():
                    if num_samples == 1:
                        output_ids = model.generate(**inputs, max_new_tokens=256)
                    else:
                        output_ids = model.generate(
                            **inputs, max_new_tokens=256,
                            temperature=0.7, do_sample=True
                        )

                generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
                prediction = processor.decode(generated_ids, skip_special_tokens=True)
                answer = extract_boxed(prediction)

                # Score
                if task == "ocr":
                    score = compute_anls(answer, [sample["ground_truth"]])
                    passed = score > 0  # ANLS > 0 means at least partially correct
                elif task == "gnd":
                    try:
                        # Parse bbox from strings like "[0.1, 0.2, 0.3, 0.4]"
                        import ast
                        pred_bbox = ast.literal_eval(answer)
                        gt_bbox = ast.literal_eval(sample["ground_truth"])
                        iou = compute_iou(pred_bbox, gt_bbox)
                        passed = iou >= 0.5
                    except Exception:
                        passed = False

                scores_list.append(1 if passed else 0)

            image.close()

            pass_rate = sum(scores_list) / len(scores_list)
            record = {
                "index": i,
                "pass_rate": round(pass_rate, 4),
                "correct_count": sum(scores_list),
                "n": len(scores_list),
                "element_type": sample["element_type"],
                "source": sample["source"],
                "doc_id": sample["doc_id"],
                "task": task,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)

            if i < 5:
                print(f"  [{i}] pr={pass_rate:.3f} type={sample['element_type']} "
                      f"gt={sample['ground_truth'][:50]}")

    # Summary
    from collections import Counter
    total = len(results)
    hard = sum(1 for r in results if r["pass_rate"] < 1.0)
    trivial = total - hard

    print(f"\n=== {task.upper()} Summary ===")
    print(f"Total: {total}, Hard (pr<1.0): {hard} ({hard/max(total,1):.1%}), "
          f"Trivial (pr=1.0): {trivial} ({trivial/max(total,1):.1%})")

    # By element type
    print(f"\nBy element type:")
    type_stats = {}
    for r in results:
        et = r["element_type"]
        if et not in type_stats:
            type_stats[et] = {"total": 0, "hard": 0}
        type_stats[et]["total"] += 1
        if r["pass_rate"] < 1.0:
            type_stats[et]["hard"] += 1

    for et, st in sorted(type_stats.items(), key=lambda x: -x[1]["total"]):
        pct = st["hard"] / st["total"] * 100
        print(f"  {et}: total={st['total']}, hard={st['hard']} ({pct:.1f}%)")

    print(f"\nSaved to: {output_file}")


def main(
    task: str = "ocr",
    model: str = "Qwen/Qwen3-VL-8B",
    num_samples: int = 8,
    max_samples: int = None,
    max_pixels: int = 401408,
    min_pixels: int = 3136,
    output_dir: str = "data/docseek/mineru_scores",
    data_root: str = "/data/151-1/users/yiming/dococr_data",
    shard_id: int = None,
    num_shards: int = 1,
):
    """
    Test MinerU GND/OCR samples with base model to measure pass rate.

    Multi-GPU parallel:
        for i in $(seq 0 7); do
            CUDA_VISIBLE_DEVICES=$i python test_mineru_passrate.py \\
                --task ocr --shard_id $i --num_shards 8 &
        done
        wait
        # Merge results:
        cat data/docseek/mineru_scores/ocr_scores_shard*.jsonl > data/docseek/mineru_scores/ocr_scores.jsonl

    Args:
        task: "ocr" or "gnd"
        model: HuggingFace model path
        num_samples: Number of inference runs per sample (pass@k)
        max_samples: Limit total samples (for testing)
        max_pixels: Must match training resolution (512*28*28=401408)
        min_pixels: Min pixels (4*28*28=3136)
        output_dir: Output directory for scores
        data_root: Root path for dococr_data
        shard_id: Shard index for multi-GPU parallel (0 to num_shards-1)
        num_shards: Total number of shards
    """
    if task not in ("ocr", "gnd"):
        print(f"Unknown task: {task}. Use 'ocr' or 'gnd'.")
        return

    mineru_dirs = {
        "docvqa": f"{data_root}/intermediate/mineru_output",
        "infovqa": f"{data_root}/intermediate/mineru_infovqa",
    }
    image_dirs = {
        "docvqa": f"{data_root}/train_source/DocVQA/docvqa_image",
        "infovqa": f"{data_root}/train_source/InfographicsVQA/images",
    }

    print(f"Task: {task}, Model: {model}, Samples per item: {num_samples}")
    print(f"Resolution: max_pixels={max_pixels}")

    # Build all samples first
    print("\n[Building samples...]")
    samples = build_samples(task, mineru_dirs, image_dirs, max_samples)

    if not samples:
        print("No samples to test.")
        return

    # Shard if requested
    if shard_id is not None and num_shards > 1:
        total = len(samples)
        shard_size = (total + num_shards - 1) // num_shards
        start = shard_id * shard_size
        end = min(start + shard_size, total)
        samples = samples[start:end]
        print(f"Shard {shard_id}/{num_shards}: samples [{start}:{end}] = {len(samples)}")
        output_file = os.path.join(output_dir, f"{task}_scores_shard{shard_id}.jsonl")
    else:
        output_file = os.path.join(output_dir, f"{task}_scores.jsonl")

    print(f"\n[Running inference on {len(samples)} samples...]")
    run_inference(samples, task, model, num_samples, max_pixels, min_pixels, output_file)


if __name__ == "__main__":
    fire.Fire(main)
