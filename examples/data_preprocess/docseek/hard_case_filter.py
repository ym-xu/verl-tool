"""
Hard Case Filter: Run base Qwen3-VL on DocVQA/InfoVQA without zoom tool,
record per-sample ANLS scores, to identify samples the model struggles with.

Run on GPU server:
    python hard_case_filter.py --dataset docvqa
    python hard_case_filter.py --dataset infovqa
    python hard_case_filter.py --dataset docvqa --use_vllm --batch_size 16
    python hard_case_filter.py --dataset docvqa --model Qwen/Qwen3-VL-72B --threshold 0.95

Output: data/docseek/hard_cases/{dataset}_scores.jsonl
    Each line: {"index": 0, "score": 0.85, "prediction": "...", "ground_truth": [...]}

Then set in config.yaml:
    hard_case_only: true
    hard_case_scores_file: "data/docseek/hard_cases/docvqa_scores.jsonl"
"""
import fire
import os
import json
import re
import torch
from pathlib import Path
from tqdm import tqdm

# Import ANLS from our metrics
import sys
sys.path.insert(0, str(Path(__file__).parents[3]))
from verl_tool.workers.reward_manager.reward_score.doc_metrics import compute_anls


DATASET_MAP = {
    "docvqa": {"hf_path": "lmms-lab/DocVQA", "split": "train", "answer_key": "answers"},
    "infovqa": {"hf_path": "lmms-lab/InfographicsVQA", "split": "train", "answer_key": "answers"},
}

PROMPT_TEMPLATE = "Look at this document image and answer the question.\n\nQuestion: {question}\n\nPut your final answer within \\boxed{{}}."


def extract_boxed(text: str) -> str:
    """Extract answer from \\boxed{...}."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else text.strip()


def run_transformers(dataset_name, model_name, batch_size, max_samples, output_dir):
    """Run inference using transformers (works on any GPU setup)."""
    import datasets
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    ds_cfg = DATASET_MAP[dataset_name]
    ds = datasets.load_dataset(ds_cfg["hf_path"], split=ds_cfg["split"])
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_scores.jsonl")

    results = []
    with open(output_file, 'w') as f:
        for i in tqdm(range(len(ds)), desc=f"Scoring {dataset_name}"):
            example = ds[i]
            question = example["question"]
            image = example["image"]
            answers = example.get(ds_cfg["answer_key"], [])
            if isinstance(answers, str):
                answers = [answers]

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_TEMPLATE.format(question=question)},
                ]}
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=256)

            # Decode only the generated tokens
            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            prediction = processor.decode(generated_ids, skip_special_tokens=True)

            answer = extract_boxed(prediction)
            score = compute_anls(answer, answers)

            record = {
                "index": i,
                "score": round(score, 4),
                "prediction": answer,
                "ground_truth": answers,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)

            if i < 5:
                print(f"  [{i}] score={score:.3f} pred='{answer}' gt={answers[:2]}")

    # Summary
    scores = [r["score"] for r in results]
    total = len(scores)
    avg = sum(scores) / total if total > 0 else 0

    # Print summary for multiple thresholds
    print(f"\n=== {dataset_name} Summary ===")
    print(f"Total: {total}, Average ANLS: {avg:.4f}")
    for t in [0.5, 0.8, 0.9, 0.95]:
        hard = sum(1 for s in scores if s < t)
        print(f"  Threshold {t}: {hard} hard cases ({hard/total:.1%})")
    print(f"Saved to: {output_file}")


def run_vllm(dataset_name, model_name, batch_size, max_samples, output_dir):
    """Run inference using vLLM for faster batch processing."""
    import datasets
    from vllm import LLM, SamplingParams

    ds_cfg = DATASET_MAP[dataset_name]
    ds = datasets.load_dataset(ds_cfg["hf_path"], split=ds_cfg["split"])
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Loading vLLM model: {model_name}")
    llm = LLM(model=model_name, trust_remote_code=True, max_model_len=4096)
    sampling_params = SamplingParams(temperature=0, max_tokens=256)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_scores.jsonl")

    # Process in batches
    results = []
    for batch_start in tqdm(range(0, len(ds), batch_size), desc=f"Scoring {dataset_name}"):
        batch_end = min(batch_start + batch_size, len(ds))
        batch = ds.select(range(batch_start, batch_end))

        prompts = []
        for example in batch:
            prompt = PROMPT_TEMPLATE.format(question=example["question"])
            prompts.append(prompt)

        # Note: vLLM multimodal requires specific input format
        # This is a simplified version; may need adjustment for your vLLM version
        outputs = llm.generate(prompts, sampling_params)

        for j, output in enumerate(outputs):
            idx = batch_start + j
            example = ds[idx]
            prediction = extract_boxed(output.outputs[0].text)
            answers = example.get(ds_cfg["answer_key"], [])
            if isinstance(answers, str):
                answers = [answers]
            score = compute_anls(prediction, answers)

            results.append({
                "index": idx,
                "score": round(score, 4),
                "prediction": prediction,
                "ground_truth": answers,
            })

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    scores = [r["score"] for r in results]
    total = len(scores)
    hard = sum(1 for s in scores if s < 0.8)
    print(f"\n=== {dataset_name} Summary ===")
    print(f"Total: {total}, Hard (<0.8): {hard} ({hard/total:.2%})")
    print(f"Saved to: {output_file}")


def main(
    dataset: str = "docvqa",
    model: str = "Qwen/Qwen3-VL-8B",
    batch_size: int = 8,
    max_samples: int = None,
    output_dir: str = "data/docseek/hard_cases",
    use_vllm: bool = False,
    threshold: float = 0.9,
):
    """
    Score DocVQA/InfoVQA samples with base model to identify hard cases.

    Use a model at or above your training base model's capability level.
    Recommended: Qwen3-VL-8B (same as training base) with threshold 0.9,
    or Qwen3-VL-72B with threshold 0.95 for stricter filtering.

    Args:
        dataset: "docvqa" or "infovqa"
        model: HuggingFace model path (default: Qwen3-VL-8B)
        batch_size: Batch size for vLLM (ignored for transformers)
        max_samples: Limit number of samples (for testing)
        output_dir: Output directory for scores
        use_vllm: Use vLLM instead of transformers
        threshold: ANLS threshold for hard case (default 0.9)
    """
    if dataset not in DATASET_MAP:
        print(f"Unknown dataset: {dataset}. Available: {list(DATASET_MAP.keys())}")
        return

    print(f"Model: {model}, Threshold: {threshold}")
    print(f"Samples with ANLS < {threshold} will be considered hard cases")

    if use_vllm:
        run_vllm(dataset, model, batch_size, max_samples, output_dir)
    else:
        run_transformers(dataset, model, batch_size, max_samples, output_dir)


if __name__ == "__main__":
    fire.Fire(main)
