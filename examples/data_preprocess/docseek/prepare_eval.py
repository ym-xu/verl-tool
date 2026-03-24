"""
Prepare evaluation benchmark data as parquet files.
Each benchmark gets its own parquet with data_source matching reward manager routing.

Usage:
    python prepare_eval.py --benchmarks docvqa,infovqa,textvqa,paper_visa,wiki_visa
    python prepare_eval.py --benchmarks all
    python prepare_eval.py --benchmarks paper_visa --output_dir data/docseek/eval
"""
import fire
import os
import datasets
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import save_image, make_sample, SYSTEM_PROMPT, GUIDELINES


ALL_BENCHMARKS = [
    "docvqa", "infovqa", "textvqa",
    "paper_visa", "wiki_visa",
    "ocrbench", "ocrbench_v2",
    "wilddoc_docvqa", "wilddoc_chartqa", "wilddoc_tablevqa",
]


def prepare_docvqa_test(output_dir: str):
    ds = datasets.load_dataset("lmms-lab/DocVQA", split="test")
    image_dir = os.path.join(output_dir, "images", "docvqa_test")

    def process(example, idx):
        image_path = os.path.join(image_dir, f"test_{idx}.png")
        save_image(example["image"], image_path)
        answers = example.get("answers", [example.get("answer", "")])
        if isinstance(answers, str):
            answers = [answers]
        return make_sample("docvqa", example["question"], image_path, answers,
                           "vqa", "docvqa", "test", idx)

    result = ds.map(process, with_indices=True, remove_columns=ds.column_names, num_proc=16)
    out = os.path.join(output_dir, "docvqa_test.parquet")
    result.to_parquet(out)
    print(f"DocVQA test: {len(result)} → {out}")


def prepare_infovqa_test(output_dir: str):
    ds = datasets.load_dataset("lmms-lab/InfographicsVQA", split="test")
    image_dir = os.path.join(output_dir, "images", "infovqa_test")

    def process(example, idx):
        image_path = os.path.join(image_dir, f"test_{idx}.png")
        save_image(example["image"], image_path)
        answers = example.get("answers", [example.get("answer", "")])
        if isinstance(answers, str):
            answers = [answers]
        return make_sample("infovqa", example["question"], image_path, answers,
                           "vqa", "infovqa", "test", idx)

    result = ds.map(process, with_indices=True, remove_columns=ds.column_names, num_proc=16)
    out = os.path.join(output_dir, "infovqa_test.parquet")
    result.to_parquet(out)
    print(f"InfoVQA test: {len(result)} → {out}")


def prepare_textvqa_val(output_dir: str):
    ds = datasets.load_dataset("facebook/textvqa", split="validation")
    image_dir = os.path.join(output_dir, "images", "textvqa_val")

    def process(example, idx):
        image_path = os.path.join(image_dir, f"val_{idx}.png")
        save_image(example["image"], image_path)
        answers = example.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        return make_sample("textvqa", example["question"], image_path, answers,
                           "vqa", "textvqa", "test", idx)

    result = ds.map(process, with_indices=True, remove_columns=ds.column_names, num_proc=16)
    out = os.path.join(output_dir, "textvqa_val.parquet")
    result.to_parquet(out)
    print(f"TextVQA val: {len(result)} → {out}")


def prepare_visa_test(output_dir: str, variant: str = "paper"):
    name = f"{variant}_visa"
    hf_path = f"MrLight/{variant}-visa"
    ds = datasets.load_dataset(hf_path, split="test")
    image_dir = os.path.join(output_dir, "images", f"{name}_test")

    def process(example, idx):
        image_path = os.path.join(image_dir, f"test_{idx}.png")
        save_image(example["image"], image_path)
        return make_sample(name, example["question"], image_path,
                           example["short_answer"], "vqa", name, "test", idx)

    result = ds.map(process, with_indices=True, remove_columns=ds.column_names, num_proc=16)
    out = os.path.join(output_dir, f"{name}_test.parquet")
    result.to_parquet(out)
    print(f"{name} test: {len(result)} → {out}")


def prepare_ocrbench(output_dir: str):
    """OCRBench v1 — requires manual download from MultimodalOCR repo."""
    # TODO: implement once OCRBench data format is confirmed
    print("[TODO] OCRBench: requires manual data download from https://github.com/Yuliang-Liu/MultimodalOCR")


def prepare_ocrbench_v2(output_dir: str):
    """OCRBench v2 — requires manual download."""
    # TODO: implement once data format is confirmed
    print("[TODO] OCRBench_v2: requires manual data download from https://github.com/Yuliang-Liu/MultimodalOCR")


def prepare_wilddoc(output_dir: str, subset: str = "docvqa"):
    """WildDoc benchmarks — requires manual download from bytedance/WildDoc."""
    # TODO: implement once data format is confirmed
    print(f"[TODO] WildDoc-{subset}: requires manual data download from https://github.com/bytedance/WildDoc")


BENCHMARK_HANDLERS = {
    "docvqa": prepare_docvqa_test,
    "infovqa": prepare_infovqa_test,
    "textvqa": prepare_textvqa_val,
    "paper_visa": lambda d: prepare_visa_test(d, "paper"),
    "wiki_visa": lambda d: prepare_visa_test(d, "wiki"),
    "ocrbench": prepare_ocrbench,
    "ocrbench_v2": prepare_ocrbench_v2,
    "wilddoc_docvqa": lambda d: prepare_wilddoc(d, "docvqa"),
    "wilddoc_chartqa": lambda d: prepare_wilddoc(d, "chartqa"),
    "wilddoc_tablevqa": lambda d: prepare_wilddoc(d, "tablevqa"),
}


def main(
    benchmarks: str = "docvqa,infovqa,paper_visa,wiki_visa",
    output_dir: str = "data/docseek/eval",
):
    """Prepare evaluation data."""
    os.makedirs(output_dir, exist_ok=True)

    if benchmarks == "all":
        benchmark_list = ALL_BENCHMARKS
    else:
        benchmark_list = [b.strip() for b in benchmarks.split(",")]

    for bench in benchmark_list:
        handler = BENCHMARK_HANDLERS.get(bench)
        if handler:
            print(f"\n[PREPARING] {bench}...")
            handler(output_dir)
        else:
            print(f"[SKIP] Unknown benchmark: {bench}. Available: {list(BENCHMARK_HANDLERS.keys())}")


if __name__ == "__main__":
    fire.Fire(main)
