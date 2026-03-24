"""
Preprocess document understanding datasets to parquet format for DocSeek training.
Supports: DocVQA, InfoVQA, VISA-Paper, VISA-Wiki
"""
import fire
import os
import datasets
from pathlib import Path

SYSTEM_PROMPT = """You are a helpful assistant specialized in document understanding.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "zoom_in", "description": "Zoom into a region of the document image to see finer details like small text, tables, or figures.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "Bounding box coordinates [x1, y1, x2, y2] as normalized values between 0 and 1, representing the region to zoom into.", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to zoom into. Index from 1 to the number of images. Choose 1 for the original document image."}}, "required": ["bbox_2d", "target_image"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

GUIDELINES = {
    'vqa': "Guidelines: Examine the document image carefully. If the text or relevant region is too small to read clearly, use the zoom_in tool to get a closer look. Answer the question based on the document content. Reason step by step, and put your final answer within \\boxed{}.",
    'gnd': "Guidelines: Examine the document image carefully. Locate the described element in the document. If needed, use the zoom_in tool to inspect regions more closely. Provide the bounding box as normalized coordinates [x1, y1, x2, y2] where each value is between 0 and 1. Put your final answer within \\boxed{[x1,y1,x2,y2]}.",
    'ocr': "Guidelines: Examine the document image carefully. If the text is too small or blurry to read clearly, use the zoom_in tool to get a closer look. Extract and transcribe the requested text exactly as it appears in the document. Put your final answer within \\boxed{}.",
}


def process_docvqa(local_dir: str, split: str = 'train', max_samples: int = None):
    """Process DocVQA dataset."""
    ds = datasets.load_dataset('lmms-lab/DocVQA', split='train' if split == 'train' else 'validation')
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def process_fn(example, idx):
        image_path = os.path.join(local_dir, 'images', 'docvqa', f'{split}_{idx}.png')
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        if not os.path.exists(image_path):
            example['image'].save(image_path)
        image_abs = os.path.abspath(image_path)
        question = example['question'] + f"\n\n{GUIDELINES['vqa']}"
        answers = example.get('answers', [example.get('answer', '')])
        if isinstance(answers, str):
            answers = [answers]
        return {
            "data_source": "docvqa",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"}
            ],
            "images": [{"image": image_abs}],
            "ability": "document_understanding",
            "reward_model": {"style": "rule", "ground_truth": answers},
            "extra_info": {
                "split": split,
                "index": idx,
                "task_type": "vqa",
                "dataset": "docvqa",
                "images": [image_abs],
            }
        }

    return ds.map(process_fn, with_indices=True, remove_columns=ds.column_names, num_proc=32)


def process_infovqa(local_dir: str, split: str = 'train', max_samples: int = None):
    """Process InfographicsVQA dataset."""
    ds = datasets.load_dataset('lmms-lab/InfographicsVQA', split='train' if split == 'train' else 'validation')
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def process_fn(example, idx):
        image_path = os.path.join(local_dir, 'images', 'infovqa', f'{split}_{idx}.png')
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        if not os.path.exists(image_path):
            example['image'].save(image_path)
        image_abs = os.path.abspath(image_path)
        question = example['question'] + f"\n\n{GUIDELINES['vqa']}"
        answers = example.get('answers', [example.get('answer', '')])
        if isinstance(answers, str):
            answers = [answers]
        return {
            "data_source": "infovqa",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"}
            ],
            "images": [{"image": image_abs}],
            "ability": "document_understanding",
            "reward_model": {"style": "rule", "ground_truth": answers},
            "extra_info": {
                "split": split,
                "index": idx,
                "task_type": "vqa",
                "dataset": "infovqa",
                "images": [image_abs],
            }
        }

    return ds.map(process_fn, with_indices=True, remove_columns=ds.column_names, num_proc=32)


def process_visa(local_dir: str, variant: str = 'paper', split: str = 'train', max_samples: int = None):
    """
    Process VISA dataset (Paper-VISA or Wiki-VISA).
    NOTE: Update dataset_path and field names based on actual VISA dataset format.
    """
    # TODO: Update with actual VISA dataset path and format
    dataset_path = f'VISA/{variant}'  # placeholder
    print(f"[WARNING] VISA-{variant} processing is a placeholder. Update dataset_path and field names.")
    return None


def main(
    local_dir: str = 'data/docseek',
    datasets_to_include: str = 'docvqa,infovqa',  # comma-separated
    max_samples_per_dataset: int = None,
    val_size: int = 100,
    seed: int = 42,
    version: str = None,
):
    """
    Prepare DocSeek training data.

    Usage:
        python prepare_train.py --local_dir=data/docseek --datasets_to_include=docvqa,infovqa
        python prepare_train.py --local_dir=data/docseek --datasets_to_include=docvqa --max_samples_per_dataset=1000 --version=small
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    dataset_list = [d.strip() for d in datasets_to_include.split(',')]
    all_datasets = []

    processors = {
        'docvqa': lambda: process_docvqa(str(local_dir), 'train', max_samples_per_dataset),
        'infovqa': lambda: process_infovqa(str(local_dir), 'train', max_samples_per_dataset),
        'visa_paper': lambda: process_visa(str(local_dir), 'paper', 'train', max_samples_per_dataset),
        'visa_wiki': lambda: process_visa(str(local_dir), 'wiki', 'train', max_samples_per_dataset),
    }

    for name in dataset_list:
        if name in processors:
            print(f"Processing {name}...")
            ds = processors[name]()
            if ds is not None:
                all_datasets.append(ds)
                print(f"  {name}: {len(ds)} samples")
        else:
            print(f"[WARNING] Unknown dataset: {name}. Available: {list(processors.keys())}")

    if not all_datasets:
        print("No datasets loaded. Exiting.")
        return

    merged = datasets.concatenate_datasets(all_datasets)
    merged = merged.shuffle(seed=seed)

    train_dataset, val_dataset = merged.train_test_split(test_size=val_size, seed=seed).values()

    print(f"\nTotal: {len(merged)} samples")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"\nSample:\n{train_dataset[0]}")

    output_dir = local_dir / version if version else local_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.to_parquet(str(output_dir / 'train.parquet'))
    val_dataset.to_parquet(str(output_dir / 'val.parquet'))
    print(f"\nSaved to {output_dir}/train.parquet and {output_dir}/val.parquet")


if __name__ == '__main__':
    fire.Fire(main)
