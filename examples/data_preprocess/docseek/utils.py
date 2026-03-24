"""Shared utilities for DocSeek data pipeline."""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_image(image, image_path: str) -> str:
    """Save PIL Image to disk, return absolute path."""
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if not os.path.exists(image_path):
        image.save(image_path)
    return os.path.abspath(image_path)


def normalize_bbox_pixel_to_01(bbox, image_width, image_height):
    """Convert pixel bbox [x1,y1,x2,y2] to normalized [0,1] coordinates."""
    x1, y1, x2, y2 = bbox
    return [
        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height,
    ]


def normalize_bbox_1000_to_01(bbox):
    """Convert MinerU 0-1000 bbox to normalized [0,1] coordinates."""
    return [v / 1000.0 for v in bbox]


def format_bbox_str(bbox):
    """Format bbox as string for ground truth: [x1, y1, x2, y2] with 4 decimals."""
    return f"[{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]"


def save_stats(stats: dict, output_path: str):
    """Save pipeline statistics to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Stats saved to {output_path}")


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


def make_sample(
    data_source: str,
    question: str,
    image_path: str,
    ground_truth,
    task_type: str,
    dataset: str,
    split: str = "train",
    index: int = 0,
    extra_fields: Optional[Dict] = None,
) -> Dict:
    """Create a unified training sample dict."""
    guideline = GUIDELINES.get(task_type, GUIDELINES['vqa'])
    image_abs = os.path.abspath(image_path)

    sample = {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\n{question}\n\n{guideline}"},
        ],
        "images": [{"image": image_abs}],
        "ability": "document_understanding",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": index,
            "task_type": task_type,
            "dataset": dataset,
            "images": [image_abs],
        },
    }
    if extra_fields:
        sample["extra_info"].update(extra_fields)
    return sample
