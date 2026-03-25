# DocSeek Data Documentation

## Data Root

**Server**: `yiming@158.109.8.151:22345`
**Path**: `/data/151-1/users/yiming/dococr_data/`

> This is the shared data root from a previous project (dococr).
> DocSeek reads from it but writes output to its own directory.

**DocSeek output**: `/home/yiming/verl-tool/data/docseek/`

## Source Data Layout

```
/data/151-1/users/yiming/dococr_data/
├── train_source/
│   ├── VISA/
│   │   ├── paper_visa_image/    73,027 images (.jpg)
│   │   └── wiki_visa_image/     48,479 images (.png)
│   ├── DocVQA/
│   │   ├── docvqa_image/        9,738 images (.png)
│   │   ├── spdocvqa_qas/        QA JSONs (train/val/test)
│   │   └── ocr/                 OCR outputs
│   └── InfographicsVQA/
│       ├── images/              JPEG images
│       ├── infographicsVQA_train_v1.0.json
│       ├── infographicsVQA_val_v1.0_withQT.json
│       ├── infographicsVQA_test_v1.0.json
│       └── ocr/
├── eval_source/
│   ├── OCRBench/        405MB
│   ├── OCRBench_v2/     5.4GB
│   ├── PaperVISA/       805MB (test split)
│   ├── WikiVISA/        3.3GB (test split)
│   └── WildDoc/         85GB (DocVQA, ChartQA, TableVQA subsets)
├── intermediate/
│   ├── mineru_output/           12,767 JSONs (DocVQA documents)
│   └── mineru_infovqa/          5,485 JSONs (InfoVQA documents)
└── pipeline/
    └── scores/Qwen3-VL-8B-Instruct/
        ├── vqa_scores.jsonl         294,590 records (DocVQA + InfoVQA + others)
        ├── paper_visa_scores_v2.jsonl  86,048 records
        ├── wiki_visa_scores_v2.jsonl   82,232 records
        ├── grounding_scores.jsonl      24,834 records
        └── ocr_scores.jsonl            25,896 records
```

## QA Data Formats

### DocVQA (`spdocvqa_qas/train_v1.0_withQT.json`)
```json
{
  "dataset_name": "SP-DocVQA",
  "data": [
    {
      "questionId": 337,
      "question": "what is the date mentioned in this letter?",
      "image": "documents/xnbl0037_1.png",
      "answers": ["1/8/93"],
      "data_split": "train"
    }
  ]
}
```
Image field `"documents/xnbl0037_1.png"` → local file `docvqa_image/xnbl0037_1.png`.

### InfographicsVQA (`infographicsVQA_train_v1.0.json`)
```json
{
  "data": [
    {
      "questionId": 65718,
      "question": "Which type of fonts offer better readability?",
      "image_local_name": "20471.jpeg",
      "answers": ["serif fonts"],
      "data_split": "train"
    }
  ]
}
```
Image field `"image_local_name"` maps directly to `images/20471.jpeg`.

### VISA (Paper-VISA / Wiki-VISA)
Loaded from HuggingFace: `MrLight/paper-visa`, `MrLight/wiki-visa`.
Local images in `paper_visa_image/` and `wiki_visa_image/` are pre-filtered
(only pass_rate < 1.0 images from previous project).

HF fields: `question`, `short_answer`, `bounding_box` (4 ints), `image` (PIL).

## MinerU Output Format

Each JSON file contains a list of parsed document elements:
```json
[
  {
    "type": "text",           // text | table | image | equation | discarded
    "text": "Section Title",
    "text_level": 1,          // 1 = heading (optional)
    "bbox": [x1, y1, x2, y2], // pixel coordinates
    "page_idx": 0,
    // Optional fields on table/image elements:
    "table_caption": ["Caption text"],
    "image_caption": ["Figure caption"],
    "table_footnote": ["Footnote text"],
    "image_footnote": ["Note text"],
    "table_body": "...",
    "img_path": "..."
  }
]
```

All keys found: `type`, `text`, `bbox`, `page_idx`, `text_level`, `text_format`,
`table_caption`, `image_caption`, `table_footnote`, `image_footnote`,
`table_body`, `img_path`.

MinerU filename (e.g., `ffbf0023_4.json`) matches image filename (`ffbf0023_4.png`).

## Hard Case Filter Scores

Format (pass@8 with Qwen3-VL-8B-Instruct at default resolution):
```json
{"id": "docvqa_train_00010", "model": "Qwen3-VL-8B-Instruct",
 "pass_rate": 1.0, "correct_count": 8, "n": 8,
 "difficulty": "trivial", "task_type": "vqa"}
```

**Hard case statistics (pass_rate < 1.0):**

| Dataset | Total | Hard | Rate |
|---------|-------|------|------|
| DocVQA | 39,463 | 1,624 | 4.1% |
| InfoVQA | 23,946 | 6,822 | 28.5% |
| Paper-VISA | 86,048 | 73,027 | 84.9% |
| Wiki-VISA | 82,232 | 48,326 | 58.8% |

> Note: Scores were measured at default Qwen resolution (~1M pixels),
> not at our training resolution (401K pixels). Hard cases at 1M resolution
> will be even harder at 401K.

## DocSeek Pipeline Output

Config: `examples/data_preprocess/docseek/config.yaml`

```bash
# Generate training data
cd examples/data_preprocess/docseek
python prepare_train.py

# Generate eval data
python prepare_eval.py --benchmarks all
```

Output: `/home/yiming/verl-tool/data/docseek/v1/`
```
data/docseek/v1/
├── train.parquet      # Training data (VQA + GND + OCR, ratio-sampled)
├── val.parquet        # Validation data (100 samples from train)
└── data_stats.json    # Pipeline statistics
```

## Notes

- **Do NOT modify** files under `/data/151-1/users/yiming/dococr_data/` — shared with other projects.
- DocSeek only reads from source data; all outputs go to `data/docseek/`.
- VISA images: `paper_visa_image/` only contains hard case images (73K of 86K).
  `wiki_visa_image/` contains 48K of 82K.
- MinerU was run on DocVQA (12,767 docs) and InfoVQA (5,485 docs) only.
  No MinerU output for VISA yet.
