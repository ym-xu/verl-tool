# VerlTool - Tool-Agent Training Framework

## Quick Reference

```bash
# Install (UV)
git submodule update --init --recursive
uv sync && source .venv/bin/activate
uv pip install -e verl
uv pip install -e ".[vllm,acecoder,torl,search_tool]"

# Start tool server
python -m verl_tool.servers.serve --host localhost --port 5500 --tool_type python_code --workers_per_tool 8

# Run training
python -m verl_tool.trainer.main_ppo algorithm.adv_estimator=grpo actor_rollout_ref.agent.enable_agent=True ...

# Test a tool
python -m verl_tool.servers.tests.test_python_code_tool python --url=http://localhost:5500/get_observation
```

## Architecture

- **verl/** - RL framework (git submodule, do NOT modify directly)
- **verl_tool/servers/** - FastAPI tool servers with `BaseTool` + `@register_tool`
- **verl_tool/trainer/** - Hydra-based PPO/GRPO training (entry: `main_ppo.py`)
- **verl_tool/workers/** - Reward managers and rollout workers
- **verl_tool/agent_loop/** - Multi-turn agent-tool interaction loop
- **examples/train/** - Training recipes (math_tir, search_r1, acecoder, etc.)
- **benchmarks/** - External benchmark repos (git submodules)
- **eval_service/** - Evaluation API service

## Conventions

- Python 3.10, type hints throughout
- Tools inherit `BaseTool`, use `@register_tool` decorator, file in `verl_tool/servers/tools/`
- Reward managers in `verl_tool/workers/reward_manager/`
- Logging: `import logging; logger = logging.getLogger(__file__)`
- Config: Hydra YAML composition, override via `dot.notation=value`
- Apache 2.0 license headers (Bytedance Ltd)

## Gotchas

- Always run `git submodule update --init --recursive` after clone/pull
- Set `VLLM_USE_V1=1` in training scripts
- `enable_agent=True` required for tool-calling training
- Python code tool uses firejail sandboxing - check availability
- Action stop tokens passed via temp file, not direct string
- FSDP2 may cause OOM, fallback to FSDP if needed
- Copy `.env.example` to `.env` and fill in API keys before running

---

## DocSeek Project

### Motivation

VLMs struggle with document images due to: (1) low-quality scans, (2) large images that must be compressed (InfoVQA, Wiki full-page screenshots). Three document tasks mutually reinforce each other:
- **VQA** → "why to look" (understanding intent)
- **GND (Grounding)** → "where to look" (spatial localization)
- **OCR** → "what to read" (text recognition)

### Approach

Train Qwen3-VL (4B→8B) with a **zoom tool** via GRPO, teaching the model to actively explore document images across all three tasks. Based on pixel_reasoner architecture.

### Image Resolution (Critical)

**`max_pixels = 512*28*28 = 401408` (~400K pixels)** — must be consistent everywhere.

This compresses large documents so small text becomes hard to read → zoom tool becomes useful.
Pixel_reasoner uses the Qwen default (1280*28*28 ≈ 1M); we intentionally use lower resolution.

| Dataset | Original | After 400K compress | Zoom value |
|---------|----------|-------------------|------------|
| InfoVQA | ~1500×2500 (3.75M) | ~490×820 (9x) | Very high |
| Wiki-VISA | 980×3920 (3.8M) | ~317×1267 (9.5x) | Very high |
| DocVQA | ~1000×1400 (1.4M) | ~535×750 (3.5x) | Medium |
| Paper-VISA | ~400×600 (240K) | No compression | Low |

Must match across: `config.yaml`, `train_qwen3vl_4b.sh`, `hard_case_filter.py`, eval scripts.

### DocSeek Files

```bash
# Tool server
python -m verl_tool.servers.serve --tool_type docseek --port 5500 --workers_per_tool 4

# Test tool
python -m verl_tool.servers.tests.test_docseek_tool direct
python -m verl_tool.servers.tests.test_docseek_tool api --url=http://localhost:5500/get_observation

# Data prep
python examples/data_preprocess/docseek/prepare_train.py --datasets_to_include=docvqa,infovqa

# Train (4B local validation)
bash examples/train/docseek/train_qwen3vl_4b.sh
```

| Component | File |
|-----------|------|
| Zoom tool | `verl_tool/servers/tools/docseek.py` |
| Reward manager | `verl_tool/workers/reward_manager/docseek.py` |
| Training metrics | `verl_tool/workers/reward_manager/reward_score/doc_metrics.py` |
| Eval: DocVQA/InfoVQA ANLS | `reward_score/eval_benchmarks/docvqa_anls.py` |
| Eval: TextVQA accuracy | `reward_score/eval_benchmarks/textvqa_accuracy.py` |
| Eval: OCRBench v1 | `reward_score/eval_benchmarks/ocrbench.py` |
| Eval: OCRBench v2 | `reward_score/eval_benchmarks/ocrbench_v2.py` |
| Eval: VISA Paper/Wiki | `reward_score/eval_benchmarks/visa_eval.py` |
| Eval: WildDoc | `reward_score/eval_benchmarks/wilddoc_eval.py` |
| Data prep | `examples/data_preprocess/docseek/prepare_train.py` |
| Training (4B) | `examples/train/docseek/train_qwen3vl_4b.sh` |
| Tool test | `verl_tool/servers/tests/test_docseek_tool.py` |

### Training Data
- VQA: DocVQA, InfoVQA, VISA-Paper, VISA-Wiki
- GND/OCR: Generated from MinerU/PaddleOCR parsing

### Evaluation Benchmarks
Reward manager routes by `data_source` to official metrics automatically.

| Benchmark | Metric | data_source key |
|-----------|--------|-----------------|
| DocVQA | ANLS (τ=0.5) | `docvqa` |
| InfographicsVQA | ANLS (τ=0.5) | `infovqa` |
| TextVQA | VQA 10-annotator accuracy | `textvqa` |
| Paper-VISA | Token subsequence match | `paper_visa` / `visa_paper` |
| Wiki-VISA | Token subsequence match | `wiki_visa` / `visa_wiki` |
| OCRBench | Substring containment | `ocrbench` |
| OCRBench_v2 | Task-specific (30 types) | `ocrbench_v2` |
| WildDoc-DocVQA | ANLS | `wilddoc_docvqa` |
| WildDoc-ChartQA | Relaxed accuracy | `wilddoc_chartqa` |
| WildDoc-TableVQA | Subset-specific | `wilddoc_tablevqa` |

### Development Strategy
1. Validate pipeline on Qwen3-VL-4B locally
2. Scale to Qwen3-VL-8B on RunPod

### Reference (pixel_reasoner)
- Tool: `verl_tool/servers/tools/pixel_reasoner.py`
- Reward: `verl_tool/workers/reward_manager/pixel_reasoner.py`
- Data prep: `examples/data_preprocess/pixel_reasoner/prepare_train.py`
- Training: `examples/train/pixel_reasoner/train_3b.sh`
