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
