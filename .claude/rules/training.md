---
paths:
  - "verl_tool/trainer/**"
  - "examples/train/**"
---
# Training Rules

- Configuration uses Hydra with YAML composition in `verl_tool/trainer/config/`
- Override config via CLI: `key.subkey=value`
- Training entry point: `python -m verl_tool.trainer.main_ppo`
- Always set `actor_rollout_ref.agent.enable_agent=True` for tool-calling
- Start tool server before training, kill after completion
- Set `VLLM_USE_V1=1` environment variable
- Action stop tokens go through temp file, not direct string
