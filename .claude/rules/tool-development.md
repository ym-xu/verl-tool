---
paths:
  - "verl_tool/servers/tools/**/*.py"
---
# Tool Development Rules

When creating or modifying tools:

1. Inherit from `BaseTool` in `verl_tool/servers/tools/base.py`
2. Use `@register_tool` decorator for auto-discovery
3. Set `tool_type` class attribute matching the filename
4. Implement required methods: `get_usage_inst()`, `load_env()`, `save_env()`, `update_env()`, `delete_env()`
5. Test via: `python -m verl_tool.servers.tests.test_{tool_name}_tool`
6. Tool server runs on FastAPI with consistent hashing on trajectory_ids
