---
description: Scaffold a new tool implementation
argument-hint: <tool-name>
---
Create a new tool named "$ARGUMENTS" following the verl-tool pattern:

1. Create `verl_tool/servers/tools/$ARGUMENTS.py`
2. Inherit from `BaseTool` and use `@register_tool` decorator
3. Set `tool_type = "$ARGUMENTS"`
4. Implement: `get_usage_inst()`, `load_env()`, `save_env()`, `update_env()`, `delete_env()`
5. Create a test file at `verl_tool/servers/tests/test_${ARGUMENTS}_tool.py`

Reference `verl_tool/servers/tools/python_code.py` as an example implementation.
