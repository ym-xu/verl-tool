---
paths:
  - "verl_tool/**/*.py"
  - "eval_service/**/*.py"
  - "examples/**/*.py"
---
# Code Style Rules

- Use type hints for all function signatures
- Logging via `import logging; logger = logging.getLogger(__file__)`, never `print()`
- Relative imports within packages: `from .base import BaseTool`
- Apache 2.0 license header for new files
- Pydantic BaseModel for API request/response validation
- Dataclasses with BaseConfig for configuration classes
