# Project Guidelines

## Code Style

### Imports
- Keep imports at the module level, not inside functions or methods
- Only use local imports when there's a genuine reason (circular dependencies, optional dependencies, etc.)

### Testing
- Use module-level constants for simple test values instead of fixtures that just return hardcoded strings
- Fixtures should be reserved for objects that require setup/teardown or dependency injection

### Exception Handling
- Never catch bare `Exception` unless at the highest level for error reporting
- Always catch specific exception types (e.g., `httpx.HTTPError`, `ValueError`)

### Dataclasses
- Only assign default values when there's a meaningful default (e.g., `is_active: bool = True`)
- Never use empty strings as placeholder defaults for required fields

## Tools

- Use `uv run` for executing Python commands (e.g., `uv run pytest`, `uv run python`)
- Use `ty` for type checking
