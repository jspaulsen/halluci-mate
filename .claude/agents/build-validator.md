# Build Validator

Verify the project installs and packages correctly.

## Steps

1. Clean cached artifacts:
   ```bash
   find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
   ```

2. Sync dependencies:
   ```bash
   uv sync
   ```

3. Verify core imports work:
   ```bash
   uv run python -c "import torch; import transformers; import datasets; import accelerate; print('Core imports OK')"
   ```

4. Verify project scripts parse without errors:
   ```bash
   uv run python -m py_compile scripts/train.py
   ```

5. Check for missing dependencies by scanning imports:
   ```bash
   uv run python -c "
   import ast, pathlib
   for f in pathlib.Path('scripts').glob('**/*.py'):
       tree = ast.parse(f.read_text())
       imports = [n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import)]
       imports += [n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
       print(f'{f}: {sorted(set(imports))}')
   "
   ```

6. Report: PASS or FAIL with details.
