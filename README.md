<div align="center">

# A Variant of Transformer Neural Process for Crops

</div>

<br>

## Environment Setup (uv)

This project uses [uv](https://docs.astral.sh/uv/) instead of conda for dependency management.

### First-time setup
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (one-time)
uv venv                                             # creates .venv using .python-version
uv sync --extra train      # if you're training models
uv sync --extra data       # if you're only running the data pipeline
uv sync --all-extras       # everything
```

### Running code
```bash
uv run python src/train.py experiment=example
uv run python -m src.data_pipeline.soil.run   # use -m for nested scripts inside src/
```

### Adding a package
```bash
uv add requests                     # core dependency
uv add --optional train torchvision # only for the "train" extra
uv add --optional data opencv-python # only for the "data" extra
```
This updates `pyproject.toml`, `uv.lock`, and your `.venv` in one step.

### Removing a package
```bash
uv remove some-package
```

### Manually edited `pyproject.toml`?
If you hand-edit `pyproject.toml` instead of using `uv add`/`uv remove`, resync the lockfile:
```bash
uv lock
uv sync
```

### Quick local-only install (not saved to pyproject/lock)
```bash
uv pip install some-throwaway-package
```
Use only for ad hoc local testing — won't be shared with the team.

### Notes
- `uv.lock` and `.python-version` are committed to git — don't gitignore them.
- No manual "activate" needed — `uv run` handles the venv automatically.