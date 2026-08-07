# Clodius

A Python library and CLI tool for aggregating large genomic datasets into tile-based formats for display at multiple resolutions (used by [HiGlass](https://higlass.io)).

## Project Structure

- `clodius/` — main package
  - `cli/` — Click-based CLI commands (`aggregate.py`, `convert.py`)
  - `core/` — the tile-serving interface layer: tile-id grammar, coordinate system, tileset protocols, wire payload shapes, server policy, error hierarchy
  - `tiles/` — tile generation modules per file type (bigwig, cooler, bed, etc.)
  - `tiles_v2/` — tilesets rewritten against `core/` (bed, bigbed, bigwig, cooler, multivec)
  - `models/` — Pydantic data models
- `test/` — pytest tests mirroring the source layout
- `test/sample_data/` — small sample files used by tests
- `data/` — larger test fixtures, stored in Git LFS

`tiles/` is what serves tiles today and is the authoritative implementation. `tiles_v2/` has no consumer yet; it is staged for differential comparison against `tiles/` and is not considered stable. New work on the interface layer belongs in `core/`.

## Development Setup

```shell
uv sync
```

This creates the virtual environment and installs the project in editable mode with the `dev` dependency group. Dev dependencies live in `[dependency-groups]` (PEP 735), not in `[project.optional-dependencies]`, so `pip install -e ".[dev]"` no longer works — pip warns and installs nothing.

## Common Commands

Commands are prefixed with `uv run`, matching CI.

Run all tests:
```shell
uv run pytest
```

Run a specific test:
```shell
uv run pytest test/cli_test.py::test_clodius_aggregate_bedgraph
```

Lint:
```shell
uv run ruff check clodius scripts test
```

Format (not enforced in CI — the tree predates `ruff format`):
```shell
uv run ruff format clodius
```

## Key Conventions

- **Linting**: ruff (configured via `pyproject.toml`), selecting `E`/`W`/`F`/`C90`. The `C901` complexity gate is set at 10; files that predate it are listed under `[tool.ruff.lint.per-file-ignores]` as a ratchet — new code must pass, and entries come off as functions are broken up. Nothing is added to that list.
- **Tests**: pytest with coverage (`uv run pytest --cov=clodius`)
- **Test file naming**: new test files are named `test_<module_name>.py`, mirroring the module under test. Most of the existing suite predates this and uses the `*_test.py` suffix; pytest collects both. Files are renamed as they are rewritten, not in a sweep.
- **Build**: hatchling
- **Dependencies**: runtime deps in `[project.dependencies]`; development deps in `[dependency-groups]` (PEP 735), managed with uv and pinned in `uv.lock`. CI installs with `uv sync --locked`.
- **Main branch**: `main` (use this as the base for PRs)
- **Python packaging**: `pyproject.toml` (no `setup.py`)
