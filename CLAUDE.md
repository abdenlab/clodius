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

## Test conventions — local carve-outs

The Python test guide is the default. These are the points where this repo diverges deliberately, recorded here so the divergence is a decision rather than drift.

- **`<method_name>` for a class, enum, constant, or module attribute.** The guide fixes the slot to a method's `__name__` and defines no form for a name that is not a method. Where a test exercises a class, enum or module-level constant as a whole, the slot carries the snake-cased symbol name: `test_tile_kind_should_*` for `TileKind`, `test_grid_policy_should_*` for `GridPolicy`, `test_ladder_should_*` for `Ladder`. Dunder attributes keep the guide's literal form, so `__all__` gives `test___all___should_*`. A name must still describe what the body verifies — that part is not carved out, and a test whose name points at a different unit than its assertions is a defect regardless of this entry.
- **`Test<Behavior>` classes over module-level functions.** §9 asks for module-level test functions and reserves `Test<Scenario>` classes for suites with "no first-party class under test". A cohesive suite over one module-level function may use a behavior-named class even where a first-party class shares the file — `TestOverlapPredicate`, `TestMaxRecords`, `TestBinCount`. The classes carry the explanatory docstrings that make these suites navigable, and flattening them would lose that with nothing gained.
- **`test/tiles/test_conformance.py` does not mirror a module.** It asserts one property — a dense tile holds exactly the bin count its `tileset_info` advertises — across bigwig, cooler and fasta, against the real fixtures in `data/`. The cross-type framing is the point: each type reconciles a ragged grid onto the uniform lattice differently, and the file exists to state that they must agree. Splitting it per module would also split the shared LFS fixture declarations and the availability check that reports on all of them at once.

## Marker conventions

Registered in `pyproject.toml` under `[tool.pytest.ini_options]`.

- `integration` — exercises a real cross-boundary interaction: a subprocess, network I/O, an on-disk format. `pytest -m "not integration"` is the fast inner loop.
- `pinned` — records behavior as observed on a question that is open, not a contract that is settled. `pytest -m pinned` enumerates every such decision in one command, which is what keeps them reviewable instead of scattered through docstrings.

Skipping on un-smudged git-LFS payloads is not a marker: `test/harness/lfs.py::requires_lfs` returns a `pytest.mark.skipif`, evaluated per call site.
