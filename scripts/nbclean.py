#!/usr/bin/env python3
"""Strip outputs from notebooks so git only ever stores their source.

Committed notebooks here are documentation: they must run top-to-bottom, so
their outputs are always reproducible and never worth versioning. Plotly
embeds the full raw series behind every figure, which turns one backtest
chart into tens of megabytes of JSON -- hence this cleaner, run on staged
notebooks by `.githooks/pre-commit`.

Deliberately stdlib-only (no nbformat import): a git hook must work without
the project venv activated.

Usage:
    python3 scripts/nbclean.py <notebook>...   # strip in place
    python3 scripts/nbclean.py --check <nb>... # exit 1 if any would change
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Per-cell metadata written by the Jupyter frontend; pure churn in a diff.
CELL_METADATA_NOISE = ("execution", "scrolled", "collapsed", "ExecuteTime")

# ipywidgets dumps its whole state blob here, and it goes stale immediately.
NOTEBOOK_METADATA_NOISE = ("widgets",)


def clean(nb: dict) -> bool:
    """Clear every code cell's outputs and execution count, plus metadata noise.

    Args:
        nb: A parsed nbformat 4 notebook, mutated in place.

    Returns:
        True if anything changed, False if the notebook was already clean.
    """
    dirty = False

    for key in NOTEBOOK_METADATA_NOISE:
        dirty |= nb.get("metadata", {}).pop(key, None) is not None

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            dirty = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            dirty = True
        for key in CELL_METADATA_NOISE:
            dirty |= cell.get("metadata", {}).pop(key, None) is not None

    return dirty


def clean_file(path: Path, check: bool) -> bool:
    """Clean one notebook on disk, matching Jupyter's own on-disk formatting.

    Args:
        path: Path to a `.ipynb` file.
        check: Report whether the file is dirty without rewriting it.

    Returns:
        True if the file was changed (or, under `check`, would be changed).
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    dirty = clean(nb)
    if dirty and not check:
        # indent=1 + trailing newline is what Jupyter writes; anything else
        # makes every save churn the whole file.
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return dirty


def main(argv: list[str]) -> int:
    """Clean the notebooks named on the command line.

    Args:
        argv: CLI arguments; an optional `--check` flag followed by paths.

    Returns:
        Process exit code -- 1 under `--check` if any notebook is dirty, else 0.
    """
    check = "--check" in argv
    paths = [Path(a) for a in argv if a != "--check"]
    if not paths:
        print(__doc__, file=sys.stderr)
        return 2

    dirty = [p for p in paths if p.suffix == ".ipynb" and clean_file(p, check)]
    for path in dirty:
        print(f"{'would clear' if check else 'cleared'} {path}", file=sys.stderr)
    return 1 if (check and dirty) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
