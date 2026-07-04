"""Strategy loading: one folder per strategy, discovered by name.

A strategy is a folder ``strategies/<name>/`` containing:

* ``strategy.py`` — a pure ``signals(panel, **params)`` function (required),
  plus optional ``valid(params)`` and ``indicators(close, volume, **params)``.
* ``config.yaml`` — four blocks: ``data`` (what panel to load), ``params``
  (single-run defaults), ``sweep`` (grid lists), ``optuna`` (study settings
  and search space).

The engine (:mod:`trading_simple.backtest`, :mod:`trading_simple.optimize`)
consumes any strategy through :class:`LoadedStrategy` — adding a new strategy
never touches engine code.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from trading_simple.config import PROJECT_ROOT

STRATEGIES_DIR = PROJECT_ROOT / "strategies"


@dataclass(slots=True)
class LoadedStrategy:
    """A strategy folder resolved into callables and config blocks.

    Attributes:
        name: Folder name, e.g. ``"volume_momentum"``.
        dir: Absolute path of the strategy folder.
        signals: The strategy's signal function.
        valid: Combo validator (defaults to always-True).
        indicators: Optional plot-overlay hook (``None`` if not defined).
        data: The ``data:`` config block (universe/bar/start/end/session/...).
        params: The ``params:`` block (single-run defaults).
        sweep: The ``sweep:`` block (grid value lists; may be empty).
        optuna: The ``optuna:`` block (study/sampler/n_trials/space).
    """

    name: str
    dir: Path
    signals: Callable
    valid: Callable[[dict], bool]
    indicators: Callable | None
    data: dict
    params: dict
    sweep: dict = field(default_factory=dict)
    optuna: dict = field(default_factory=dict)


def available() -> list[str]:
    """List strategy folder names that contain a ``strategy.py`` (skips ``_template``)."""
    return sorted(p.parent.name for p in STRATEGIES_DIR.glob("*/strategy.py")
                  if not p.parent.name.startswith("_"))


def load(name: str) -> LoadedStrategy:
    """Load a strategy folder by name.

    Args:
        name: Folder name under ``strategies/``.

    Returns:
        The resolved :class:`LoadedStrategy`.

    Raises:
        FileNotFoundError: If the folder or its files are missing (message
            lists the available strategies).
        AttributeError: If ``strategy.py`` defines no ``signals`` function.
    """
    folder = STRATEGIES_DIR / name
    module_file, config_file = folder / "strategy.py", folder / "config.yaml"
    if not module_file.exists() or not config_file.exists():
        raise FileNotFoundError(
            f"Strategy {name!r} needs {module_file.name} + {config_file.name} in {folder}. "
            f"Available: {available()}"
        )
    spec = importlib.util.spec_from_file_location(f"strategies.{name}", module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "signals"):
        raise AttributeError(f"{module_file} must define a signals(panel, **params) function")
    cfg = yaml.safe_load(config_file.read_text()) or {}
    return LoadedStrategy(
        name=name,
        dir=folder,
        signals=module.signals,
        valid=getattr(module, "valid", lambda p: True),
        indicators=getattr(module, "indicators", None),
        data=cfg.get("data", {}),
        params=cfg.get("params", {}),
        sweep=cfg.get("sweep", {}),
        optuna=cfg.get("optuna", {}),
    )
