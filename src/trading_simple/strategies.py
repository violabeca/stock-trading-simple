"""Strategy loading: one folder per strategy, discovered by name.

A strategy is a folder ``strategies/<name>/`` containing:

* ``strategy.py`` — a pure ``signals(panel, **params)`` function (required),
  plus optional ``valid(params)`` and ``indicators(close, volume, **params)``.
* ``config.yaml`` — three blocks: ``data`` (what panel to load), ``optuna``
  (study settings), and ``params``, where **each parameter is declared once**::

      params:
        q_in:
          doc: Buy when the composite rank reaches this quantile.
          value: 0.85                                            # what SHIPS
          search: {type: float, low: 0.6, high: 0.95, step: 0.01}  # optuna explores
          grid: [0.75, 0.85]                                     # coarse-sweep values
        retrim_k: 2.0        # scalar shorthand: ships this, never searched or swept

  ``value`` is the single source of truth. ``search`` present means the study
  explores that knob; ``grid`` present means the coarse sweep varies it, and
  **omitting ``grid`` pins the sweep at ``value``** — so a grid can never express a
  strategy the config does not ship. :func:`load` enforces exactly that as two
  load-time invariants (see :func:`_unpack`), because the failure they prevent is
  silent: a walk-forward re-picking from a drifted grid faithfully validates a
  strategy nobody runs, and reports no error at all while doing it.

:class:`LoadedStrategy` still exposes the three flat views the engine consumes —
``params`` / ``sweep`` / ``optuna["space"]`` — so this declaration style is the only
thing that changed; :mod:`trading_simple.backtest`, :mod:`trading_simple.optimize`
and the CLI read exactly what they read before.
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
        params: Every parameter's shipped ``value`` (single-run defaults).
        sweep: Grid value lists, for the parameters that declare a ``grid``.
        optuna: The ``optuna:`` block plus the ``space`` assembled from every
            parameter's ``search``.
        docs: Each parameter's ``doc`` string, for the ones that carry one.
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
    docs: dict = field(default_factory=dict)


#: Keys a long-form parameter entry may declare. Anything else is a typo, and a typo in
#: ``serach:`` would otherwise mean "not searched" — silently, and only visible as a study
#: that never moves the knob.
_PARAM_KEYS = {"value", "doc", "search", "grid"}


def _in_space(value, spec: dict) -> bool:
    """Is ``value`` reachable by the search space ``spec``?

    Args:
        value: The shipped value.
        spec: A ``search:`` entry (``{type, low, high, ...}`` or
            ``{type: categorical, choices: [...]}``).

    Returns:
        True when the study could itself have suggested ``value``. Step alignment is
        deliberately NOT checked: Optuna snaps to the grid, so a value off-step is
        reachable in practice and flagging it would be a false alarm.
    """
    if spec.get("type") == "categorical":
        return value in spec.get("choices", [])
    low, high = spec.get("low"), spec.get("high")
    return low is None or high is None or low <= value <= high


def _unpack(cfg: dict, name: str) -> tuple[dict, dict, dict, dict]:
    """Split the unified ``params:`` block into the flat views the engine consumes.

    Also the place the config's two invariants are enforced, at load time, so a drifted
    config fails on ``pytest``, ``tsim`` and the first notebook cell alike rather than
    quietly producing numbers about a different strategy:

    1. A shipped ``value`` must be reachable by its own ``search`` space. Otherwise the
       study can never suggest what you actually run, and every boundary/stability
       report is drawn around a point outside the search.
    2. A shipped ``value`` must appear in its own ``grid``. This is the one that bit this
       repo twice: the sweep grid is what the walk-forward re-picks from, so a grid that
       cannot express the shipped shape means the validation is about another strategy.

    Args:
        cfg: The raw ``params:`` mapping from YAML.
        name: Strategy name, for error messages.

    Returns:
        ``(params, sweep, space, docs)``.

    Raises:
        ValueError: On an unknown entry key, a missing ``value``, or either invariant.
    """
    params, sweep, space, docs = {}, {}, {}, {}
    for key, spec in (cfg or {}).items():
        if not isinstance(spec, dict) or "value" not in spec:
            params[key] = spec                      # scalar shorthand: just a value
            continue
        if unknown := set(spec) - _PARAM_KEYS:
            raise ValueError(f"{name}/config.yaml: params.{key} has unknown key(s) "
                             f"{sorted(unknown)}; allowed: {sorted(_PARAM_KEYS)}")
        value = params[key] = spec["value"]
        if "doc" in spec:
            docs[key] = spec["doc"]
        if "search" in spec:
            space[key] = spec["search"]
            if not _in_space(value, spec["search"]):
                raise ValueError(
                    f"{name}/config.yaml: params.{key} ships {value!r}, which its own "
                    f"search space {spec['search']} cannot reach — the study could never "
                    f"suggest the value you run.")
        if "grid" in spec:
            sweep[key] = spec["grid"]
            if value not in spec["grid"]:
                raise ValueError(
                    f"{name}/config.yaml: params.{key} ships {value!r} but its grid is "
                    f"{spec['grid']} — the sweep (and every walk-forward re-picking from "
                    f"it) would be validating a strategy this config does not run.")
    return params, sweep, space, docs


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
        ValueError: If the config declares a parameter the search or the grid
            contradicts (see :func:`_unpack`).
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
    params, sweep, space, docs = _unpack(cfg.get("params", {}), name)
    return LoadedStrategy(
        name=name,
        dir=folder,
        signals=module.signals,
        valid=getattr(module, "valid", lambda p: True),
        indicators=getattr(module, "indicators", None),
        data=cfg.get("data", {}),
        params=params,
        sweep=sweep,
        # `space` is assembled from the parameters themselves, so a searched knob and the
        # value it ships can never sit in two places that disagree.
        optuna={**cfg.get("optuna", {}), "space": space},
        docs=docs,
    )
