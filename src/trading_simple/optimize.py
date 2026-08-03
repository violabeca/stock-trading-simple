"""Optuna optimization: multi-objective, resumable, one study file per run.

Every study maximizes **two objectives at once** — Sharpe ratio and total
return — so the result is a *Pareto front* of defensible trade-offs rather
than a single (possibly degenerate) winner. :func:`pick_best` then reduces the
front to one trial under an explicit preference weighting, and
:func:`boundary_report` checks whether the winner is pressing against the
searched ranges (a hint to widen ``optuna.space`` and re-run).

Resume is free by construction: studies live in append-only **journal files**
(``strategies/<name>/studies/<study>.log``). Every finished trial is persisted
immediately, ``load_if_exists=True`` picks a study back up after any crash or
Ctrl-C, and the file lock makes several worker processes on one study safe —
which is exactly how ``jobs.run_optimize(workers=N)`` parallelizes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from trading_simple import backtest as bt
from trading_simple.config import Settings
from trading_simple.data import Panel

if TYPE_CHECKING:
    import optuna

    from trading_simple.strategies import LoadedStrategy

log = logging.getLogger(__name__)

#: The two study objectives, both maximized, in this order.
#:
#: **Benchmark-relative, deliberately.** Optimizing an *absolute* Sharpe and return
#: rewards market exposure rather than skill: a long-only strategy in a rising market
#: raises both simply by owning more names for longer, so the optimizer drifts toward
#: becoming the index. (Observed, not theoretical — `informed_flow` did exactly this:
#: it pushed its entry quantile to the least selective value allowed and still lost to
#: buy-and-hold on both train and holdout, while posting a "good" Sharpe of 0.94.)
#:
#: **Per-year, equally deliberately.** Both objectives are annualized, so they measure
#: the same thing per unit of time. A *cumulative* excess return does not: it grows with
#: the length of the backtest window, so it silently rewards a long window for being
#: long. That makes it useless for comparing strategies (this repo's windows run from one
#: month to four years) and — worse, because it is easy to miss — useless for comparing
#: **walk-forward folds of unequal length within a single strategy**, which is exactly
#: what the fold-picking code does. Mixing one annualized objective with one cumulative
#: one, as this study did before, is incoherent on its own terms.
#:
#: Both are measured against an equal-weight buy-and-hold of the same panel
#: (:func:`backtest.benchmark`), so a strategy scores above zero only by beating the
#: market it trades in. Absolute Sharpe and return are still recorded on every trial —
#: as diagnostics, not as targets.
#:
#: What is deliberately NOT normalized away: **time in market**. Scaling a 30%-invested
#: strategy's return up by 3x to "compare it fairly" inverts the truth — sitting in cash
#: is a real cost paid against a fully-invested benchmark, and it is the very thing that
#: has to be earned back. The benchmark-relative framing prices it correctly already.
OBJECTIVES = ("info_ratio", "excess_cagr")
#: Secondary metrics attached to each trial as user attributes. ``years`` and
#: ``trades_per_year`` are the sample size behind the objectives: annualizing makes a
#: one-month backtest *comparable*, it does not make it *trustworthy*, so the evidence
#: rides along with the claim.
EXTRA_METRICS = ("sharpe", "cagr", "total_return", "max_drawdown", "win_rate",
                 "pct_profitable", "n_trades", "trades_per_year", "years")


def study_path(strategy: LoadedStrategy, study_name: str):
    """The journal file backing one study (``studies/<study_name>.log``).

    Args:
        strategy: The strategy owning the study.
        study_name: Study name.

    Returns:
        The journal file ``Path`` — also what ``optuna-dashboard`` opens.
    """
    return strategy.dir / "studies" / f"{study_name}.log"


def storage_for(strategy: LoadedStrategy, study_name: str) -> optuna.storages.JournalStorage:
    """Open (or create) the journal storage file for one study.

    Args:
        strategy: The strategy owning the study.
        study_name: Study name — maps to ``studies/<study_name>.log``.

    Returns:
        A multi-process-safe Optuna storage.
    """
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    path = study_path(strategy, study_name)
    path.parent.mkdir(exist_ok=True)
    return JournalStorage(JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path))))


def make_sampler(cfg: dict) -> optuna.samplers.BaseSampler:
    """Build the configured sampler (``nsga2`` | ``tpe`` | ``random``), seeded.

    Args:
        cfg: The strategy's ``optuna:`` block.

    Returns:
        An Optuna sampler.

    Raises:
        KeyError: On an unknown sampler name (message lists the choices).
    """
    import optuna

    seed = cfg.get("seed")
    samplers = {
        "nsga2": lambda: optuna.samplers.NSGAIISampler(seed=seed),
        "tpe": lambda: optuna.samplers.TPESampler(seed=seed, multivariate=True),
        "random": lambda: optuna.samplers.RandomSampler(seed=seed),
    }
    name = cfg.get("sampler", "nsga2")
    try:
        return samplers[name]()
    except KeyError:
        raise KeyError(f"Unknown sampler {name!r}. Choices: {sorted(samplers)}") from None


def suggest_params(trial: optuna.Trial, space: dict[str, dict]) -> dict:
    """Draw one parameter set from a declarative YAML search space.

    Space entries (see any strategy's ``config.yaml``):
    ``{type: int, low, high, [step], [log]}``, ``{type: float, low, high,
    [step], [log]}``, or ``{type: categorical, choices: [...]}``.

    Args:
        trial: The Optuna trial doing the suggesting.
        space: ``optuna.space`` mapping of parameter name to spec.

    Returns:
        ``{param: suggested value}``.

    Raises:
        ValueError: On an unknown spec type.
    """
    params = {}
    for name, spec in space.items():
        kind = spec.get("type")
        if kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"],
                                             step=spec.get("step", 1), log=spec.get("log", False))
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"],
                                               step=spec.get("step"), log=spec.get("log", False))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown space type {kind!r} for parameter {name!r}")
    return params


def objective(trial: optuna.Trial, strategy: LoadedStrategy, panel: Panel,
              settings: Settings) -> tuple[float, float]:
    """Evaluate one trial: suggest params, backtest, return the benchmark-relative pair.

    Invalid parameter combos and degenerate results (NaN metrics — usually
    zero trades) are pruned rather than scored, so they never pollute the
    Pareto front. So are trials trading below the strategy's optional
    ``optuna.min_trades_per_year`` rate — a few lucky fills can post a huge
    info_ratio, and this admissibility floor keeps such artifacts off the
    front. Per YEAR, not an absolute trade count: an absolute floor is
    implicitly far stricter on a short backtest window than a long one for
    the identical trading style. Secondary metrics land in ``trial.user_attrs``.

    Args:
        trial: The Optuna trial.
        strategy: A loaded strategy.
        panel: The price panel to backtest on (loaded once per process).
        settings: Global settings.

    Returns:
        ``(info_ratio, excess_cagr)`` — see :data:`OBJECTIVES` for why these and not
        the absolute Sharpe, and why per-year and not cumulative.

    Raises:
        optuna.TrialPruned: For invalid or degenerate parameter sets.
    """
    import optuna

    params = suggest_params(trial, strategy.optuna.get("space", {}))
    if not strategy.valid({**strategy.params, **params}):
        raise optuna.TrialPruned("invalid parameter combination")
    stats = bt.evaluate(strategy, panel, params, settings=settings)
    if any(pd.isna(stats[m]) for m in OBJECTIVES):
        raise optuna.TrialPruned("degenerate result (no trades?)")
    min_rate = strategy.optuna.get("min_trades_per_year", 0)
    if stats["trades_per_year"] < min_rate:
        # A handful of lucky trades can post a huge info_ratio and crown the front;
        # the floor keeps such artifacts out of the Pareto set entirely. Per-YEAR,
        # not an absolute count: an absolute floor is implicitly far stricter on a
        # short backtest window than a long one, for the exact same trading style.
        raise optuna.TrialPruned(
            f"only {stats['trades_per_year']:.1f} trades/year (< min_trades_per_year={min_rate})")
    for name in EXTRA_METRICS:
        trial.set_user_attr(name, stats[name])
    return tuple(stats[m] for m in OBJECTIVES)


def progress_callback():
    """An Optuna callback printing one parseable line per finished trial.

    Streams live through ``tsim optimize`` into notebooks: objectives, the
    current front size, and the trial's parameters — enough to watch the
    search move without a dashboard.

    Returns:
        A ``(study, trial)`` callable for ``study.optimize(callbacks=...)``.
    """
    import optuna

    def report(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        params = " ".join(f"{k}={v}" for k, v in trial.params.items())
        if trial.state != optuna.trial.TrialState.COMPLETE:
            print(f"trial {trial.number:>4}: {trial.state.name.lower()} | {params}", flush=True)
            return
        # Print the objectives by their real names. Hard-coding "sharpe=" here once meant
        # the live log claimed to show a metric the study had long since stopped using.
        scored = " ".join(f"{name}={value:+.3f}"
                          for name, value in zip(OBJECTIVES, trial.values, strict=True))
        print(f"trial {trial.number:>4}: {scored} front={len(study.best_trials)} | {params}",
              flush=True)

    return report


def run_study(strategy: LoadedStrategy, panel: Panel, *, settings: Settings,
              study_name: str | None = None, n_trials: int | None = None,
              timeout_s: float | None = None, verbose: bool = False) -> optuna.Study:
    """Run — or RESUME — a strategy's optimization study.

    Re-running with the same study name continues exactly where the journal
    left off; nothing is ever recomputed or lost.

    Args:
        strategy: A loaded strategy (its ``optuna:`` block configures the run).
        panel: The price panel to optimize on.
        settings: Global settings.
        study_name: Override the configured study name.
        n_trials: Trials to run now (default: config ``n_trials``).
        timeout_s: Stop after this many seconds (default: config ``timeout_s``).
        verbose: Print one line per finished trial (see :func:`progress_callback`).

    Returns:
        The (re)loaded study, including all previous trials.
    """
    import optuna

    cfg = strategy.optuna
    name = study_name or cfg.get("study", f"{strategy.name}-v1")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=name,
        storage=storage_for(strategy, name),
        directions=["maximize"] * len(OBJECTIVES),
        sampler=make_sampler(cfg),
        load_if_exists=True,
    )
    done = len(study.trials)
    if done:
        log.info("study %s: resuming with %d existing trials", name, done)
    study.optimize(
        lambda t: objective(t, strategy, panel, settings),
        n_trials=n_trials if n_trials is not None else cfg.get("n_trials", 100),
        timeout=timeout_s if timeout_s is not None else cfg.get("timeout_s"),
        show_progress_bar=False,
        callbacks=[progress_callback()] if verbose else None,
    )
    return study


def load_study(strategy: LoadedStrategy, study_name: str | None = None) -> optuna.Study:
    """Reload an existing study read-only-style (e.g. in a notebook).

    Args:
        strategy: The strategy owning the study.
        study_name: Study name (default: the configured one).

    Returns:
        The study with all persisted trials.
    """
    import optuna

    name = study_name or strategy.optuna.get("study", f"{strategy.name}-v1")
    return optuna.load_study(study_name=name, storage=storage_for(strategy, name))


def trials_frame(study: optuna.Study) -> pd.DataFrame:
    """All completed trials as a tidy frame (params + objectives + extras).

    Args:
        study: An Optuna study.

    Returns:
        One row per completed trial with a ``trial`` number column.
    """
    import optuna

    rows = [
        {"trial": t.number, **t.params,
         **dict(zip(OBJECTIVES, t.values, strict=True)), **t.user_attrs}
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
    ]
    return pd.DataFrame(rows)


def pareto_frame(study: optuna.Study) -> pd.DataFrame:
    """The study's Pareto-optimal trials as a tidy frame.

    Long-running NSGA-II studies re-sample winning parameter sets; such exact
    duplicates (same params, same objectives, different trial number) are
    collapsed to the earliest trial so the front stays readable.

    Args:
        study: An Optuna study.

    Returns:
        One row per distinct Pareto trial, sorted by the first objective descending.
    """
    rows = [
        {"trial": t.number, **t.params,
         **dict(zip(OBJECTIVES, t.values, strict=True)), **t.user_attrs}
        for t in study.best_trials
    ]
    frame = pd.DataFrame(rows).sort_values("trial")
    frame = frame.drop_duplicates(subset=[c for c in frame.columns if c != "trial"])
    return frame.sort_values(OBJECTIVES[0], ascending=False).reset_index(drop=True)


def pick_best(study: optuna.Study, *, weights: tuple[float, float] = (1.0, 1.0)):
    """Reduce the Pareto front to ONE trial under an explicit preference.

    Each objective is min-max normalized across the front, then the trial
    with the highest weighted sum wins. ``weights=(1, 0)`` = pure ``info_ratio``,
    ``(1, 1)`` = balanced, ``(0, 1)`` = pure ``excess_cagr``.

    Args:
        study: An Optuna study with at least one completed trial.
        weights: Importance of each entry of :data:`OBJECTIVES`, in that order.

    Returns:
        The chosen ``optuna.trial.FrozenTrial`` (its ``.params`` re-runs the
        backtest; its ``.number`` highlights it on the Pareto plot).

    Raises:
        ValueError: If the study has no Pareto-optimal trials yet.
    """
    front = study.best_trials
    if not front:
        raise ValueError("study has no completed trials yet")
    values = pd.DataFrame([t.values for t in front], columns=list(OBJECTIVES))
    span = values.max() - values.min()
    normalized = (values - values.min()) / span.replace(0, 1.0)
    score = normalized.mul(pd.Series(dict(zip(OBJECTIVES, weights, strict=True)))).sum(axis=1)
    return front[int(score.idxmax())]


def _span_position(values, spec: dict) -> np.ndarray:
    """Normalized 0–1 position of value(s) inside a numeric space entry.

    ``log: true`` ranges are measured in log space, so "5% from the edge"
    means the same thing on a 5–120 log range as on a linear one.

    Args:
        values: Scalar or array of parameter values.
        spec: The ``optuna.space`` entry (``{type, low, high, [log]}``).

    Returns:
        Array of positions (0 = at ``low``, 1 = at ``high``; 0.5 for a
        degenerate ``low == high`` range).
    """
    vals = np.asarray(values, dtype=float)
    low, high = float(spec["low"]), float(spec["high"])
    if spec.get("log"):
        vals, low, high = np.log(vals), np.log(low), np.log(high)
    return (vals - low) / (high - low) if high > low else np.full_like(vals, 0.5)


def boundary_report(best_params: dict, space: dict[str, dict], *,
                    front: pd.DataFrame | None = None, rel_tol: float = 0.05) -> pd.DataFrame:
    """Flag optimized parameters that sit on a search-space boundary.

    A picked trial hugging one end of its searched range is evidence the true
    optimum may lie OUTSIDE the range: widen that bound in ``optuna.space``
    and re-run the study (same study name resumes; a new name restarts clean).
    Works for any strategy — it only needs the picked parameters and the
    declarative space.

    A parameter counts as at an edge when it lands within ``rel_tol`` of a
    bound, measured on the searched scale (log space for ``log: true``
    ranges). Categorical parameters have no edges and are skipped.

    Args:
        best_params: The picked trial's parameters (``pick_best(study).params``).
        space: The strategy's ``optuna.space`` block.
        front: Optional Pareto front (:func:`pareto_frame`) — adds the share
            of front trials on each edge; a whole front piling onto one bound
            is stronger evidence than a single trial.
        rel_tol: Edge-zone width as a fraction of the searched span.

    Returns:
        One row per numeric parameter (indexed by name): ``low, high, best,
        at_edge`` (``"low"`` | ``"high"`` | ``""``), ``front_at_low`` /
        ``front_at_high`` shares when ``front`` is given, and ``action``
        (``"ok"``, or which bound to widen — flagged when the best value is
        at an edge or at least half the front is).
    """
    rows = []
    for name, spec in space.items():
        if spec.get("type") == "categorical" or name not in best_params:
            continue
        pos = float(_span_position(best_params[name], spec))
        at_edge = "low" if pos <= rel_tol else "high" if pos >= 1.0 - rel_tol else ""
        row = {"param": name, "low": spec["low"], "high": spec["high"],
               "best": best_params[name], "at_edge": at_edge}
        if front is not None and name in front.columns:
            fpos = _span_position(front[name].to_numpy(), spec)
            row["front_at_low"] = float((fpos <= rel_tol).mean())
            row["front_at_high"] = float((fpos >= 1.0 - rel_tol).mean())
        sides = [s for s in ("low", "high")
                 if at_edge == s or row.get(f"front_at_{s}", 0.0) >= 0.5]
        row["action"] = " & ".join(f"widen {s}" for s in sides) if sides else "ok"
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["low", "high", "best", "at_edge", "action"]).rename_axis("param")
    return pd.DataFrame(rows).set_index("param")


# ---------------------------------------------------------------------------
# Local robustness: is the pick a plateau or a spike?
# ---------------------------------------------------------------------------
#: Perturbation sizes for :func:`neighbors`, as a fraction of each parameter's
#: **searched span** — the same normalized scale :func:`boundary_report` measures
#: edges on, so §8 and the stability check speak one language.
NEIGHBOR_SPANS = (0.05, 0.10)


def _span_value(pos: float, spec: dict):
    """Inverse of :func:`_span_position`: a 0–1 span position back to a legal value.

    Snaps to the space's own grid (``step``, integers for ``type: int``) and clips
    to the bounds, so every value returned is one the study could itself have
    suggested — a "neighbor" the optimizer could never reach would be testing a
    strategy nobody can ship.

    Args:
        pos: Position inside the searched range (0 = ``low``, 1 = ``high``);
            clipped into that interval.
        spec: The ``optuna.space`` entry.

    Returns:
        The snapped value, ``int`` for an int space and ``float`` otherwise.
    """
    low, high = float(spec["low"]), float(spec["high"])
    pos = min(max(pos, 0.0), 1.0)
    value = (np.exp(np.log(low) + pos * (np.log(high) - np.log(low))) if spec.get("log")
             else low + pos * (high - low))
    if step := spec.get("step"):
        value = low + round((value - low) / step) * step
    value = min(max(value, low), high)
    return int(round(value)) if spec.get("type") == "int" else round(float(value), 10)


def neighbors(picked: dict, space: dict[str, dict], *, valid=lambda p: True,
              spans: tuple[float, ...] = NEIGHBOR_SPANS) -> pd.DataFrame:
    """The local star around a picked parameter set: perturb one coordinate at a time.

    A study reports the best point it found; it says nothing about whether that
    point is a **plateau** (neighbors score about the same — the parameter values
    are not load-bearing and small real-world drift is survivable) or a **spike**
    (neighbors collapse — the pick is fitted to noise). Answering that needs
    points *near the pick*, which neither the Pareto front nor the coarse
    ``grid:`` sweep provides: the front is a global trade-off curve, and the grid
    is a fixed ablation lattice that generally contains no point near the optimum
    at all.

    Perturbations are measured **on the searched scale** — ±5% and ±10% of each
    parameter's own span, in log space for ``log: true`` ranges — so one bar of the
    resulting tornado is comparable to the next. Coarse parameters are the
    exception worth knowing about: a 1–3 integer cannot move by 10% of its span,
    so it falls back to its smallest representable step rather than being silently
    reported as inert. Categorical parameters contribute one row per alternative
    choice (``offset`` is NaN — a switch has no near or far side).

    Args:
        picked: The full parameter set to perturb (typically the study's pick).
        space: The strategy's ``optuna.space`` block; keys absent from ``picked``
            and picked keys absent from the space are both skipped.
        valid: The strategy's combo validator — perturbations it rejects (an exit
            band crossing its entry band, say) are dropped, not backtested.
        spans: Perturbation sizes as fractions of each searched span.

    Returns:
        One row per candidate, the unperturbed center first: every parameter as a
        column, plus ``perturbed`` (which coordinate moved, ``""`` for the center)
        and ``offset`` (the requested span offset, signed).
    """
    rows = [{**picked, "perturbed": "", "offset": 0.0}]
    offsets = sorted({*spans, *(-s for s in spans)}, key=abs)     # nearest first: see dedupe
    for name, spec in space.items():
        if name not in picked:
            continue
        if spec.get("type") == "categorical":
            candidates = {c: float("nan") for c in spec["choices"] if c != picked[name]}
        else:
            pos = float(_span_position(picked[name], spec))
            step = spec.get("step") or 1
            candidates = {}
            for offset in offsets:
                value = _span_value(pos + offset, spec)
                if value == picked[name]:
                    # Too coarse to move by this offset (a 1-3 int, a 0.25-step weight
                    # near a bound): take the smallest step it CAN make instead, so the
                    # knob is judged rather than reported inert by construction.
                    value = picked[name] + (step if offset > 0 else -step)
                    if not spec["low"] <= value <= spec["high"]:
                        continue
                    value = int(value) if spec.get("type") == "int" else round(float(value), 10)
                # Nearest offset wins when two offsets snap to the same value.
                candidates.setdefault(value, offset)
        for value, offset in sorted(candidates.items(),
                                    key=lambda kv: (kv[1] if kv[1] == kv[1] else 0.0, str(kv[0]))):
            candidate = {**picked, name: value}
            if valid(candidate):
                rows.append({**candidate, "perturbed": name, "offset": offset})
    return pd.DataFrame(rows)


def sensitivity(strategy: LoadedStrategy, panel: Panel, picked: dict, *, settings: Settings,
                spans: tuple[float, ...] = NEIGHBOR_SPANS,
                space: dict[str, dict] | None = None) -> pd.DataFrame:
    """Backtest the local star around ``picked`` — the input to :func:`stability_report`.

    One :func:`backtest.sweep` call, so the whole neighborhood costs about what a
    coarse grid does. The metric to read is ``excess_cagr``: benchmark-relative and
    per-year, i.e. one of the study's own :data:`OBJECTIVES`. (``info_ratio``, the
    other one, needs per-combo equity curves a column-stacked sweep does not keep.)

    Args:
        strategy: A loaded strategy.
        panel: The **training** panel — re-centering a pick is a train-window
            decision, and scoring candidates on the holdout would burn it.
        picked: Full parameter set to perturb.
        settings: Global settings.
        spans: Perturbation sizes (see :func:`neighbors`).
        space: Search space; defaults to the strategy's ``optuna.space``.

    Returns:
        :func:`neighbors`' ``perturbed``/``offset`` labels joined to every
        :func:`backtest.sweep` metric, one row per candidate.
    """
    star = neighbors(picked, space if space is not None else strategy.optuna["space"],
                     valid=strategy.valid, spans=spans)
    keys = list(picked)
    # Cast back through the picked set's own python types: a DataFrame column upcasts an
    # int grid to float the moment any row differs, and signals() receives the raw value.
    grid = [{k: type(picked[k])(row[k]) for k in keys}
            for row in star[keys].to_dict("records")]
    stats = bt.sweep(strategy, panel, grid, settings=settings)
    return pd.concat([star[["perturbed", "offset"]], stats], axis=1)


def stability_report(star: pd.DataFrame, *, metric: str = "excess_cagr", inert: float = 0.25,
                     cliff: float = 0.5, min_trades: int = 0) -> pd.DataFrame:
    """Per-parameter verdict on a local star, with the action each verdict implies.

    Reads like :func:`boundary_report` — one row per parameter, ending in an
    ``action`` column — and is deliberately **relative to the center**, never a
    fixed threshold: ``metric`` is signed and benchmark-relative, so a rule like
    "median > 0.5 × max" silently inverts as soon as the best neighbor is negative.

    Verdicts, and what each one means you should do:

    * ``inert`` — the whole neighborhood scores within ``inert`` of the center.
      The knob is not load-bearing *here*: pin it in ``params:`` and spend the
      trial budget on knobs that move the result. Note the scope — this is a
      **local** claim, and a knob that is flat within ±10% of its span can still
      be decisive at the far end of it; the sweep grid and the front say that,
      not this table.
    * ``plateau`` — moves cost something but nothing collapses. Ship it.
    * ``cliff (low|high)`` — one side loses more than ``cliff`` of the center's
      score, or drops below the benchmark, within one perturbation. The pick is on
      an edge; ``suggest`` holds the nearest value on the safe side, or ``None``
      when the pick sits on a searched **bound** with the collapse beside it —
      there is nowhere to step back to, and the fix is a wider bound (see
      :func:`boundary_report`), not a re-centering.
    * ``spike`` — *both* sides collapse. The classic overfit optimum: do not ship
      it, prefer a flatter point from the front.
    * ``off-center`` — a neighbor beats the center by more than ``inert``. Not an
      instruction to move: an in-sample bump is exactly what a walk-forward fold
      exists to referee.
    * ``n/a`` — the center itself does not beat buy-and-hold, so its neighborhood
      is not the problem.

    Args:
        star: Output of :func:`sensitivity`.
        metric: Column to judge on (benchmark-relative by default).
        inert: Fraction of the center's score below which a spread counts as noise.
        cliff: Fraction of the center's score a side must lose to count as a cliff.
        min_trades: Trade-count floor; neighbors below it are **counted** in
            ``n_thin`` rather than dropped — a perturbation that starves the book
            is a finding about that parameter, not a row to hide.

    Returns:
        One row per perturbed parameter, indexed by name: ``value`` (the picked
        one), ``tried_low``/``tried_high``, ``worst``, ``best``, ``range``,
        ``n_thin``, ``verdict``, ``suggest`` and ``action``.
    """
    cols = ["value", "tried_low", "tried_high", "worst", "best", "range", "n_thin",
            "verdict", "suggest", "action"]
    center = star[star["perturbed"] == ""]
    moved = star[star["perturbed"] != ""]
    if not len(center) or not len(moved):
        return pd.DataFrame(columns=cols).rename_axis("param")
    mid = float(center[metric].iloc[0])
    rows = []
    for name, group in moved.groupby("perturbed", sort=False):
        scores, values = group[metric], group[name]
        worst, best = float(scores.min()), float(scores.max())
        row = {"param": name, "value": center[name].iloc[0],
               "tried_low": values.min(), "tried_high": values.max(),
               "worst": worst, "best": best, "range": max(best, mid) - min(worst, mid),
               "n_thin": int((group["n_trades"] < min_trades).sum()),
               "suggest": None}
        def gave_way(scores: pd.Series) -> bool:
            """Did this side lose the benchmark, or most of the center's score?"""
            return len(scores) > 0 and bool(scores.min() <= 0 or mid - scores.min() >= cliff * mid)

        collapsed = {}       # side -> the nearest value on the OTHER side of the cliff, if any
        for side, on_side in (("low", group["offset"] < 0), ("high", group["offset"] > 0)):
            if gave_way(group.loc[on_side, metric]):
                # No safe side means the pick sits on a searched BOUND with the collapse
                # right next to it — there is nowhere to step back to, and the move is
                # not a re-centering but a wider bound (§8's job, not this table's).
                safe = group[~on_side & group["offset"].notna()]
                collapsed[side] = values[safe["offset"].abs().idxmin()] if len(safe) else None
        # A categorical has no near or far side — its alternatives carry offset NaN.
        switch = group[group["offset"].isna()]
        if mid <= 0:
            row |= {"verdict": "n/a",
                    "action": "center loses to buy-and-hold — read §10a, not this table"}
        elif row["range"] <= inert * mid:
            row |= {"verdict": "inert",
                    "action": f"flat near the pick: pin {name}, spend the trials elsewhere "
                              "(a LOCAL claim — it may still bite far from here)"}
        elif gave_way(switch[metric]):
            row |= {"verdict": "decisive",
                    "action": f"the other choice loses the edge — keep {name}={row['value']}"}
        elif len(collapsed) == 2:
            row |= {"verdict": "spike",
                    "action": "both sides collapse — do not ship this pick; take a flatter "
                              "point off the front"}
        elif collapsed:
            side, safe_value = next(iter(collapsed.items()))
            row |= {"verdict": f"cliff ({side})", "suggest": safe_value,
                    "action": (f"on an edge: ship {name}={safe_value}, one step clear of it"
                               if safe_value is not None else
                               f"on an edge with no room to step back — {name} is at a "
                               "searched bound; widen it (§8) and re-run")}
        elif best > mid + inert * abs(mid):
            row |= {"verdict": "off-center",
                    "action": "a neighbor scores higher in-sample — refer it to §10b, "
                              "do not just take it"}
        else:
            row |= {"verdict": "plateau", "action": "ok"}
        rows.append(row)
    return pd.DataFrame(rows).set_index("param")[cols]
