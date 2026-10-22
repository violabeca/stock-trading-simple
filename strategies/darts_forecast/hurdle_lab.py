"""Hurdle helpers (``darts_forecast_hurdle_marimo.py``): the tree-model sibling of the movers
notebook. Same question (will a stock move at least ``theta`` percent between the 13:30 decision
and the horizon's end, and which way?), same stock-days, same labels, same calls per day, same
ladder; the model is gradient-boosted trees on a short table of numbers instead of a TSMixer on
the raw path.

One row is one STOCK-DAY. The columns come in four groups (:data:`GROUPS`):

* ``activity``: what the stock's previous sessions did in this clock window (the movers
  notebook's hurdle rule, as two numbers);
* ``day``: the gap, the previous sessions' move and the same-hour move (``tsmixer_lab``'s day
  inputs);
* ``morning``: a few summaries of the input minutes (returns and volatility over the last 15 / 30 /
  60 / L minutes, range, volume, flow, market and sector over the last 30-60), every window ending
  AT the decision;
* ``descriptors``: the stock's typical day move, log price and log trailing dollar volume (no
  one-hot id, so a stock never gets a column of its own).

The model (:func:`fit`) is two trees (``moved``: will it move, on all rows; ``up``: which way, on
the rows that moved) or one direct three-class tree as the control. :func:`proba` turns either
into ``[P(down), P(flat), P(up)]`` so ``movers_lab.ladder`` scores it unchanged.

It builds on ``movers_lab`` (the outcome, the labels, the activity, the picks, the ladder; loaded
by path as :data:`ml`) and edits nothing there. ``lab`` (``tsmixer_lab``) is passed in where the
day inputs and the validity mask are needed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    """A helper module of this folder, loaded by path once (shared with the other notebooks)."""
    key = f"darts_{name}"
    path = HERE / f"{name}.py"
    mod = sys.modules.get(key)
    if mod is not None and Path(getattr(mod, "__file__", "")) == path:
        return mod
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


ml = _sibling("movers_lab")
GROUPS = ("activity", "day", "morning", "descriptors")
DAY = ("gap", "prev", "same_hour")
DOLLAR_DAYS = 20                 # sessions behind the trailing dollar volume
#: The trees' knobs, fixed (nothing is searched): shallow, slow, regularized, big leaves.
KNOBS = {"max_depth": 3, "learning_rate": 0.05, "l2_regularization": 1.0,
         "min_samples_leaf": 100, "max_bins": 64}


# ---------------------------------------------------------------------------------------------
# The table: one row per valid stock-day.
# ---------------------------------------------------------------------------------------------
def _morning(F: dict, D: int, L: int) -> tuple[list[np.ndarray], list[str]]:
    """Summaries of the input minutes ``[D - L, D)``, each (sessions x stocks). A window never
    reaches past the decision ``D``; NaN counts as 0 (a minute without a trade)."""
    bar = F["bar"]

    def tail(name: str, w: int) -> np.ndarray:
        return np.nan_to_num(np.asarray(bar[name][:, D - w:D, :], float))

    cols, names = [], []
    for w in sorted({w for w in (15, 30, 60, L) if w <= L}):
        cols.append(tail("path", w).sum(axis=1) / np.sqrt(w))
        names.append(f"ret{w}")
    for w in sorted({w for w in (30, L) if w <= L}):
        cols.append(np.sqrt((tail("path", w) ** 2).mean(axis=1)))
        names.append(f"vol{w}")
        for f in ("range", "volume"):
            if f in bar:
                cols.append(tail(f, w).mean(axis=1))
                names.append(f"{f}{w}")
    if "flow" in bar and L >= 30:
        cols.append(tail("flow", 30).mean(axis=1))
        names.append("flow30")
    for f in ("market", "sector"):
        if f in bar and L >= 60:
            cols.append(tail(f, 60).sum(axis=1) / np.sqrt(60))
            names.append(f"{f}60")
    return cols, names


def _descriptors(F: dict, D: int) -> tuple[list[np.ndarray], list[str]]:
    """The stock's typical close-to-close move (previous sessions), the log of its last price
    before the decision and the log of its trailing dollar volume (previous sessions)."""
    pp = F["pp"]
    dollars = np.nansum(pp["volume"] * pp["cf"], axis=1)
    trail = pd.DataFrame(dollars).rolling(DOLLAR_DAYS, min_periods=DOLLAR_DAYS // 2).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        cols = [np.log(np.maximum(pp["day_rms"], 1e-6)), np.log(pp["cf"][:, D - 1, :]),
                np.log(np.maximum(trail.shift(1).to_numpy(), 1.0))]
    return cols, ["log_day_rms", "log_price", "log_dollars"]


def table(lab, F: dict, cell, days, groups, *, prev_days: int = 1, theta: float = 1.0) -> tuple:
    """The feature table of every valid stock-day of ``days``: the same rows ``movers_lab.examples``
    would build (``lab.valid``, a known class, a known trailing activity), in day then stock order.
    Every column is known at the decision: a window ends at it, a trailing statistic stops at the
    previous session.

    Args:
        lab: The ``tsmixer_lab`` module (validity mask, day inputs).
        F: ``lab.build``'s inputs.
        cell: (1, offset, input, output) minutes.
        days: Session indices.
        groups: Which of :data:`GROUPS` to include.
        prev_days: Sessions behind the "prev" day input.
        theta: The size of a "move", in percent.

    Returns:
        ``(X (rows x columns, float32, NaN allowed), column names, class, day, stock, activity)``.
    """
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        raise ValueError(f"unknown feature groups {unknown}; known: {list(GROUPS)}")
    _, off, L, _ = cell
    D = off + L
    cls = ml.labels(ml.moves(F, cell), theta)
    act = ml.activity(F, cell, theta)
    cols, names = [], []
    if "activity" in groups:
        cols += [act["log_mean"], act["big"]]
        names += ["act_log_mean", "act_big"]
    if "day" in groups:
        day = lab.day_inputs(F, cell, list(DAY), prev_days)
        cols += [day[..., i] for i in range(day.shape[-1])]
        names += list(DAY)
    if "morning" in groups:
        c, n = _morning(F, D, L)
        cols, names = cols + c, names + n
    if "descriptors" in groups:
        c, n = _descriptors(F, D)
        cols, names = cols + c, names + n
    if not cols:
        raise ValueError("no feature group chosen")
    ok = lab.valid(F, cell) & (cls >= 0) & np.isfinite(act["log_mean"])
    days = np.asarray(days, int)
    d, k = np.nonzero(ok[days])
    d = days[d]
    cube = np.stack(cols, axis=-1)
    return cube[d, k].astype(np.float32), names, cls[d, k], d, k, act["mean"][d, k]


# ---------------------------------------------------------------------------------------------
# The trees.
# ---------------------------------------------------------------------------------------------
def spread_held(dates, days, every: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """``days`` split into ``(fit, held)``: every ``every``-th calendar week is held out to choose
    the number of boosting rounds.

    Spread over the whole window, the held days see its calm and its busy months alike. The last
    days of a window can be one regime: the 2025 holidays, where 13% of the stock-days moved
    against 28% over the year, gave the "up" tree 45 rows and chose anything from 38 to 286
    rounds.
    """
    days = np.asarray(days, int)
    week = np.asarray((pd.to_datetime(np.asarray(dates)[days]) - pd.Timestamp("2000-01-03")).days)
    held = week // 7 % every == every - 1
    return days[~held], days[held]


def _tree(X, y, Xh, yh, *, max_iter: int, seed: int, hp: dict) -> tuple:
    """One boosted classifier whose number of rounds minimizes the held-out log loss (fit
    ``max_iter`` rounds on the fit rows, read every stage on the held-out rows), then fit again
    with that count on the fit AND the held-out rows. Returns ``(model, rounds, held-out loss,
    held-out loss of the fit rows' class shares)``."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss

    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("the fit rows hold a single class")
    make = lambda n: HistGradientBoostingClassifier(  # noqa: E731
        max_iter=n, early_stopping=False, random_state=seed, **{**KNOBS, **hp})
    losses = [log_loss(yh, p, labels=classes) for p in make(max_iter).fit(X, y)
              .staged_predict_proba(Xh)]
    best = int(np.argmin(losses)) + 1
    share = np.tile([np.mean(y == c) for c in classes], (len(yh), 1))
    model = make(best).fit(np.concatenate([X, Xh]), np.concatenate([y, yh]))
    return model, best, float(losses[best - 1]), float(log_loss(yh, share, labels=classes))


def fit(X, cls, Xh, clsh, *, kind: str = "hurdle", max_iter: int = 300, seed: int = 0,
        **hp) -> dict:
    """Fit the trees: the held-out rows choose the number of rounds, then the trees learn from
    the fit and the held-out rows with that count.

    Args:
        X: Fit features (rows x columns); ``cls`` their classes (0 down, 1 flat, 2 up).
        Xh: Held-out features; ``clsh`` their classes.
        kind: "hurdle" (``moved`` on every row, ``up`` on the rows that moved) or "direct" (one
            three-class tree).
        max_iter: Most boosting rounds tried.
        seed: Random seed.
        **hp: Overrides of :data:`KNOBS` (e.g. ``min_samples_leaf`` for a tiny table).

    Returns:
        ``{"kind", "trees" (name -> model), "info" (name -> rounds, held-out loss, prior loss)}``.
    """
    cls, clsh = np.asarray(cls), np.asarray(clsh)
    if kind == "direct":
        jobs = {"direct": (cls, clsh, np.ones(len(cls), bool), np.ones(len(clsh), bool))}
    elif kind == "hurdle":
        jobs = {"moved": ((cls != 1).astype(int), (clsh != 1).astype(int),
                          np.ones(len(cls), bool), np.ones(len(clsh), bool)),
                "up": ((cls == 2).astype(int), (clsh == 2).astype(int), cls != 1, clsh != 1)}
    else:
        raise ValueError(f"kind {kind!r}: 'hurdle' or 'direct'")
    trees, info = {}, {}
    for name, (y, yh, rows, rowsh) in jobs.items():
        model, rounds, held, prior = _tree(X[rows], y[rows], Xh[rowsh], yh[rowsh],
                                           max_iter=max_iter, seed=seed, hp=hp)
        trees[name], info[name] = model, {"rounds": rounds, "held_loss": held,
                                          "prior_loss": prior}
    return {"kind": kind, "trees": trees, "info": info}


def proba(model: dict, X) -> np.ndarray:
    """``[P(down), P(flat), P(up)]`` of every row. The hurdle model multiplies its two answers:
    P(up) = P(moved) x P(up | moved), P(down) = P(moved) x (1 - P(up | moved))."""
    t = model["trees"]
    if model["kind"] == "direct":
        p = t["direct"].predict_proba(X)
        if p.shape[1] != 3:
            raise ValueError("the direct tree did not see all three classes")
        return p.astype(np.float32)
    moved = t["moved"].predict_proba(X)[:, 1]
    up = t["up"].predict_proba(X)[:, 1]
    return np.column_stack([moved * (1 - up), 1 - moved, moved * up]).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# The luck band.
# ---------------------------------------------------------------------------------------------
def shuffle_band(cls, d, act, ps: dict, *, K: int, prior: int, draws: int = 1000,
                 seed: int = 0) -> dict:
    """What the ladder's score (``excess_z_prior``) looks like when the labels carry no
    information: inside every day the classes are shuffled among that day's stocks (a day keeps
    its number of down / flat / up, its stocks keep their ``p`` and their activity), ``draws``
    times. ONE shuffle per draw serves every model of ``ps``, so the best model's z can be held
    against the best of the shuffled models: the luck bar for picking the best of several.

    Args:
        cls: Class of every row (the same rows for every model).
        d: Its day.
        act: Its trailing activity.
        ps: Model name -> class probabilities (rows x 3).
        K: Calls per day.
        prior: The prior side (``movers_lab.prior_side`` of the days the models learn from).
        draws: Shuffles (a 97.5th percentile needs about a thousand).
        seed: Random seed.

    Returns:
        ``{"names", "real" (each model's z on the real labels, the ladder's own), "z" (draws x
        models), "best" (the best model's z of every draw)}``.
    """
    cls, d, act = np.asarray(cls), np.asarray(d), np.asarray(act, float)
    days, code = np.unique(d, return_inverse=True)
    nv = ml.picks(np.nan_to_num(act, nan=-1.0), d, K)
    picked = [ml.calls(p, d, act, K) for p in ps.values()]

    def day_mean(idx, x):
        return np.bincount(code[idx], weights=x, minlength=len(days)) / np.bincount(
            code[idx], minlength=len(days))

    def scores(c) -> list[float]:
        base = day_mean(nv, c[nv] == prior)
        return [ml._z(pd.Series(day_mean(mp, c[mp] == side) - base))[2] for mp, side in picked]

    rng = np.random.default_rng(seed)
    by_day = np.argsort(d, kind="stable")
    z = np.empty((draws, len(picked)))
    for i in range(draws):
        shuffled = np.empty_like(cls)
        shuffled[by_day] = cls[np.lexsort((rng.random(len(d)), d))]
        z[i] = scores(shuffled)
    return {"names": list(ps), "real": np.array(scores(cls)), "z": z, "best": z.max(axis=1)}
