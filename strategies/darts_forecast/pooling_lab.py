"""Pooling (``darts_forecast_pooling_marimo.py``): one model for every stock, or one per stock?

The same stock-days, inputs and targets, learned three ways (:data:`APPROACHES`):

* ``pooled``: one model for all stocks, blind to which stock a row is;
* ``pooled + stock``: one model that also reads the stock as a category, so a tree can split on
  it wherever enough of a stock's rows say that stock behaves differently;
* ``per stock``: one model per stock, learned from that stock's rows alone.

Every model is the same kind of boosted tree, and each chooses its own leaf size and number of
rounds on its own held-out rows, so no approach runs on knobs sized for another. The measure is
the WITHIN-STOCK AUC on the graded rows: for one stock, how often the model rates a day that
turned out positive above one that did not (0.5: it cannot tell them apart). It is read inside
each stock, so a model earns nothing for knowing that one stock moves more than another, only
for telling a stock's days apart.

The rows are ``hurdle_lab.table``'s; the targets come from ``movers_lab.moves``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

APPROACHES = ("pooled", "pooled + stock", "per stock")
TARGETS = ("moved", "up")
PLANTS = ("none", "shared", "per stock")
LEAVES = (10, 30, 100)  # the smallest leaf (rows) each model may choose
MAX_ITER = 300          # the most boosting rounds tried
#: The trees' fixed knobs: shallow and slow. 255 bins leave room for one category per stock.
KNOBS = {"max_depth": 3, "learning_rate": 0.05, "l2_regularization": 1.0, "max_bins": 255}


def targets(mv: np.ndarray, theta: float) -> dict[str, np.ndarray]:
    """The 0/1 targets of every move (percent): ``moved`` (at least ``theta`` either way) and
    ``up`` (above 0: every day counts, no threshold). NaN where the move is unknown."""
    mv = np.asarray(mv, float)
    known = np.isfinite(mv)
    with np.errstate(invalid="ignore"):
        out = {"moved": np.abs(mv) >= theta, "up": mv > 0}
    return {name: np.where(known, v, np.nan) for name, v in out.items()}


class _Constant:
    """A model that always says one probability (a stock whose rows hold a single class)."""

    def __init__(self, share: float):
        self.share = share

    def predict_proba(self, X) -> np.ndarray:
        """``[1 - share, share]`` for every row."""
        return np.tile([1 - self.share, self.share], (len(X), 1))


def fit_tree(X, y, Xh, yh, *, categorical: list[int] | None = None, seed: int = 0) -> tuple:
    """One boosted classifier that chooses its leaf size (:data:`LEAVES`) and number of rounds on
    the held-out rows (the lowest log loss), then learns from the fit AND the held-out rows with
    them. Fit rows of a single class, or no held-out rows, give a constant.

    Args:
        X: Fit inputs (rows x columns); ``y`` their 0/1 target.
        Xh: Held-out inputs; ``yh`` their target.
        categorical: Columns read as categories (the stock), or None.
        seed: Random seed.

    Returns:
        ``(model, {"leaf", "rounds", "held_loss", "prior_loss"})``; ``prior_loss`` is the
        held-out loss of always saying the fit rows' share.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss

    y, yh = np.asarray(y, int), np.asarray(yh, int)
    prior = (float(log_loss(yh, np.full(len(yh), y.mean()), labels=[0, 1]))
             if len(yh) and 0 < y.mean() < 1 else np.nan)
    if len(np.unique(y)) < 2 or not len(yh):
        share = float(np.r_[y, yh].mean()) if len(y) + len(yh) else 0.5
        return _Constant(share), {"leaf": 0, "rounds": 0, "held_loss": prior,
                                  "prior_loss": prior}
    extra = {"categorical_features": categorical} if categorical else {}

    def make(leaf, n):
        return HistGradientBoostingClassifier(
            max_iter=n, min_samples_leaf=leaf, early_stopping=False, random_state=seed,
            **KNOBS, **extra)

    best = (np.inf, LEAVES[0], 1)
    for leaf in LEAVES:
        losses = [log_loss(yh, p[:, 1], labels=[0, 1])
                  for p in make(leaf, MAX_ITER).fit(X, y).staged_predict_proba(Xh)]
        i = int(np.argmin(losses))
        if losses[i] < best[0]:
            best = (losses[i], leaf, i + 1)
    loss, leaf, rounds = best
    model = make(leaf, rounds).fit(np.concatenate([X, Xh]), np.concatenate([y, yh]))
    return model, {"leaf": leaf, "rounds": rounds, "held_loss": float(loss),
                   "prior_loss": prior}


def auc_by_stock(y, p, k, n_stocks: int) -> np.ndarray:
    """Per stock, the AUC of ``p`` for the 0/1 ``y`` on that stock's rows: the share of
    (positive, negative) pairs of its days that ``p`` puts in the right order. NaN for a stock
    without both classes or without predictions."""
    from sklearn.metrics import roc_auc_score

    y, p, k = np.asarray(y), np.asarray(p, float), np.asarray(k)
    out = np.full(n_stocks, np.nan)
    for s in range(n_stocks):
        m = (k == s) & np.isfinite(p)
        if 0 < y[m].sum() < m.sum():
            out[s] = roc_auc_score(y[m], p[m])
    return out


def compare(X, y, k, d, fit_days, held_days, graded_days, n_stocks: int, *,
            seed: int = 0) -> dict:
    """The three approaches on one target, each learned on the fit days (leaf and rounds chosen
    on the held-out days) and measured on the graded days, stock by stock.

    Args:
        X: The inputs of every row (``hurdle_lab.table``).
        y: The 0/1 target of every row (NaN: unknown, left out).
        k: The stock of every row (0 .. ``n_stocks`` - 1).
        d: The session of every row.
        fit_days: Sessions the models learn from.
        held_days: Sessions that choose each model's leaf size and rounds (then learned too).
        graded_days: Sessions the models are measured on.
        n_stocks: How many stocks (at most 255: one tree category each).
        seed: Random seed.

    Returns:
        ``{"auc"`` (stocks x approaches: within-stock AUC on the graded rows), ``"transfer"``
        (stocks x stocks: the AUC of the model learned on the ROW's stock, on the graded days of
        the COLUMN's stock; the diagonal is each stock's own model), ``"info"`` (approach -> the
        chosen leaf and rounds; per stock: one entry per stock, None without fit rows),
        ``"rows"`` and ``"rate"`` (graded rows and share of 1s per stock)``}``.
    """
    y, k, d = np.asarray(y, float), np.asarray(k), np.asarray(d)
    if n_stocks > KNOBS["max_bins"]:
        raise ValueError(f"{n_stocks} stocks: at most {KNOBS['max_bins']} tree categories")
    ok = np.isfinite(y)
    f, h, g = (ok & np.isin(d, days) for days in (fit_days, held_days, graded_days))
    yi = np.where(ok, y, 0).astype(int)
    Xs = np.column_stack([X, k]).astype(np.float32)   # + the stock, read as a category
    p, info = {}, {}
    model, info["pooled"] = fit_tree(X[f], yi[f], X[h], yi[h], seed=seed)
    p["pooled"] = model.predict_proba(X[g])[:, 1]
    model, info["pooled + stock"] = fit_tree(Xs[f], yi[f], Xs[h], yi[h],
                                             categorical=[X.shape[1]], seed=seed)
    p["pooled + stock"] = model.predict_proba(Xs[g])[:, 1]
    cross = np.full((n_stocks, int(g.sum())), np.nan)  # every stock's model on every graded row
    info["per stock"] = []
    for s in range(n_stocks):
        fs, hs = f & (k == s), h & (k == s)
        if not fs.any():
            info["per stock"].append(None)
            continue
        model, i = fit_tree(X[fs], yi[fs], X[hs], yi[hs], seed=seed)
        cross[s] = model.predict_proba(X[g])[:, 1]
        info["per stock"].append(i)
    kg, yg = k[g], yi[g]
    p["per stock"] = cross[kg, np.arange(len(kg))]
    return {"auc": pd.DataFrame({a: auc_by_stock(yg, p[a], kg, n_stocks) for a in APPROACHES}),
            "transfer": np.vstack([auc_by_stock(yg, cross[s], kg, n_stocks)
                                   for s in range(n_stocks)]),
            "info": info, "rows": np.bincount(kg, minlength=n_stocks),
            "rate": np.bincount(kg, weights=yg, minlength=n_stocks) / np.maximum(
                np.bincount(kg, minlength=n_stocks), 1)}


def plant(X, k, column: int, kind: str, *, strength: float, rate: float, n_stocks: int,
          seed: int = 0) -> np.ndarray:
    """0/1 labels planted on real rows, to see what each answer looks like: the log-odds are
    ``rate``'s plus ``strength`` x the input ``column`` (standardized inside each stock) x a sign:

    * ``none``: 0 (nothing to find);
    * ``shared``: +1 for every stock (one pattern for all);
    * ``per stock``: +1 for a random half of the stocks, -1 for the other half (one model for
      all sees the two halves cancel, unless other inputs tell the halves apart).

    Args:
        X: The real inputs (rows x columns).
        k: The stock of every row.
        column: The input that carries the planted effect.
        kind: One of :data:`PLANTS`.
        strength: Log-odds per standard deviation of the input.
        rate: The share of 1s with the input at its stock's mean.
        n_stocks: How many stocks.
        seed: Random seed.

    Returns:
        The planted 0/1 labels (float).
    """
    if kind not in PLANTS:
        raise ValueError(f"plant {kind!r}: one of {PLANTS}")
    rng = np.random.default_rng(seed)
    k = np.asarray(k)
    x = pd.Series(np.asarray(X, float)[:, column]).groupby(k).transform(
        lambda v: (v - v.mean()) / v.std(ddof=0) if v.std(ddof=0) > 0 else v * 0)
    x = np.nan_to_num(x.to_numpy())
    sign = {"none": np.zeros(n_stocks), "shared": np.ones(n_stocks),
            "per stock": np.where(rng.permutation(n_stocks) % 2 == 0, 1.0, -1.0)}[kind]
    logit = np.log(rate / (1 - rate)) + strength * sign[k] * x
    return (rng.random(len(x)) < 1 / (1 + np.exp(-logit))).astype(float)
