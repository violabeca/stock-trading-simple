"""Do the stocks differ? (``darts_forecast_hurdle_marimo.py``, section "Do the stocks differ?"):
before anyone trains one model per stock, check whether an input acts differently on different
stocks at all, more than the noise of a few hundred stock-days allows.

For every input and outcome (``moved``: moved at least ``theta`` either way, on every stock-day;
``up``: went up, on the stock-days that moved) each stock gets its OWN slope: the outcome on the
input standardized inside the stock (its own level and spread taken out), by least squares, with
a heteroskedasticity-robust (HC1) standard error, both divided by p(1 - p) of the stock's own
outcome rate: a slope in log-odds per standard deviation. In probability points an input that
raises every stock's odds alike looks three times stronger on a stock that moves on 45% of its
days than on one that moves on 15%; a pooled model reads that from the stock's activity, so it
must not count as a difference (measured: in probability points it did, 88% of the time). Two
questions per input and outcome:

* **Do the slopes differ more than their noise?** Cochran's Q over the stocks, read against the
  same Q after the stocks are relabelled at random INSIDE each day (a day keeps its shared mood,
  a stock keeps its number of rows on every day; only "which stock" is lost). The textbook
  chi-squared p-value is NOT used: with 50 moved days per stock it says "different" on 20% of
  stocks that are all alike (measured; 6-10% at 200). ``tau`` is the spread of the true slopes
  (the slopes' spread beyond the relabelled one), ``I2`` its share of the whole spread.
* **Do the differences come back?** The correlation, over stocks, of the TRAIN slopes with the
  VALIDATION slopes (disjoint days), against that correlation with the VALIDATION slopes shuffled
  over the stocks. A model per stock can only profit from a difference that comes back.

The rows are ``hurdle_lab.table``'s (one per valid stock-day, every column known at the
decision); nothing here reads prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ("moved", "up")
MIN_ROWS = 20   # a stock's slope needs at least this many rows ...
MIN_EACH = 5    # ... and at least this many of each outcome


def slopes(x: np.ndarray, y: np.ndarray, k: np.ndarray, n_stocks: int) -> tuple:
    """Per stock, the least-squares slope of ``y`` on ``x`` standardized inside the stock, and
    its robust (HC1) standard error, both divided by p(1 - p) of the stock's rate of ``y``
    (log-odds per standard deviation, for a small effect). NaN for a stock with under
    :data:`MIN_ROWS` rows, under :data:`MIN_EACH` of either outcome, or a constant ``x``.

    Args:
        x: One input per row (NaN rows are skipped).
        y: The 0/1 outcome per row.
        k: The stock of each row (0 .. ``n_stocks`` - 1); or several labellings at once
            (labellings x rows), one answer per labelling.
        n_stocks: How many stocks.

    Returns:
        ``(slope, standard error)``, each of length ``n_stocks`` (labellings x ``n_stocks``).
    """
    m = np.isfinite(x)
    k = np.asarray(k)[..., m]
    P = 1 if k.ndim == 1 else len(k)
    lab = (np.atleast_2d(k) + n_stocks * np.arange(P)[:, None]).ravel()  # labelling x stock
    x, y = np.tile(np.asarray(x, float)[m], P), np.tile(np.asarray(y, float)[m], P)
    s = lambda v: np.bincount(lab, v, minlength=P * n_stocks)  # noqa: E731  (a per-stock sum)
    n, ys = s(np.ones(len(lab))), s(y)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = s(x) / n, ys / n
        sx = np.sqrt(s((x - mx[lab]) ** 2) / n)
        z, yc = (x - mx[lab]) / sx[lab], y - my[lab]          # per stock: sum of z² is n
        b = s(z * yc) / n
        se = np.sqrt(n / (n - 2) * s(z ** 2 * (yc - b[lab] * z) ** 2)) / n
        b, se = b / (my * (1 - my)), se / (my * (1 - my))   # probability -> log-odds
    bad = (n < MIN_ROWS) | (np.minimum(ys, n - ys) < MIN_EACH) | ~(sx > 1e-9 * (np.abs(mx) + 1))
    b[bad], se[bad] = np.nan, np.nan
    shape = (n_stocks,) if k.ndim == 1 else (P, n_stocks)
    return b.reshape(shape), se.reshape(shape)


def _q(b: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Cochran's Q (last axis: the stocks): the precision-weighted spread of the slopes around
    their weighted mean; NaN with under 3 slopes."""
    ok = np.isfinite(b) & (se > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(ok, 1 / se ** 2, 0.0)
        bb = np.where(ok, b, 0.0)
        mean = (w * bb).sum(-1, keepdims=True) / w.sum(-1, keepdims=True)
        q = (w * (bb - mean) ** 2).sum(-1)
    return np.where(ok.sum(-1) >= 3, q, np.nan)


def relabel(d: np.ndarray, k: np.ndarray, rng) -> np.ndarray:
    """``k`` with the stocks shuffled among the rows of each day."""
    d, k = np.asarray(d), np.asarray(k)
    out = np.empty_like(k)
    out[np.argsort(d, kind="stable")] = k[np.lexsort((rng.random(len(d)), d))]
    return out


def carry(b_a: np.ndarray, b_b: np.ndarray, *, perms: int = 2000, seed: int = 0) -> tuple:
    """The correlation over stocks of two periods' slopes, and the share of ``perms`` shuffles of
    the second period's slopes over the stocks that reach it (one-sided: a difference that comes
    back correlates positively).

    Returns:
        ``(r, p)``; NaN with under 4 stocks that have both slopes.
    """
    ok = np.isfinite(b_a) & np.isfinite(b_b)
    if ok.sum() < 4:
        return np.nan, np.nan
    std = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    a, c = std(b_a[ok]), std(b_b[ok])
    r = float(a @ c / len(a))
    order = np.argsort(np.random.default_rng(seed).random((perms, len(c))), axis=1)
    null = c[order] @ a / len(a)
    return r, float((1 + (null >= r - 1e-12).sum()) / (1 + perms))


def differ(train: tuple, valid: tuple, cols, n_stocks: int, *, perms: int = 500,
           seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Every input and outcome: do the stocks' slopes differ (TRAIN), and do the differences
    come back (VALIDATION)?

    Args:
        train: ``(X, cls, d, k)`` of the TRAIN rows (``hurdle_lab.table``: features, class 0
            down / 1 flat / 2 up, day, stock).
        valid: The same for the VALIDATION rows.
        cols: The names of ``X``'s columns.
        n_stocks: How many stocks.
        perms: Relabellings behind each ``Q p`` (shared by every input) and stock shuffles behind
            each ``carry p``.
        seed: Random seed.

    Returns:
        ``(table, per_stock)``: one row per outcome and input: ``stocks`` (with a slope),
        ``pooled`` (the precision-weighted slope, log-odds per standard deviation of the
        input), ``tau``, ``I2``, ``Q p``, ``carry r``, ``carry p`` and ``typical se`` (the median
        stock's standard error); ``per_stock[(outcome, input)] = (TRAIN slopes, VALIDATION
        slopes)``.
    """
    rng = np.random.default_rng(seed)
    rows, per_stock = [], {}
    for outcome in OUTCOMES:
        parts = []
        for X, cls, d, k in (train, valid):
            cls = np.asarray(cls)
            keep = cls != 1 if outcome == "up" else np.ones(len(cls), bool)
            y = (cls == 2) if outcome == "up" else (cls != 1)
            parts.append((np.asarray(X, float)[keep], y[keep].astype(float),
                          np.asarray(d)[keep], np.asarray(k)[keep]))
        (Xa, ya, da, ka), (Xb, yb, _, kb) = parts
        shuffled = np.stack([relabel(da, ka, rng) for _ in range(perms)])
        for j, col in enumerate(cols):
            b_a, se_a = slopes(Xa[:, j], ya, ka, n_stocks)
            b_b, _ = slopes(Xb[:, j], yb, kb, n_stocks)
            b_null, se_null = slopes(Xa[:, j], ya, shuffled, n_stocks)
            q, q_null = float(_q(b_a, se_a)), _q(b_null, se_null)
            spread = float(np.nanvar(b_a)) if np.isfinite(b_a).sum() >= 3 else np.nan
            luck = np.nanvar(b_null[np.isfinite(b_null).sum(axis=1) >= 3], axis=1)
            tau2 = max(0.0, spread - luck.mean()) if np.isfinite(spread) and len(luck) else np.nan
            ok = np.isfinite(b_a) & (se_a > 0)
            w = 1 / se_a[ok] ** 2
            r, p = carry(b_a, b_b, perms=max(perms, 1000), seed=seed)
            rows.append({
                "outcome": outcome, "input": col, "stocks": int(ok.sum()),
                "pooled": float(w @ b_a[ok] / w.sum()) if ok.sum() else np.nan,
                "tau": float(np.sqrt(tau2)), "I2": tau2 / spread if spread > 0 else np.nan,
                "Q p": float((1 + (q_null >= q - 1e-12).sum()) / (1 + perms))
                if np.isfinite(q) else np.nan,
                "carry r": r, "carry p": p,
                "typical se": float(np.nanmedian(se_a)) if ok.any() else np.nan})
            per_stock[(outcome, col)] = (b_a, b_b)
    return pd.DataFrame(rows), per_stock
