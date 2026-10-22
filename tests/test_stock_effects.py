"""stock_effects: per-stock slopes with robust errors, Cochran's Q over the stocks and the
TRAIN -> VALIDATION carry of the differences. Synthetic data only (no DB): a planted
stock-specific slope must be found and must carry; a slope shared by every stock and an input
without any effect must not look different; both tests must hold their size under the null.
"""

import importlib.util

import numpy as np

from trading_simple import strategies

_DIR = strategies.STRATEGIES_DIR / "darts_forecast"
spec = importlib.util.spec_from_file_location("test_stock_effects_mod", _DIR / "stock_effects.py")
SE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SE)


def period(rng, beta, n_days=250, move=0.35):
    """``n_days`` x stocks rows; input 0 tilts the side by ``beta[0][stock]`` (in probability per
    standard deviation), input 1 by ``beta[1][stock]``, input 2 is noise. A day's stocks share a
    market mood that pushes them all the same way (the shock that makes rows dependent)."""
    N = len(beta[0])
    k = np.tile(np.arange(N), n_days)
    z = rng.normal(0, 1, (len(k), 3))
    X = 2 + z * rng.uniform(0.5, 3, N)[k, None]       # each stock its own level and spread
    mood = np.repeat(rng.normal(0, 0.15, n_days), N)
    p_up = np.clip(0.5 + mood + beta[0][k] * z[:, 0] + beta[1][k] * z[:, 1], 0.02, 0.98)
    moved = rng.random(len(k)) < move
    cls = np.where(moved, np.where(rng.random(len(k)) < p_up, 2, 0), 1)
    return X, cls, np.repeat(np.arange(n_days), N), k


def test_the_slope_is_the_within_stock_least_squares_slope_on_the_standardized_input():
    rng = np.random.default_rng(0)
    k = np.repeat([0, 1], 300)
    x = rng.normal(5, 2, 600)
    y = (rng.random(600) < 0.3 + 0.1 * (x > 5) * (k == 1)).astype(float)
    b, se = SE.slopes(x, y, k, 3)
    for s in (0, 1):
        m = k == s
        z = (x[m] - x[m].mean()) / x[m].std()
        odds = y[m].mean() * (1 - y[m].mean())        # probability -> log-odds
        assert np.isclose(b[s], np.polyfit(z, y[m], 1)[0] / odds)
        classic = np.sqrt(((y[m] - np.polyval(np.polyfit(z, y[m], 1), z)) ** 2).sum()
                          / (m.sum() - 2) / (z ** 2).sum()) / odds
        assert 0.85 < se[s] / classic < 1.15          # robust ≈ classic when the noise is even
    assert np.isnan(b[2]) and np.isnan(se[2])         # a stock without rows has no slope


def test_a_stock_specific_slope_is_found_and_comes_back_a_shared_one_does_not_differ():
    rng = np.random.default_rng(1)
    N = 40
    sign = np.where(np.arange(N) % 2, 1.0, -1.0)
    beta = (0.15 * sign, np.full(N, 0.12))            # input 0: ±0.15 by stock; input 1: shared
    tab, per = SE.differ(period(rng, beta), period(rng, beta), ["own", "shared", "noise"], N,
                         perms=500)
    up = tab[tab["outcome"] == "up"].set_index("input")
    assert up.loc["own", "Q p"] < 0.005 and up.loc["own", "tau"] > 0.3  # no relabelling beat it
    assert up.loc["own", "carry r"] > 0.5 and up.loc["own", "carry p"] < 0.01
    assert abs(up.loc["own", "pooled"]) < 0.2         # pooled, the two halves cancel
    assert up.loc["shared", "pooled"] > 0.3 and up.loc["shared", "Q p"] > 0.01  # 0.12 x 4
    assert up.loc["noise", "Q p"] > 0.01 and up.loc["noise", "carry p"] > 0.01
    b_train, b_valid = per[("up", "own")]
    assert np.corrcoef(b_train, sign)[0, 1] > 0.7 and len(b_valid) == N


def test_both_tests_keep_their_size_when_every_stock_is_alike_even_with_few_moved_days():
    # ~50 moved days per stock: where the chi-squared p-value of Q says "different" on 20%
    rng = np.random.default_rng(2)
    N, runs = 30, 40
    beta = (np.full(N, 0.05), np.zeros(N))
    q_p, c_p = [], []
    for _ in range(runs):
        tab, _ = SE.differ(period(rng, beta, n_days=140), period(rng, beta, n_days=140),
                           ["a", "b", "c"], N, perms=100)
        q_p += list(tab["Q p"])
        c_p += list(tab["carry p"])
    q_p, c_p = np.array(q_p), np.array(c_p)
    assert 0.01 < (q_p < 0.05).mean() < 0.09          # about 5%, both outcomes
    assert 0.01 < (c_p < 0.05).mean() < 0.09


def test_relabel_keeps_every_day_and_every_stock_count_of_every_day():
    rng = np.random.default_rng(3)
    d = np.repeat(np.arange(50), 7)
    k = np.tile(np.arange(7), 50)
    keep = rng.random(len(d)) < 0.4                   # the moved rows: a different set each day
    d, k = d[keep], k[keep]
    kp = SE.relabel(d, k, rng)
    assert not np.array_equal(kp, k)
    for day in np.unique(d):
        assert sorted(kp[d == day]) == sorted(k[d == day])


def test_an_input_that_raises_every_stocks_odds_alike_does_not_differ_however_often_they_move():
    rng = np.random.default_rng(4)
    N, runs, flagged = 30, 25, 0
    for _ in range(runs):
        parts = []
        for n_days in (200, 120):
            k = np.tile(np.arange(N), n_days)
            z = rng.normal(0, 1, len(k))
            logit = np.log(np.linspace(0.15, 0.45, N) / (1 - np.linspace(0.15, 0.45, N)))[k]
            moved = rng.random(len(k)) < 1 / (1 + np.exp(-(logit + 0.4 * z)))
            parts.append((z[:, None], np.where(moved, 2, 1), np.repeat(np.arange(n_days), N), k))
        tab, _ = SE.differ(*parts, ["x"], N, perms=100)
        flagged += int(tab.set_index("outcome").loc["moved", "Q p"] < 0.05)
    assert flagged <= 4                               # about 5% of 25; in probability points: 88%
