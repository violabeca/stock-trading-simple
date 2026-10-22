"""pooling_lab: the targets, the within-stock AUC, a tree that chooses its own leaf and rounds,
and the three approaches telling a pattern every stock shares from one that flips between
stocks. Synthetic data only (no DB).
"""

import importlib.util

import numpy as np
import pytest

from trading_simple import strategies

_PATH = strategies.STRATEGIES_DIR / "darts_forecast" / "pooling_lab.py"
_SPEC = importlib.util.spec_from_file_location("test_pooling_lab_module", _PATH)
PL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(PL)


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    """Fewer rounds: the synthetic patterns are strong."""
    monkeypatch.setattr(PL, "MAX_ITER", 60)


def rows(n_stocks=8, n_days=160, seed=0):
    """Every stock on every day, three inputs of pure noise."""
    rng = np.random.default_rng(seed)
    d = np.repeat(np.arange(n_days), n_stocks)
    k = np.tile(np.arange(n_stocks), n_days)
    return rng, rng.normal(size=(len(d), 3)).astype(np.float32), d, k


def split(n_days=160):
    """Days 0-89 learn, every 4th of them chooses leaf and rounds, 90-159 are graded."""
    days = np.arange(90)
    return days[days % 4 != 3], days[days % 4 == 3], np.arange(90, n_days)


def test_targets_are_moved_beyond_theta_and_up_on_every_day():
    t = PL.targets(np.array([-1.2, -0.3, 0.0, 0.4, 0.6, np.nan]), 0.5)
    np.testing.assert_array_equal(t["moved"][:5], [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(t["up"][:5], [0, 0, 0, 1, 1])
    assert np.isnan(t["moved"][5]) and np.isnan(t["up"][5])


def test_auc_is_read_inside_each_stock():
    k = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    y = np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    auc = PL.auc_by_stock(y, p, k, 4)
    assert auc[0] == 1.0 and auc[1] == 0.0          # stock 1: every pair in the wrong order
    assert np.isnan(auc[2]) and np.isnan(auc[3])    # one class / no rows


def test_a_tree_chooses_leaf_and_rounds_and_a_single_class_gives_a_constant():
    rng, X, d, k = rows()
    y = (X[:, 0] + rng.normal(0, 0.5, len(X)) > 0).astype(int)
    model, info = PL.fit_tree(X[:600], y[:600], X[600:900], y[600:900])
    assert info["leaf"] in PL.LEAVES and 1 <= info["rounds"] <= PL.MAX_ITER
    assert info["held_loss"] < info["prior_loss"]
    const, info = PL.fit_tree(X[:50], np.zeros(50), X[50:60], y[50:60])
    share = y[50:60].sum() / 60                      # the share over the fit and held-out rows
    assert info["rounds"] == 0 and np.allclose(const.predict_proba(X[:3])[:, 1], share)


def test_a_shared_pattern_is_found_by_every_approach_and_best_by_the_pooled_ones():
    rng, X, d, k = rows()
    y = (rng.random(len(X)) < 1 / (1 + np.exp(-1.5 * X[:, 0]))).astype(float)
    out = PL.compare(X, y, k, d, *split(), 8)
    mean = out["auc"].mean()
    assert (mean > 0.7).all()
    assert mean["pooled"] > mean["per stock"]       # the same pattern, 8x the rows
    assert out["transfer"].shape == (8, 8)
    off = out["transfer"][~np.eye(8, dtype=bool)]
    assert np.nanmean(off) > 0.7                     # every stock's model works on the others


def test_a_pattern_that_flips_between_stocks_is_missed_by_the_blind_pooled_model():
    rng, X, d, k = rows()
    sign = np.where(np.arange(8) % 2 == 0, 1.0, -1.0)[k]
    y = (rng.random(len(X)) < 1 / (1 + np.exp(-1.5 * sign * X[:, 0]))).astype(float)
    out = PL.compare(X, y, k, d, *split(), 8)
    mean = out["auc"].mean()
    assert abs(mean["pooled"] - 0.5) < 0.06          # the halves cancel
    assert mean["per stock"] > 0.7 and mean["pooled + stock"] > 0.7
    t = out["transfer"]
    own, same = np.diag(t), t[np.equal.outer(np.arange(8) % 2, np.arange(8) % 2)]
    opposite = t[np.not_equal.outer(np.arange(8) % 2, np.arange(8) % 2)]
    assert own.mean() > 0.7 and np.nanmean(same) > 0.7 and np.nanmean(opposite) < 0.3


def test_planted_labels_keep_the_rate_and_carry_the_sign_asked_for():
    rng, X, d, k = rows(n_stocks=10, n_days=400)
    none = PL.plant(X, k, 0, "none", strength=1.0, rate=0.3, n_stocks=10)
    assert abs(none.mean() - 0.3) < 0.02 and abs(np.corrcoef(none, X[:, 0])[0, 1]) < 0.05
    shared = PL.plant(X, k, 0, "shared", strength=1.0, rate=0.3, n_stocks=10)
    assert np.corrcoef(shared, X[:, 0])[0, 1] > 0.2
    flips = PL.plant(X, k, 0, "per stock", strength=1.0, rate=0.3, n_stocks=10)
    per = np.array([np.corrcoef(flips[k == s], X[k == s, 0])[0, 1] for s in range(10)])
    assert (per > 0.1).sum() == 5 and (per < -0.1).sum() == 5
    with pytest.raises(ValueError, match="plant"):
        PL.plant(X, k, 0, "tomorrow", strength=1.0, rate=0.3, n_stocks=10)
