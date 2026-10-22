"""hurdle_lab: the feature table (the rows the movers examples would build, every column known at
the decision), the hurdle / direct trees and the probabilities they hand to the movers ladder, and
the within-day shuffle band. Synthetic data only (no DB). The table needs ``tsmixer_lab`` (day
inputs, validity), which is skipped without the optional ``darts`` extra.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies

_DIR = strategies.STRATEGIES_DIR / "darts_forecast"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"test_{name}", _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HL = _load("hurdle_lab")
ML = HL.ml
D0, H = 240, 120                      # decision at 13:30, horizon to 15:30
CELL = (1, D0 - 60, 60, H)
SMALL = {"min_samples_leaf": 10}      # the real knob is sized for ~10,000 rows


def synthetic_F(n=140, N=4, seed=0):
    """A hand-made ``lab.build`` result: the sign of the last 60 minutes' average move decides
    the class of the horizon's move (up 1.5%, down -1.5%, else 0.2%)."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, (n, 390, N)).astype(np.float32)
    s = y[:, D0 - 60:D0, :].mean(axis=1)
    mv = np.where(s > 0.12, 1.5, np.where(s < -0.12, -1.5, 0.2))
    cf = np.full((n, 390, N), 100.0)
    cf[:, D0 + H - 1:, :] = 100.0 + mv[:, None, :]
    dates = list(pd.bdate_range("2025-03-03", periods=n).strftime("%Y-%m-%d"))
    pp = {"cf": cf, "usable": np.ones(n, bool), "volume": np.ones((n, 390, N)),
          "first": np.zeros((n, N), int), "day_rms": np.full((n, N), 0.02)}
    return {"pp": pp, "open": np.full((n, 390, N), 100.0), "y": y, "u": np.ones((n, 390, N)),
            "gap": np.zeros((n, N)),
            "prev_move": {1: np.zeros((n, N))}, "names": [f"S{i}" for i in range(N)],
            "dates": np.array(dates),
            "bar": {"path": y, "volume": rng.normal(0, 1, (n, 390, N)).astype(np.float32),
                    "range": rng.normal(0, 1, (n, 390, N)).astype(np.float32)}}


@pytest.fixture(scope="module")
def lab():
    pytest.importorskip("darts")
    return _load("tsmixer_lab")


def test_the_table_has_the_rows_the_movers_examples_would_build(lab):
    F = synthetic_F(n=40)
    X, cols, cls, d, k, act = HL.table(lab, F, CELL, range(15, 40), HL.GROUPS)
    ok = lab.valid(F, CELL) & (ML.labels(ML.moves(F, CELL), 1.0) >= 0) & np.isfinite(
        ML.activity(F, CELL, 1.0)["log_mean"])
    assert len(X) == len(cls) == len(d) == len(k) == len(act) == ok[15:40].sum() > 0
    assert X.shape[1] == len(cols) and len(set(cols)) == len(cols)
    assert {"act_log_mean", "gap", "ret60", "vol60", "log_price", "log_dollars"} <= set(cols)
    assert (np.diff(d) >= 0).all()                       # day, then stock
    np.testing.assert_array_equal(cls, ML.labels(ML.moves(F, CELL), 1.0)[d, k])
    with pytest.raises(ValueError, match="unknown feature groups"):
        HL.table(lab, F, CELL, range(15, 40), ["tomorrow"])


def test_no_column_reads_a_minute_at_or_after_the_decision(lab):
    F = synthetic_F(n=60)
    X0, *_ = HL.table(lab, F, CELL, range(20, 45), HL.GROUPS)
    F2 = synthetic_F(n=60)
    for name in F2["bar"]:
        F2["bar"][name][44, D0:, :] += 50.0              # session 44, from the decision on
    F2["y"][44, D0:, :] += 50.0
    F2["pp"]["volume"][44, D0:, :] *= 100
    X1, *_ = HL.table(lab, F2, CELL, range(20, 45), HL.GROUPS)
    np.testing.assert_array_equal(X0, X1)                # nothing of session 44 up to 44 moved


def test_the_hurdle_trees_learn_the_planted_signal_and_hand_the_ladder_probabilities(lab):
    F = synthetic_F(n=300, N=6)
    cols = ["morning"]
    Xf, _, cf, *_ = HL.table(lab, F, CELL, range(12, 200), ["activity", *cols])
    Xh, _, ch, *_ = HL.table(lab, F, CELL, range(200, 220), ["activity", *cols])
    Xv, _, cv, dv, _, av = HL.table(lab, F, CELL, range(220, 300), ["activity", *cols])
    prior = ML.prior_side(F, CELL, range(12, 220), 1.0)
    for kind in ("hurdle", "direct"):
        m = HL.fit(Xf, cf, Xh, ch, kind=kind, **SMALL)
        p = HL.proba(m, Xv)
        assert p.shape == (len(cv), 3) and np.allclose(p.sum(axis=1), 1, atol=1e-5)
        assert (p.argmax(axis=1) == cv).mean() > 0.7     # chance is ~0.4
        s = ML.ladder(cv, dv, av, p, K=2, prior=prior)
        assert s["dir_acc"] > 0.8 and s["side_z"] > 3     # the sides beat the prior side
    model = HL.fit(Xf, cf, Xh, ch, **SMALL)
    info = model["info"]
    assert info["moved"]["held_loss"] < info["moved"]["prior_loss"]
    assert info["up"]["held_loss"] < info["up"]["prior_loss"]
    # the final trees learn from the fit AND the held-out rows, with the held-out rounds
    from sklearn.ensemble import HistGradientBoostingClassifier
    again = HistGradientBoostingClassifier(
        max_iter=info["moved"]["rounds"], early_stopping=False, random_state=0,
        **{**HL.KNOBS, **SMALL}).fit(np.r_[Xf, Xh], np.r_[cf, ch] != 1)
    np.testing.assert_allclose(model["trees"]["moved"].predict_proba(Xv), again.predict_proba(Xv))


def test_the_held_out_weeks_spread_over_the_window():
    dates = np.array(pd.bdate_range("2025-01-06", "2025-12-31").strftime("%Y-%m-%d"))
    days = np.arange(5, len(dates))
    fit, held = HL.spread_held(dates, days, every=4)
    assert sorted(np.r_[fit, held].tolist()) == days.tolist() and not set(fit) & set(held)
    assert 0.2 < len(held) / len(days) < 0.3                # every 4th week
    weeks = pd.to_datetime(dates[held]).to_period("W").unique()
    assert len(weeks) >= 12                                 # spread over the year, not one block
    assert dates[held].min() < "2025-02-15" and dates[held].max() > "2025-11-15"


def test_the_probability_order_is_down_flat_up():
    class Fixed:
        def __init__(self, p1):
            self.p1 = p1

        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 1 - self.p1), np.full(len(X), self.p1)])

    X = np.zeros((3, 2))
    p = HL.proba({"kind": "hurdle", "trees": {"moved": Fixed(0.4), "up": Fixed(0.75)}}, X)
    np.testing.assert_allclose(p[0], [0.4 * 0.25, 0.6, 0.4 * 0.75], atol=1e-6)


def test_a_tree_that_only_knows_activity_ties_the_activity_rule(lab):
    rng = np.random.default_rng(1)
    n, N = 260, 8
    F = synthetic_F(n=n, N=N)
    # the first four stocks really move more (35% of the days against 8%); the side is a coin
    big = np.arange(N) < 4
    moved = rng.random((n, N)) < np.where(big, 0.35, 0.08)
    mv = np.where(moved, rng.choice([-1.5, 1.5], size=(n, N)), 0.2)
    F["pp"]["cf"][:, D0 + H - 1:, :] = 100.0 + mv[:, None, :]
    cols = ["activity"]
    Xf, _, cf, *_ = HL.table(lab, F, CELL, range(25, 180), cols)
    Xh, _, ch, *_ = HL.table(lab, F, CELL, range(180, 200), cols)
    Xv, _, cv, dv, _, av = HL.table(lab, F, CELL, range(200, 260), cols)
    p = HL.proba(HL.fit(Xf, cf, Xh, ch, **SMALL), Xv)
    s = ML.ladder(cv, dv, av, p, K=2, prior=ML.prior_side(F, CELL, range(25, 200), 1.0))
    assert s["move_naive"] > 1.4 * s["move_random"]      # activity alone is strong here
    assert abs(s["excess_z_prior"]) < 3                  # and the tree adds nothing to it


def test_the_shuffle_band_is_the_ladders_score_centred_on_luck_and_a_perfect_one_clears_it():
    rng = np.random.default_rng(2)
    d = np.repeat(np.arange(120), 8)
    cls = rng.choice([0, 1, 2], size=len(d), p=[0.15, 0.7, 0.15])
    act = rng.normal(size=len(d))
    long = np.tile([0.0, 0.5, 0.5], (len(d), 1)) + rng.random((len(d), 1)) * [[0, -0.4, 0.4]]
    perfect = np.eye(3)[cls] * 0.9 + 0.05
    ps = {"long": long, "perfect": perfect}
    band = HL.shuffle_band(cls, d, act, ps, K=2, prior=2, draws=300)
    assert band["z"].shape == (300, 2) and band["names"] == ["long", "perfect"]
    for name, z in zip(band["names"], band["real"], strict=True):  # what the ladder scores
        assert z == pytest.approx(ML.ladder(cls, d, act, ps[name], K=2, prior=2)["excess_z_prior"])
    assert abs(band["z"][:, 0].mean()) < 0.2 and 0.8 < band["z"][:, 0].std() < 1.2
    np.testing.assert_array_equal(band["best"], band["z"].max(axis=1))
    assert band["real"][1] > band["best"].max()            # a perfect model is beyond luck
