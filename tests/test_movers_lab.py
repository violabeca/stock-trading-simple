"""movers_lab: the outcome in percent and its classes, a trailing activity that never sees the
day itself, the K calls per day and the ladder that scores them (every call one vote, against
random and activity picks, the side against the prior side), and a tiny TSMixer that learns a
planted signal through the down / flat / up head. Synthetic data only (no DB); the Darts test is
skipped without the optional ``darts`` extra.
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


ML = _load("movers_lab")
D0, H = 240, 120                      # decision at 13:30, horizon to 15:30
CELL = (1, D0 - 60, 60, H)


def fake_prices(n=12, N=2, move=None):
    """Every minute opens at 100 (the fill), the horizon's last trade (``cf``) is 100 + move."""
    cf = np.full((n, 390, N), 100.0)
    cf[:, D0 + H - 1:, :] = 100.0 + (np.zeros((n, N)) if move is None else move)[:, None, :]
    return {"pp": {"cf": cf, "usable": np.ones(n, bool)}, "open": np.full((n, 390, N), 100.0)}


def test_moves_are_percent_from_the_first_trade_after_the_decision():
    mv = np.array([[2.0, -1.5], [0.2, 0.0]])
    F = fake_prices(2, 2, mv)
    np.testing.assert_allclose(ML.moves(F, CELL), mv)
    F["pp"]["cf"][:, D0 - 1, :] = 90.0     # the decision minute's last trade is not the entry
    np.testing.assert_allclose(ML.moves(F, CELL), mv)
    F["open"][0, D0, 0] = np.nan           # no trade in the decision minute: the next one fills
    F["open"][0, D0 + 1, 0] = 101.0
    assert ML.moves(F, CELL)[0, 0] == pytest.approx((102.0 / 101.0 - 1) * 100)
    F["open"][1, D0:D0 + H, 1] = np.nan    # no trade in the whole horizon: no move
    assert np.isnan(ML.moves(F, CELL)[1, 1]) and np.isfinite(ML.moves(F, CELL)[1, 0])
    F["pp"]["usable"][1] = False           # a half-day has no such window
    assert np.isnan(ML.moves(F, CELL)[1]).all()


def test_the_opens_follow_the_stock_order_and_drop_the_placeholder_minutes():
    n = 2
    vol = np.ones((n, 390, 2))
    vol[0, D0, 1] = 0                      # IBKR's volume-0 placeholder: nobody traded
    F = {"pp": {"volume": vol}, "names": ["A", "B"]}
    minute = np.arange(n * 390, dtype=float)
    ML.with_opens(F, pd.DataFrame({"B": 1000 + minute, "A": minute}))  # columns in another order
    assert F["open"].shape == (n, 390, 2) and F["open"][1, 5, 0] == 390 + 5
    assert F["open"][1, 5, 1] == 1000 + 390 + 5 and np.isnan(F["open"][0, D0, 1])


def test_labels_use_theta_both_ways_and_flag_unknown_moves():
    mv = np.array([-2.0, -1.0, -0.99, 0.0, 0.99, 1.0, 2.0, np.nan])
    np.testing.assert_array_equal(ML.labels(mv, 1.0), [0, 0, 1, 1, 1, 2, 2, -1])


def test_activity_never_includes_the_session_itself():
    rng = np.random.default_rng(0)
    mv = rng.normal(0, 1, (60, 3))
    F = fake_prices(60, 3, mv)
    a0 = ML.activity(F, CELL, 1.0)
    mv2 = mv.copy()
    mv2[40] *= 50  # an enormous move on session 40
    a1 = ML.activity(fake_prices(60, 3, mv2), CELL, 1.0)
    for key in ("mean", "big", "log_mean"):
        np.testing.assert_allclose(a0[key][:41], a1[key][:41])  # nothing up to and incl. 40
        assert not np.allclose(a0[key][41], a1[key][41])         # the next session sees it
    assert np.isnan(a0["mean"][:5]).all() and np.isfinite(a0["mean"][30]).all()


def test_picks_take_the_k_best_of_every_day_and_all_of_a_short_day():
    d = np.array([0, 0, 0, 0, 1, 1, 2])
    score = np.array([0.1, 0.9, 0.5, 0.7, 0.2, 0.4, 0.3])
    got = ML.picks(score, d, 2)
    assert sorted(got.tolist()) == [1, 3, 4, 5, 6]  # day 0: 0.9, 0.7; day 1 and 2: all


def test_equal_scores_fall_back_on_the_tie_not_on_the_stocks_order():
    d = np.zeros(5, int)
    score = np.array([0.3, 0.3, 0.3, 0.3, 0.1])
    tie = np.array([1.0, 4.0, np.nan, 3.0, 9.0])
    assert sorted(ML.picks(score, d, 2, tie=tie).tolist()) == [1, 3]
    assert sorted(ML.picks(score, d, 2).tolist()) == [0, 1]           # without: the order
    p = np.column_stack([score / 2, 1 - score, score / 2 + 1e-9])      # float noise in p_move
    mp, side = ML.calls(p, d, tie, 2)
    assert sorted(mp.tolist()) == [1, 3] and (side == 2).all()


def planted(n_days=200, N=8, K=2, seed=0):
    """Each day ``N`` stocks; the first ``K`` sit at the top of the activity ranking."""
    rng = np.random.default_rng(seed)
    d = np.repeat(np.arange(n_days), N)
    act = np.tile(np.arange(N)[::-1].astype(float), n_days) + rng.normal(0, 0.1, len(d))
    return rng, d, act


def test_a_perfect_model_gets_every_call_right_and_a_blind_one_matches_luck():
    rng, d, act = planted()
    cls = rng.choice([0, 1, 2], size=len(d), p=[0.15, 0.7, 0.15])
    perfect = np.eye(3)[cls] * 0.9 + 0.05
    s = ML.ladder(cls, d, act, perfect, K=2, prior=2)
    assert s["call_prec"] > 0.99 * s["move_rate"] and s["dir_acc"] == 1.0 and s["side_z"] > 5
    assert s["excess_z_random"] > 5 and s["excess_z_prior"] > 5
    assert s["move_rate"] > s["move_random"]
    blind = rng.dirichlet([1, 1, 1], size=len(d))
    b = ML.ladder(cls, d, act, blind, K=2, prior=2)
    assert abs(b["excess_z_random"]) < 3 and abs(b["move_z_random"]) < 3  # no better than luck
    assert 0.35 < b["dir_acc"] < 0.65 and abs(b["side_z"]) < 3


def test_in_a_rising_market_always_long_beats_a_coin_but_not_the_prior_side():
    rng, d, act = planted(n_days=300)
    cls = rng.choice([0, 1, 2], size=len(d), p=[0.10, 0.70, 0.20])  # twice as many up moves
    long = np.tile([0.0, 0.6, 0.4], (len(d), 1))
    long[:, [1, 2]] += (act / 100)[:, None] * [-1, 1]                # picks: the activity ones
    s = ML.ladder(cls, d, act, long, K=2, prior=2)
    assert s["call_prec"] > s["call_naive"] + 0.03                   # "skill" against a coin
    assert s["excess_z_prior"] == 0 and s["side_z"] == 0             # none against the prior side
    assert s["call_prior"] == pytest.approx(s["call_prec"]) and s["dir_acc"] == s["dir_prior"]


def test_the_prior_side_is_the_side_the_learning_days_leaned_to():
    up = np.r_[np.full(30, 1.5), np.full(20, -1.5), np.zeros(10)]   # 30 up, 20 down, 10 flat
    F = fake_prices(len(up), 2, np.column_stack([up, np.zeros(len(up))]))
    assert ML.prior_side(F, CELL, range(len(up)), 1.0) == 2          # 30 up against 20 down
    assert ML.prior_side(F, CELL, range(25, len(up)), 1.0) == 0      # 5 up against 20 down


def test_average_precision_is_read_within_days_and_knowing_busy_days_earns_nothing():
    rng = np.random.default_rng(3)
    d = np.repeat(np.arange(100), 8)
    busy = np.repeat(rng.random(100) < 0.3, 8)
    cls = np.where(rng.random(len(d)) < np.where(busy, 0.6, 0.1), 2, 1)
    days_only = np.column_stack([np.zeros(len(d)), 1 - busy * 0.5 - 0.2, busy * 0.5 + 0.2])
    s = ML.ladder(cls, d, rng.normal(size=len(d)), days_only, K=2, prior=2)
    assert s["ap"] == pytest.approx(s["ap_base"])                    # pooled AP would be ~0.6


def test_the_activity_rule_is_the_hurdle_and_a_model_that_only_knows_activity_ties_it():
    rng, d, act = planted()
    # the stocks at the top of the activity ranking really move more (40% against 10% of days)
    rank = np.tile(np.arange(8)[::-1], 200)
    cls = np.where(rng.random(len(d)) < np.where(rank >= 6, 0.4, 0.1),
                   rng.choice([0, 2], size=len(d)), 1)
    base = ML.ladder(cls, d, act, K=2, prior=2)
    assert base["move_naive"] > 2 * base["move_random"]  # activity alone is strong
    assert base["call_naive"] == pytest.approx(base["move_naive"] / 2)  # random side: half right
    p_act = np.column_stack([act < 0, np.ones(len(d)), act < 0]).astype(float)
    p_act[:, [0, 2]] = (np.clip(act, 0, 8)[:, None] / 20) * [[1, 1]]
    p_act[:, 1] = 1 - p_act[:, 0] - p_act[:, 2]
    m = ML.ladder(cls, d, act, p_act, K=2, prior=2)
    assert abs(m["excess_z_prior"]) < 3  # same picks, same (prior) side: nothing above the hurdle
    assert m["ap_naive"] > m["ap_base"]


def synthetic_F(n=120, N=4, seed=0, theta=1.0):
    """A hand-made ``lab.build`` result: the sign of the last 60 minutes' average move decides
    the class of the horizon's move (up 1.5%, down -1.5%, else 0.2%)."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, (n, 390, N)).astype(np.float32)
    s = y[:, D0 - 60:D0, :].mean(axis=1)
    mv = np.where(s > 0.12, 1.5, np.where(s < -0.12, -1.5, 0.2))
    F = fake_prices(n, N, mv)
    dates = list(pd.bdate_range("2025-03-03", periods=n).strftime("%Y-%m-%d"))
    F["pp"] |= {"volume": np.ones((n, 390, N)), "first": np.zeros((n, N), int)}
    F |= {"y": y, "u": np.ones((n, 390, N)), "gap": np.zeros((n, N)),
          "prev_move": {1: np.zeros((n, N))}, "names": [f"S{i}" for i in range(N)],
          "dates": np.array(dates),
          "bar": {"path": y, "volume": rng.normal(0, 1, (n, 390, N)).astype(np.float32)}}
    return F


def test_the_examples_carry_only_the_label_on_the_last_step_and_the_input_minutes():
    pytest.importorskip("darts")
    lab = _load("tsmixer_lab")
    F = synthetic_F(n=30)
    ser, cov, d, k = ML.examples(lab, F, CELL, range(12, 30), ["volume"], full=True)
    assert len(ser) == len(cov) == len(d) == len(k) > 0
    v = ser[0].values()
    assert v.shape == (60 + H, 3) and v[:-1].sum() == 0 and v[-1].sum() == 1  # label last
    assert cov[0].n_timesteps == 60 and list(cov[0].components) == ["path", "volume"]
    assert "is_S0" in ser[0].static_covariates.columns
    inp, _, _, _ = ML.examples(lab, F, CELL, range(12, 30), ["volume"], full=False)
    assert inp[0].n_timesteps == 60 and inp[0].values().sum() == 0  # no label in the input


def test_a_tiny_tsmixer_learns_the_planted_signal_through_the_three_class_head():
    pytest.importorskip("darts")
    pytest.importorskip("torch")
    lab = _load("tsmixer_lab")
    F = synthetic_F(n=140, N=4)
    hp = {"hidden_size": 16, "ff_size": 16, "num_blocks": 1, "dropout": 0.0, "lr": 5e-3,
          "weight_decay": 1e-5, "patience": 8, "flat_weight": 1.0}
    kw = dict(prev_days=1, stock_id=True, theta=1.0)
    model, info = ML.train(lab, F, CELL, range(12, 100), range(100, 110), ["volume"], hp,
                           epochs=40, **kw)
    p, d, k = ML.predict_proba(lab, model, F, CELL, range(110, 140), ["volume"], **kw)
    cls, act = ML.targets(F, CELL, d, k, 1.0)
    assert p.shape == (len(cls), 3) and np.allclose(p.sum(axis=1), 1, atol=1e-5)
    assert (p.argmax(axis=1) == cls).mean() > 0.6  # chance is ~0.33, the biggest class ~0.4
    assert info["val_loss"] < info["prior_loss"]    # it beat "always the class shares"
    s = ML.ladder(cls, d, act, p, K=2, prior=ML.prior_side(F, CELL, range(12, 110), 1.0))
    assert s["dir_acc"] > 0.8 and s["dir_acc"] > s["dir_prior"] + 0.2
