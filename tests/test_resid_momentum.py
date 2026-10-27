"""resid_momentum (proposal 2): causal month-end residual-momentum book, beta removed from the
ranking, banded slots, a quantized book-volatility scale and borrow tiers — synthetic panels,
no DB. Engine runs go through the proposals lab's runner (``lab.run_portfolio``, v2), which
produces every lab number, and the shared engine where the difference is the point."""
import importlib.util
import inspect
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_simple import backtest as bt
from trading_simple import strategies
from trading_simple.data import Panel

RM = strategies.load("resid_momentum")
MOD = types.SimpleNamespace(**RM.signals.__globals__)   # the loaded strategy.py
_SPEC = importlib.util.spec_from_file_location(
    "proposals_lab_rm", Path(__file__).resolve().parents[1] / "strategies/proposals_lab/lab.py")
LAB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(LAB)
SMALL = {"lookback": 120, "skip": 10, "beta_window": 120, "min_history": 120, "slots": 3,
         "q_in": 0.8, "q_out": 0.7, "vol_window": 40}
GROSS = RM.params["gross"]


def _panel(n=700, m=30, seed=5, market_drift=0.0004, betas=None, drift=None, noise=0.01,
           dtc=None, sectors=None):
    """``m`` names driven by one market factor: ``r = beta × market + drift + noise`` (plus a
    1%-a-day factor per sector, and the ``sector`` feature, when ``sectors`` codes are given)."""
    idx = pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    market = rng.normal(market_drift, 0.01, n)
    betas = np.linspace(0.5, 1.5, m) if betas is None else np.asarray(betas, float)
    drift = (np.linspace(-0.001, 0.001, m)[rng.permutation(m)] if drift is None
             else np.asarray(drift, float))
    rets = np.outer(market, betas) + drift + rng.normal(0.0, noise, (n, len(betas)))
    if sectors is not None:
        codes = np.asarray(sectors, int)
        rets += rng.normal(0.0, 0.01, (n, codes.max() + 1))[:, codes]
    names = [f"S{i:02d}" for i in range(len(betas))]
    close = pd.DataFrame(50.0 * np.cumprod(1 + rets, axis=0), index=idx, columns=names)
    days = np.full(close.shape, 1.0) if dtc is None else np.tile(dtc, (n, 1))
    feats = {"short_interest": pd.DataFrame(days, index=idx, columns=names),
             "fund_shares": pd.DataFrame(1e8, index=idx, columns=names)}
    if sectors is not None:
        feats["sector"] = pd.DataFrame(np.tile(np.asarray(sectors, float), (n, 1)), index=idx,
                                       columns=names)
    return Panel(close=close, open_=close * 0.999, high=close * 1.01, low=close * 0.99,
                 volume=pd.DataFrame(1e6, index=idx, columns=names), bar="1d", features=feats)


def _cut(p, end):
    keep = p.close.index <= end
    return Panel(close=p.close[keep], open_=p.open_[keep], high=p.high[keep], low=p.low[keep],
                 volume=p.volume[keep], bar="1d",
                 features={k: v[keep] for k, v in p.features.items()})


def _run(p, **over):
    return RM.signals(p, **{**RM.params, **SMALL, **over})


@pytest.mark.parametrize("where", ["mid-month", "month-end"])
def test_signals_on_a_truncated_panel_reproduce_the_full_panels_weights(where):
    p = _panel()
    full = _run(p)
    idx = p.close.index
    if where == "mid-month":
        end = idx[455]
    else:                                   # the last bar of a month: a decision row in full
        month = idx.year * 12 + idx.month
        end = idx[np.flatnonzero(month[1:] != month[:-1])[20]]
    part = _run(_cut(p, end))
    assert (part["weights"].abs().sum(axis=1) > 0).any()        # the test exercises a book
    pd.testing.assert_frame_equal(full["weights"].loc[:end], part["weights"])
    pd.testing.assert_frame_equal(full["borrow"].loc[:end], part["borrow"])


def test_the_book_trades_only_on_the_two_bars_after_a_month_end_closing_first():
    p = _panel()
    reb = MOD.compose(p, **{k: v for k, v in {**RM.params, **SMALL}.items()
                            if k in inspect.signature(MOD.compose).parameters})["rebalance"]
    first, second = reb.shift(1, fill_value=False), reb.shift(2, fill_value=False)
    for phased, allowed in ((False, first), (True, first | second)):
        w = _run(p, vol_target=0.05, phased=phased)["weights"]
        changed = w.ne(w.shift()).any(axis=1)
        changed.iloc[0] = False
        assert changed.any()
        assert not (changed & ~allowed).any()                    # never mid-month
    old = w.shift(fill_value=0.0)                                # w: the phased book
    opens_or_grows = ((old == 0) & (w != 0)) | ((np.sign(old) == np.sign(w))
                                                 & (w.abs() > old.abs()))
    closes_or_shrinks = (old != 0) & ((np.sign(old) != np.sign(w)) | (w.abs() < old.abs()))
    assert opens_or_grows[second].any().any() and closes_or_shrinks[first].any().any()
    assert not opens_or_grows[first].any().any()                 # first bar: release cash
    assert not closes_or_shrinks[second].any().any()             # second bar: spend it


def test_the_residual_score_removes_the_market_bet_that_raw_momentum_makes():
    """In a strongly rising market the raw ranking is a beta bet; the residual one is not."""
    betas = list(np.linspace(0.2, 2.0, 18)) + [1.0, 1.0]
    drift = [0.0] * 18 + [0.002, -0.002]                    # S18 trends up, S19 down on its own
    p = _panel(n=400, betas=betas, drift=drift, market_drift=0.003, noise=0.004, seed=11)
    kw = {"lookback": 120, "skip": 10, "beta_window": 120, "min_history": 120}
    raw = MOD.compose(p, score="raw", **kw)["value"].iloc[-1]
    res = MOD.compose(p, score="residual", **kw)["value"].iloc[-1]
    pure = raw.index[:18]                                      # no trend of their own
    beta_rank = pd.Series(betas[:18], index=pure).rank()
    assert raw[pure].rank().corr(beta_rank) > 0.9              # raw: a ranking of beta
    assert raw["S17"] > raw["S18"]                             # ... above a real own trend
    assert abs(res[pure].rank().corr(beta_rank)) < 0.6         # residual: beta-blind
    assert res.idxmax() == "S18" and res.idxmin() == "S19"     # the names' own trends
    assert abs(res["S17"]) < 0.5 * res["S18"]                  # beta is no longer rewarded


def test_the_sector_regressor_removes_the_sectors_own_trend():
    """Two sectors trending apart: market-only residuals rank the whole up-sector on top; with
    the sector regressor only each name's OWN trend is left."""
    m = 20
    drift = np.r_[np.full(10, 0.002), np.full(10, -0.002)]       # sector 0 up, sector 1 down
    drift[0] += 0.002                                             # S00: its own trend on top
    drift[10] -= 0.002                                            # S10: its own fall
    p = _panel(n=400, m=m, betas=np.ones(m), drift=drift, noise=0.004, seed=8,
               sectors=np.r_[np.zeros(10), np.ones(10)])
    kw = {"lookback": 120, "skip": 10, "beta_window": 120, "min_history": 120}
    one = MOD.compose(p, **kw)["value"].iloc[-1].rank()
    two = MOD.compose(p, factors="market+sector", **kw)["value"].iloc[-1].rank()
    assert set(one.nlargest(10).index) == {f"S{i:02d}" for i in range(10)}   # a sector bet
    assert two.idxmax() == "S00" and two.idxmin() == "S10"
    rest_up, rest_down = two[[f"S{i:02d}" for i in range(1, 10)]], two[[f"S{i:02d}" for i in
                                                                          range(11, 20)]]
    # full sector separation would put the two groups' mean ranks 9 apart (market-only does);
    # what is left is the planted names' own trends inside their sectors' leave-one-out means
    assert abs(rest_up.mean() - rest_down.mean()) < 6
    w = RM.signals(p, **{**RM.params, **SMALL, "factors": "market+sector"})["weights"]
    assert (w["S00"] > 0).any() and (w["S10"] < 0).any()


def test_beta_is_the_trailing_ols_slope_known_at_the_bar():
    p = _panel(n=300, m=5, seed=2)
    parts = MOD.compose(p, beta_window=100, lookback=60, skip=5, min_history=60)
    r, m = parts["returns"], parts["market"]
    t = 250
    x, y = m.iloc[t - 99:t + 1], r["S03"].iloc[t - 99:t + 1]
    assert parts["beta"]["S03"].iloc[t] == pytest.approx(np.cov(y, x)[0, 1] / x.var(), rel=1e-9)


def test_slots_are_dollar_neutral_and_the_book_scale_moves_in_quarter_steps():
    p = _panel()
    slot = GROSS * 0.5 / 3
    last = _run(p, vol_target=0.0)["weights"].iloc[-1]
    assert last[last > 0].sum() == pytest.approx(GROSS / 2) and (last > 0).sum() == 3
    assert last[last < 0].sum() == pytest.approx(-GROSS / 2) and (last < 0).sum() == 3
    tiny = _run(p, vol_target=1e-4)["weights"].iloc[-1]
    assert np.abs(tiny[tiny != 0]).to_numpy() == pytest.approx(np.full(6, 0.25 * slot))
    huge = _run(p, vol_target=10.0)["weights"].iloc[-1]
    assert np.abs(huge[huge != 0]).to_numpy() == pytest.approx(np.full(6, slot))
    w = _run(p, vol_target=0.05)["weights"]
    steps = np.unique(np.round(np.abs(w.to_numpy()[w.to_numpy() != 0]) / slot, 9))
    assert set(steps) <= {0.25, 0.5, 0.75, 1.0} and len(steps) >= 2


def test_bands_hold_a_name_until_it_leaves_the_exit_zone_and_slots_take_the_best_first():
    idx = pd.date_range("2024-01-31", periods=4, freq="ME", tz="UTC")
    rank = pd.DataFrame({"A": [1.0, 0.75, 0.75, 0.65], "B": [0.05, 0.25, 0.25, 0.35],
                         "C": [0.95, 0.95, 0.95, 0.95], "D": [0.5, 0.5, 0.5, 0.5]}, index=idx)
    reb = pd.Series(True, index=idx)
    priced = pd.DataFrame(True, index=idx, columns=rank.columns)
    shortable = priced.copy()
    longs, shorts = MOD._select(rank, shortable, reb, priced, q_in=0.9, q_out=0.7, slots=1)
    assert longs["A"].tolist() == [True, True, True, False]     # 0.75 holds, 0.65 leaves
    assert longs["C"].tolist() == [False, False, False, True]   # waits for the slot, then fills
    assert shorts["B"].tolist() == [True, True, True, False]    # mirror: 0.25 holds, 0.35 out
    stuck = priced.copy()
    stuck.loc[idx[3], "A"] = False                              # no price: cannot be sold
    longs, _ = MOD._select(rank, shortable, reb, stuck, q_in=0.9, q_out=0.7, slots=1)
    assert longs["A"].iloc[3] and not longs["C"].iloc[3]


def test_borrow_tiers_follow_short_interest_and_the_crowded_name_is_never_shorted():
    m = 20
    drift = np.linspace(0.001, -0.001, m)              # S19 has the worst momentum
    dtc = np.full(m, 5.0)                              # 5 days x 1e6 / 1e8 shares = 5% short
    dtc[[16, 17, 18, 19]] = [15.0, 25.0, 15.0, 35.0]   # 15%, 25%, 15%, 35%
    p = _panel(m=m, drift=drift, betas=np.ones(m), dtc=dtc, noise=0.004)
    out = _run(p, vol_target=0.0)
    b = out["borrow"].iloc[-1]
    assert b["S00"] == 0.005 and b["S16"] == 0.03 and b["S17"] == 0.1
    assert (out["weights"]["S19"] >= 0).all()          # above short_max_si: never shorted
    assert (out["weights"][["S17", "S18"]] < 0).any().all()


def test_a_holiday_stub_never_carries_a_decision_or_an_order():
    p = _panel()
    idx = p.close.index
    month = idx.year * 12 + idx.month
    stub = np.flatnonzero(month[1:] != month[:-1])[15]       # make a month's last bar a stub
    close = p.close.copy()
    close.iloc[stub, 1:] = np.nan                            # one stray name prints
    q = Panel(close=close, open_=close * 0.999, high=close * 1.01, low=close * 0.99,
              volume=p.volume, bar="1d", features=p.features)
    rows = MOD._full_rows(close)
    assert not rows.iloc[stub] and rows.drop(idx[stub]).all()
    reb = MOD._month_ends(idx, rows)
    assert reb.iloc[stub - 1] and not reb.iloc[stub]         # decided on the last full row
    w = _run(q)["weights"]
    assert (w.iloc[stub] == w.iloc[stub - 1]).all()          # the stub carries: no order
    decided = pd.DataFrame(0.0, index=idx, columns=close.columns)
    decided.iloc[stub - 1:, 0] = 0.1                         # a decision on the last full row
    done = MOD._execute(decided, rows, phased=False)["S00"]
    assert done.iloc[stub] == 0.0 and done.iloc[stub + 1] == 0.1   # ... fills after the stub
    done = MOD._execute(decided, rows, phased=True)["S00"]   # phased: an open waits one row
    assert done.iloc[stub + 1] == 0.0 and done.iloc[stub + 2] == 0.1


def _run_book(p, weights, runner="lab", **extra):
    """``weights`` through the lab's runner (v2) or the shared engine, at open fills."""
    bt.configure()
    run = LAB.run_portfolio if runner == "lab" else bt._run_portfolio
    return run(p, weights, price=p.open_, fees=0.0011, slippage=0.0001, init_cash=10_000.0,
               **extra)


def _flat_book(decided, close=None):
    """Hand-made decided weights run three ways: the shared engine unphased and phased, and the
    lab's runner unphased (constant prices unless ``close`` is given; the open equals the close,
    so a filled target is held exactly at its fill bar's close)."""
    idx = decided.index
    close = pd.DataFrame(100.0, index=idx, columns=decided.columns) if close is None else close
    p = Panel(close=close, open_=close, high=close, low=close,
              volume=pd.DataFrame(1e6, index=idx, columns=decided.columns), bar="1d")
    rows = pd.Series(True, index=idx)
    out = {}
    for runner, phased in (("engine", False), ("engine", True), ("lab", False)):
        w = MOD._execute(decided, rows, phased)
        pf = _run_book(p, w, runner)
        out[runner, phased] = (w, pf.asset_value(group_by=False).div(pf.value(), axis=0))
    return out


def test_a_new_short_needs_its_cover_first_phasing_or_the_lab_runner():
    """A short rotating into another short at gross 0.98. Sent on one bar through the shared
    engine before engine proposal #46 (2026-10-24), the new short came first (``call_seq="auto"``
    sells before it buys), found ~2% free cash because a short's proceeds stay locked until it
    is covered, was filled for that 2% and never retried. Phased, the cover trades a bar
    earlier; the lab's runner (v2) and now the shared engine place the cover first on the same
    bar."""
    idx = pd.date_range("2024-01-01", periods=60, freq="B", tz="UTC")
    decided = pd.DataFrame(0.0, index=idx, columns=["L1", "L2", "S1", "S2", "S3"])
    decided.iloc[10:, :2] = 0.245                            # 2 longs + 2 shorts = 0.98 gross
    decided.iloc[10:, 2:4] = -0.245
    decided.iloc[30:, 3], decided.iloc[30:, 4] = 0.0, -0.245  # decided on row 30: S2 -> S3
    books = _flat_book(decided)
    for w, real in books.values():
        assert w.iloc[-1].tolist() == pytest.approx([0.245, 0.245, -0.245, 0.0, -0.245])
        assert real.iloc[25].to_numpy() == pytest.approx(w.iloc[25].to_numpy(), abs=0.005)
    for key in (("engine", False), ("engine", True), ("lab", False)):
        w, real = books[key]
        assert real.iloc[40].to_numpy() == pytest.approx(w.iloc[40].to_numpy(), abs=0.005), key
    w = books["engine", True][0]
    assert w["S2"].iloc[31] == 0.0 and w["S3"].iloc[31] == 0.0 and w["S3"].iloc[32] == -0.245
    w = books["lab", False][0]
    assert w["S2"].iloc[31] == 0.0 and w["S3"].iloc[31] == -0.245   # one bar, cover first


def test_the_lab_runner_holds_every_target_inside_the_gross_limit():
    p = _panel()
    sig = _run(p)
    w = sig["weights"]
    pf = _run_book(p, w, retrim_k=sig["retrim_k"], borrow=sig["borrow"])
    value = pf.value()
    assert np.isfinite(value).all() and value.iloc[-1] > 0
    assert w.abs().sum(axis=1).max() <= GROSS + 1e-12        # targets: within gross
    held = pf.asset_value(group_by=False)
    assert ((held > 0).sum(axis=1) <= 3).all() and ((held < 0).sum(axis=1) <= 3).all()
    # every target is held, on the side asked for: no entry is rejected
    assert (np.sign(pf.assets()) == np.sign(w)).all().all()
    # covers run before new shorts, so every short is placed in full. A long, the bar's last
    # order, can still be cut when held shorts carry gains (the next test; this 3-slot book's
    # 15% slots make that likelier than the real 20-slot book, which the audit found uncut)
    placed = bt.rotation_targets(w).notna() & (w < 0)
    placed.iloc[0] = False
    gap = (held.div(value, axis=0) - w).to_numpy()[placed.to_numpy()]
    assert len(gap) > 10 and np.abs(gap).max() < 0.005


def test_valid_rejects_what_the_book_cannot_express():
    assert RM.valid(RM.params)
    for bad in ({"q_out": 0.9}, {"skip": 252}, {"lookback": 30, "skip": 21},
                {"score": "total"}, {"retrim_k": 1.0}, {"gross": 1.2}, {"si_mid": 0.3},
                {"vol_step": 0.0}, {"phased": "yes"}):
        assert not RM.valid(bad), bad


def test_signals_accepts_every_config_param():
    keywords = set(inspect.signature(RM.signals).parameters)
    assert set(RM.params) <= keywords
    assert set(RM.optuna["space"]) <= set(RM.params)


def test_beta_sizing_balances_the_sides_market_risk_and_freezes_each_slot():
    m = 30
    betas = np.linspace(0.5, 1.8, m)
    drift = np.linspace(0.0015, -0.0015, m)            # winners low-beta, losers high-beta
    p = _panel(m=m, betas=betas, drift=drift, noise=0.006, seed=4)
    beta = MOD.compose(p, **{k: v for k, v in {**RM.params, **SMALL}.items()
                             if k in inspect.signature(MOD.compose).parameters})["beta"]
    net = {}
    for sizing in ("equal", "beta"):
        w = _run(p, sizing=sizing, vol_target=0.0)["weights"]
        net[sizing] = (w * beta.shift(1)).sum(axis=1).iloc[-200:].abs().mean()
        assert w.abs().sum(axis=1).max() <= GROSS + 1e-12           # the gross guard
    assert net["beta"] < 0.5 * net["equal"]
    w = _run(p, sizing="beta", vol_target=0.0)["weights"]
    for col in w.columns:                                          # frozen while held
        held = w[col][w[col] != 0]
        runs = (np.sign(w[col]) != np.sign(w[col].shift())).cumsum()[held.index]
        assert (held.groupby(runs).nunique() == 1).all()


def test_winning_shorts_lock_their_gains_so_the_gross_keeps_a_margin():
    """vectorbt's free cash leaves a winning short's gain out twice (its proceeds and its
    collateral stay locked at the entry price) while targets are shares of the marked book. With
    the shorts up ~25%, a long rotation through the lab's runner at gross 0.98 gets only part of
    its slot and the rest is never sent again; at the config's 0.90 it gets all of it. No call
    order can fix this one (the sale of L2 already runs first): only gross headroom does."""
    idx = pd.date_range("2024-01-01", periods=60, freq="B", tz="UTC")
    cols = ["L1", "L2", "L3", "S1", "S2"]
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    close.loc[idx[15:], ["S1", "S2"]] = np.linspace(100.0, 60.0, 45)[:, None]
    fill = {}
    for gross in (0.98, GROSS):
        slot = gross / 4
        decided = pd.DataFrame(0.0, index=idx, columns=cols)
        decided.iloc[5:, :2], decided.iloc[5:, 3:] = slot, -slot
        decided.iloc[40:, 1], decided.iloc[40:, 2] = 0.0, slot     # decided on row 40: L2 -> L3
        w, real = _flat_book(decided, close)["lab", False]
        assert w["L2"].iloc[41] == 0.0 and w["L3"].iloc[41] == slot  # one bar, unphased
        fill[gross] = real["L3"].iloc[41] / slot
    assert fill[0.98] < 0.99 and fill[GROSS] == pytest.approx(1.0, abs=1e-3)   # fees only
