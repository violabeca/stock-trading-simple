"""quality_junk (proposal 4): quality minus junk, beta-neutral — synthetic panels, no DB.

Causality (a truncated panel reproduces the full panel's weights), the ranking direction, the
gross-profit -> ROE fallback, within-sector ranking, month-end-only rebalancing with hysteresis,
beta sizing frozen at entry, the daily squeeze stop, and the borrow tiers.
"""
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies
from trading_simple.data import Panel

QJ = strategies.load("quality_junk")
S = SimpleNamespace(**QJ.signals.__globals__)            # the loaded strategy.py helpers


def _panel(gpa, vols, *, betas=None, roe=None, sectors=None, dtc=None, n=420, seed=5,
           start="2022-01-03", price=100.0, shocks=None, choppy=()):
    """Names N00.. with gross profit / assets ``gpa`` (NaN = no gross profit), ROE ``roe``,
    idiosyncratic daily volatility ``vols``, market betas ``betas`` and sector codes. A name in
    ``choppy`` alternates +vol / -vol every bar instead (volatile but trendless), and
    ``shocks`` {(bar, name index): return} overrides single returns."""
    k = len(gpa)
    names = [f"N{i:02d}" for i in range(k)]
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    betas = np.ones(k) if betas is None else np.asarray(betas, float)
    market = rng.normal(0.0003, 0.008, n)
    rets = np.outer(market, betas) + rng.normal(0.0, 1.0, (n, k)) * np.asarray(vols, float)
    for j in choppy:
        rets[:, j] = np.where(np.arange(n) % 2 == 0, 1.0, -1.0) * vols[j]
    if shocks:
        for (bar, j), r in shocks.items():
            rets[bar, j] = r
    px = pd.DataFrame(price * np.cumprod(1 + rets, axis=0), index=idx, columns=names)
    const = lambda v: pd.DataFrame(np.tile(np.asarray(v, float), (n, 1)), index=idx,  # noqa: E731
                                   columns=names)
    roe = np.full(k, 0.1) if roe is None else roe
    feats = {"fund_gp_ttm": const(np.asarray(gpa, float) * 1e9),
             "fund_assets": const(np.full(k, 1e9)),
             "fund_ni_ttm": const(np.asarray(roe, float) * 5e8),
             "fund_equity": const(np.full(k, 5e8)),
             "short_interest": const(np.full(k, 1.0) if dtc is None else dtc),
             "fund_shares": const(np.full(k, 1e8)),
             "sector": const(np.zeros(k) if sectors is None else sectors)}
    return Panel(close=px, open_=px.copy(), high=px.copy(), low=px.copy(),
                 volume=pd.DataFrame(1e6, index=idx, columns=names), bar="1d", features=feats)


def _cross(k=30, seed=5, **kw):
    """A cross-section where quality is unambiguous: profitability rises and volatility falls
    with the name's number."""
    gpa = np.linspace(0.05, 0.6, k)
    vols = np.linspace(0.035, 0.006, k)
    return _panel(gpa, vols, seed=seed, **kw)


SMALL = {"slots": 3}


def test_signals_accepts_every_config_param():
    keywords = set(inspect.signature(S.signals).parameters)
    assert set(QJ.params) <= keywords
    assert QJ.valid(QJ.params)


def test_truncated_panel_reproduces_the_full_panels_weights():
    p = _cross(seed=11)
    full = S.signals(p, **{**QJ.params, **SMALL})
    for cut in ("2022-12-15", "2023-01-31", "2023-03-01"):
        ts = pd.Timestamp(cut, tz="UTC")
        sl = lambda f, ts=ts: f.loc[f.index <= ts]  # noqa: E731
        q = Panel(close=sl(p.close), open_=sl(p.open_), high=sl(p.high), low=sl(p.low),
                  volume=sl(p.volume), bar="1d",
                  features={k: sl(v) for k, v in p.features.items()})
        part = S.signals(q, **{**QJ.params, **SMALL})
        pd.testing.assert_frame_equal(part["weights"], full["weights"].loc[q.close.index])
        pd.testing.assert_frame_equal(part["borrow"], full["borrow"].loc[q.close.index])


def test_long_quality_short_junk_and_beta_sizing_tilts_dollars_long():
    p = _cross(betas=np.linspace(1.8, 0.5, 30))           # junk is also high beta
    w = S.signals(p, **{**QJ.params, **SMALL})["weights"].iloc[-1]
    longs, shorts = w[w > 0].index, w[w < 0].index
    assert len(longs) == 3 and len(shorts) == 3
    assert min(int(c[1:]) for c in longs) >= 24            # the most profitable, calmest names
    assert max(int(c[1:]) for c in shorts) <= 5            # the least profitable, wildest
    assert w[longs].sum() > -w[shorts].sum()               # low-beta longs are bigger
    assert w.abs().sum() <= 0.98 + 1e-9                    # the gross guard


def test_leg_sizing_makes_the_two_sides_carry_the_same_beta():
    p = _cross(betas=np.linspace(1.8, 0.5, 30))
    params = {**QJ.params, **SMALL, "sizing": "legs"}
    w = S.signals(p, **params)["weights"]
    beta = S.compose(p)["beta"]
    decide = S.decision_bars(p.close.index)
    t = int(np.flatnonzero(decide)[-2])                    # a decision bar; filled next bar
    held, b = w.iloc[t + 1], beta.iloc[t].clip(*S.BETA_CLIP)
    longs, shorts = held[held > 0], held[held < 0]
    assert longs.sum() > 2 * -shorts.sum()                  # calm longs carry more dollars
    exposure = (longs * b[longs.index]).sum() + (shorts * b[shorts.index]).sum()
    assert abs(exposure) < 0.25 * (longs * b[longs.index]).sum()   # sides offset (within band)
    assert w.abs().sum(axis=1).max() <= 0.98 + 1e-9
    changed = w.ne(w.shift()).any(axis=1).iloc[1:]
    first_of_month = pd.Series(w.index.month != np.r_[0, w.index.month[:-1]],
                               index=w.index).iloc[1:]
    assert not (changed & ~first_of_month).any()            # re-sized only at rebalances


def test_rebalances_only_after_month_ends_and_holds_between():
    p = _cross(seed=3)
    w = S.signals(p, **{**QJ.params, **SMALL})["weights"]
    changed = w.ne(w.shift()).any(axis=1).iloc[1:]
    idx = w.index
    first_of_month = pd.Series(idx.month != np.r_[0, idx.month[:-1]], index=idx).iloc[1:]
    assert changed.any()
    assert not (changed & ~first_of_month).any()           # no trade inside a month
    held = w.iloc[-1][w.iloc[-1] != 0].index[0]
    run = w[held][w[held] != 0]
    assert run.nunique() == 1                              # a held name is never re-sized


def test_the_panels_last_bar_is_a_month_end_on_the_exchange_calendar():
    """The session before Good Friday 2024 (03-28) and before Memorial Day 2021 (05-28) end
    their months, and a panel cut there must see the same decision bars as the full panel."""
    for last, holiday in (("2024-03-28", "2024-03-29"), ("2021-05-28", "2021-05-31"),
                          ("2022-12-30", "2023-01-02")):
        idx = pd.bdate_range(end=pd.Timestamp(last) + pd.Timedelta(days=20), periods=60, tz="UTC")
        idx = idx[idx != pd.Timestamp(holiday, tz="UTC")]
        full = S.decision_bars(idx)
        cut = int(np.searchsorted(idx, pd.Timestamp(last, tz="UTC")))
        part = S.decision_bars(idx[: cut + 1])
        assert part[-1] and full[cut]
        np.testing.assert_array_equal(part, full[: cut + 1])
    assert not S.decision_bars(pd.bdate_range(end="2024-03-27", periods=30, tz="UTC"))[-1]
    assert not S.decision_bars(pd.bdate_range(end="2026-10-16", periods=30, tz="UTC"))[-1]
    assert S.next_session("2018-12-04") == pd.Timestamp("2018-12-06")   # a one-off closure
    assert S.next_session("2021-12-30") == pd.Timestamp("2021-12-31")   # New Year on a Saturday
    assert S.next_session("2022-06-17") == pd.Timestamp("2022-06-21")   # Juneteenth observed


def test_quarterly_rebalance_trades_only_after_quarter_ends():
    w = S.signals(_cross(seed=4), **{**QJ.params, **SMALL, "rebalance": "quarter"})["weights"]
    changed = w.ne(w.shift()).any(axis=1)
    months = {(d.month) for d in w.index[changed][1:]}
    assert months <= {1, 4, 7, 10}


def test_roe_fallback_ranks_names_without_gross_profit_and_none_is_ineligible():
    gpa = np.linspace(0.05, 0.6, 12)
    gpa[[0, 1]] = np.nan                                    # two banks: no gross profit
    roe = np.full(12, 0.1)
    roe[0], roe[1] = 0.3, np.nan                            # one profitable bank, one without data
    p = _panel(gpa, np.full(12, 0.01), roe=roe)
    parts = S.compose(p)
    last = parts["score"].iloc[-1]
    assert np.isnan(last["N01"])                            # no measure at all: never ranked
    assert parts["measure"].iloc[-1]["N00"] == 1.0          # ROE used
    assert parts["raw"]["prof"].iloc[-1]["N00"] == pytest.approx(0.3)
    assert parts["eligible"].iloc[-1].sum() == 11


def test_profitability_is_ranked_within_sector():
    gpa = np.r_[np.linspace(0.4, 0.6, 5), np.linspace(0.05, 0.15, 5)]
    p = _panel(gpa, np.full(10, 0.01), sectors=[0] * 5 + [1] * 5)
    ranks = S.compose(p)["ranks"]["prof"].iloc[-1]
    assert ranks["N04"] == pytest.approx(1.0) and ranks["N09"] == pytest.approx(1.0)
    assert ranks["N00"] == pytest.approx(0.2) and ranks["N05"] == pytest.approx(0.2)
    no_sector = _panel(gpa, np.full(10, 0.01))
    no_sector.features.pop("sector")
    flat = S.compose(no_sector)["ranks"]["prof"].iloc[-1]
    assert flat["N09"] < 0.6                                # one group: sector 1 ranks low


def test_each_measure_is_ranked_against_its_own_kind_inside_a_sector():
    gpa = np.r_[np.linspace(0.3, 0.5, 6), np.full(6, np.nan), np.full(3, np.nan)]
    roe = np.r_[np.full(6, 0.1), np.linspace(0.05, 0.10, 6), np.linspace(0.2, 0.3, 3)]
    p = _panel(gpa, np.full(15, 0.01), roe=roe, sectors=[0] * 12 + [1] * 3)
    ranks = S.compose(p)["ranks"]["prof"].iloc[-1]
    assert ranks["N05"] == pytest.approx(1.0)               # best gross-profit name of sector 0
    assert ranks["N11"] == pytest.approx(1.0)               # best ROE name of sector 0: its peers
    assert ranks["N06"] == pytest.approx(1 / 6)             # are ROE names, not gross profit
    # sector 1 has only 3 ROE names (< MIN_GROUP): ranked against all 9 ROE names instead
    assert ranks["N14"] == pytest.approx(1.0) and ranks["N12"] == pytest.approx(7 / 9)


def test_growth_sleeve_is_off_by_default_and_ranks_improving_profitability_high():
    p = _cross(k=12, seed=8)
    base = S.compose(p)
    off = S.compose(p, w_growth=0.0)
    pd.testing.assert_frame_equal(base["score"], off["score"])       # U0 untouched
    gp = p.features["fund_gp_ttm"]
    gp.iloc[300:, 0] *= 6.0                                  # N00 (the least profitable) improves
    on = S.compose(p, w_growth=1.0, growth_window=252)
    assert on["ranks"]["growth"].iloc[-1]["N00"] == pytest.approx(1.0)
    assert np.isnan(on["ranks"]["growth"].iloc[100]).all()   # no year of history yet
    assert on["score"].iloc[-1]["N00"] > base["score"].iloc[-1]["N00"]


def test_hysteresis_keeps_a_long_inside_the_band_and_drops_it_below():
    score = pd.DataFrame(np.nan, index=pd.date_range("2024-01-01", periods=90, freq="B",
                                                      tz="UTC"), columns=list("ABCDE"))
    score.loc[:, :] = [0.2, 0.4, 0.6, 0.7, 0.95]
    score.iloc[30:, 4] = 0.85                               # inside the band: kept
    score.iloc[60:, 4] = 0.75                               # below q_out: leaves at a month end
    close = pd.DataFrame(10.0, index=score.index, columns=score.columns)
    decide = S.decision_bars(score.index)
    longs, _, exits = S._select(score, close, pd.DataFrame(True, index=score.index,
                                                           columns=score.columns),
                                decide, q_in=0.9, q_out=0.8, slots=2, short_stop=1.0,
                                sides="long")
    first = int(np.flatnonzero(decide)[0])
    assert longs["E"].iloc[first] and longs["E"].iloc[59]
    out = [e for e in exits if e[1] == "E"]
    assert len(out) == 1 and out[0][4] == "band"
    assert out[0][3] == score.index[decide & (np.arange(90) >= 60)][0]


def test_squeeze_stop_buys_back_the_next_day_and_blocks_reentry():
    k = 12
    shocks = {(300, 0): 1.5}                                # the junkiest name jumps 150%
    p = _panel(np.linspace(0.05, 0.6, k), np.linspace(0.035, 0.006, k), shocks=shocks,
               choppy=(0,))
    out = S.signals(p, **{**QJ.params, "slots": 2})
    w = out["weights"]["N00"]
    assert (w.iloc[130:301] < 0).all()                     # short before the jump
    assert (w.iloc[301:] == 0).all()                       # bought back at the next open and,
    #                                                        still junk, never shorted again
    parts = S.compose(p)
    _, _, exits = S._select(parts["score"], p.close, parts["shortable"],
                            S.decision_bars(p.close.index), q_in=0.9, q_out=0.8, slots=2,
                            short_stop=1.0)
    assert any(e[1] == "N00" and e[4] == "short_stop" for e in exits)


def test_borrow_tiers_and_crowded_names_are_never_shorted():
    k = 10
    dtc = np.full(k, 1.0)
    dtc[0] = 40.0                                           # 40 d x 1e6 / 1e8 = 40% short: crowded
    dtc[1] = 15.0                                           # 15%: mid tier
    p = _panel(np.linspace(0.05, 0.6, k), np.linspace(0.035, 0.006, k), dtc=dtc)
    wide = {**QJ.params, "slots": 2, "q_in": 0.8, "q_out": 0.7}   # short zone: the bottom two
    out = S.signals(p, **wide)
    b = out["borrow"].iloc[-1]
    assert b["N00"] == 0.1 and b["N01"] == 0.03 and b["N05"] == 0.005
    assert (out["weights"]["N00"] == 0).all()              # the junkiest, but too crowded
    assert (out["weights"]["N01"] < 0).any()
    p.features["short_interest"].iloc[:, :] = np.nan        # no data: general collateral
    assert (S.signals(p, **wide)["borrow"] == 0.005).all().all()


def test_valid_rejects_what_the_book_cannot_express():
    assert QJ.valid({"q_in": 0.9, "q_out": 0.8})
    assert QJ.valid({"q_in": 0.85, "q_out": 0.8})
    assert not QJ.valid({"q_in": 0.8, "q_out": 0.85})
    assert not QJ.valid({"q_in": 0.8500000000000001, "q_out": 0.85})   # float noise: no band
    assert not QJ.valid({"w_prof": 0.0, "w_safe": 0.0})
    assert not QJ.valid({"rebalance": "week"})
    assert not QJ.valid({"safety": "size"})
