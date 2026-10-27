"""xs_si_mom_composite (proposals lab cross-strategy): one dollar-neutral book ranked on short
interest plus residual momentum, built from three component strategies loaded by path and
pinned by SHA-256. Synthetic panels, no DB."""
import hashlib
import importlib.util
import inspect
import types

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies
from trading_simple.data import Panel

XS = strategies.load("xs_si_mom_composite")
MOD = types.SimpleNamespace(**XS.signals.__globals__)          # the loaded strategy.py
P1 = strategies.load("informed_flow_ls_p1")
RM = MOD.component("rm")
SI = {"long_sleeves": "si", "short_sleeves": "si"}
MOM = {"long_sleeves": "mom", "short_sleeves": "mom"}


def _panel(n=620, m=150, seed=4, gaps=False):
    """``m`` names on one market factor with persistent own drifts (momentum to rank),
    days-to-cover random walks (short interest to rank), dollar volumes spread over two
    orders of magnitude, a few names too crowded to short, and profitability fundamentals in
    four sectors. ``gaps``: some days-to-cover missing (ineligible) and one name listed late."""
    idx = pd.date_range("2016-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, 0.01, n)
    betas = rng.uniform(0.5, 1.6, m)
    regime = np.repeat(rng.normal(0.0, 0.0015, (n // 120 + 1, m)), 120, axis=0)[:n]
    rets = np.outer(market, betas) + regime + rng.normal(0.0, 0.015, (n, m))
    names = [f"N{i:03d}" for i in range(m)]
    close = pd.DataFrame(60.0 * np.cumprod(1 + rets, axis=0), index=idx, columns=names)
    volume = pd.DataFrame(np.exp(rng.normal(13.0, 0.3, (n, m)) + np.linspace(0, 4.6, m)),
                          index=idx, columns=names)
    dtc = np.exp(np.log(rng.uniform(1.0, 6.0, m)) + np.cumsum(rng.normal(0, 0.04, (n, m)),
                                                               axis=0))
    shares = np.full(m, 3e9)
    shares[:4] = 1e5                                     # >30% of shares short: never shorted
    sector = np.arange(m) % 4
    gp = rng.uniform(0.05, 0.6, m) * 1e9
    gp[::7] = np.nan                                     # no gross profit: ROE stands in
    feats = {"short_interest": pd.DataFrame(dtc, index=idx, columns=names),
             "fund_shares": pd.DataFrame(np.tile(shares, (n, 1)), index=idx, columns=names),
             "fund_gp_ttm": pd.DataFrame(np.tile(gp, (n, 1)), index=idx, columns=names),
             "fund_assets": pd.DataFrame(1e9, index=idx, columns=names),
             "fund_ni_ttm": pd.DataFrame(np.tile(rng.uniform(-0.1, 0.3, m) * 1e9, (n, 1)),
                                         index=idx, columns=names),
             "fund_equity": pd.DataFrame(1e9, index=idx, columns=names),
             "sector": pd.DataFrame(np.tile(sector.astype(float), (n, 1)), index=idx,
                                    columns=names)}
    open_ = close * (1 + rng.normal(0, 0.002, (n, m)))
    if gaps:
        si = feats["short_interest"]
        si.iloc[300:340, 10:20] = np.nan                 # ineligible for p1, still ranked by rm
        close.iloc[:200, 30] = np.nan                    # a name listed late
        open_.iloc[:200, 30] = np.nan
    return Panel(close=close, open_=open_, high=close * 1.01, low=close * 0.99, volume=volume,
                 bar="1d", features=feats)


def _cut(p, end):
    keep = p.close.index <= end
    return Panel(close=p.close[keep], open_=p.open_[keep], high=p.high[keep], low=p.low[keep],
                 volume=p.volume[keep], bar="1d",
                 features={k: v[keep] for k, v in p.features.items()})


def _run(p, **over):
    return XS.signals(p, **{**XS.params, **over})


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def gapped():
    return _panel(gaps=True)


def test_c_si_at_098_reproduces_informed_flow_ls_p1_exactly(panel):
    """C_si is p1 v6 itself: same ranks, bands, stops and slots. The panels carry no option
    chains, so p1's skew sleeve is silent and its config (w_skew 1) ranks ``(si + 0.5) / 2``:
    the same ordering in exact arithmetic, but SI percentiles taken inside 5 liquidity groups
    tie across groups, and float rounding splits those ties differently from ``si`` alone. So
    the exact reproduction is against the config with the silent sleeve's weight at 0, and the
    config itself differs only by those split ties (RESEARCH_LOG.md §0.4)."""
    mine = _run(panel, **SI, gross=0.98)
    ref = P1.signals(panel, **{**P1.params, "w_skew": 0.0})
    assert (mine["weights"] != 0).sum().sum() > 1000                 # a real book
    pd.testing.assert_frame_equal(mine["weights"], ref["weights"])
    pd.testing.assert_frame_equal(mine["borrow"], ref["borrow"])
    assert mine["retrim_k"] == ref["retrim_k"] == 2.0
    p1 = MOD.component("p1")
    keys = inspect.signature(p1.compose).parameters
    kw = {k: v for k, v in P1.params.items() if k in keys}
    cfg, off = (p1.compose(panel, **{**kw, "w_skew": w})["score"] for w in (1.0, 0.0))
    pd.testing.assert_frame_equal(cfg.isna(), off.isna())
    rho = np.array([cfg.iloc[t].corr(off.iloc[t], method="spearman")
                    for t in range(300, len(cfg), 25)])
    assert rho.min() > 0.999                                         # ties, not another signal


def test_the_mom_rank_at_t_is_the_last_month_ends_rank_and_nan_where_p1_is_ineligible(gapped):
    r = MOD.ranks(gapped, {"mom"})
    mom, ok = r["ranks"]["mom"], r["base"]["eligible"]
    parts = RM.compose(gapped, min_price=5.0)
    reb = parts["rebalance"].to_numpy()
    at = np.flatnonzero(reb)
    assert len(at) > 6
    last = np.searchsorted(at, np.arange(len(reb)), side="right") - 1   # last month end <= t
    for t in range(len(reb)):
        row = mom.iloc[t]
        if last[t] < 0:
            assert row.isna().all()
            continue
        want = parts["rank"].iloc[at[last[t]]].where(ok.iloc[t])
        pd.testing.assert_series_equal(row, want, check_names=False)
    # never a later month end: the bar before a month end still reads the previous one
    live = [a for a in at if parts["rank"].iloc[a].notna().sum() > 50]
    assert len(live) > 6
    t = live[3]
    assert mom.iloc[t - 1].notna().sum() > 50 and not mom.iloc[t - 1].equals(mom.iloc[t])
    # the gaps: rm ranks names p1 calls ineligible, and the composite never sees them
    blank = (~ok) & parts["rank"].loc[reb].reindex(gapped.close.index, method="ffill").notna()
    assert blank.to_numpy().sum() > 100 and mom[blank].isna().all().all()


@pytest.mark.parametrize("where", ["mid-month", "month-end"])
def test_a_truncated_panel_reproduces_the_weights_through_its_last_bar(gapped, where):
    idx = gapped.close.index
    month = idx.year * 12 + idx.month
    ends = np.flatnonzero(month[1:] != month[:-1])
    end = idx[ends[20] - 7] if where == "mid-month" else idx[ends[20]]   # month-end: a MOM
    for over in ({}, {"long_sleeves": "si", "short_sleeves": "mom"},     # decision in full
                 {"short_gate": "prof"}):
        full, part = _run(gapped, **over), _run(_cut(gapped, end), **over)
        assert (part["weights"].iloc[-60:] != 0).any().any()
        pd.testing.assert_frame_equal(full["weights"].loc[:end], part["weights"])
        pd.testing.assert_frame_equal(full["borrow"].loc[:end], part["borrow"])


@pytest.mark.parametrize("over", [{}, {"long_sleeves": "si", "short_sleeves": "mom"},
                                  {"short_gate": "prof"}])
def test_book_invariants_sides_gross_and_one_weight_per_hold(gapped, over):
    p = {**XS.params, **over}
    w = _run(gapped, **over)["weights"]
    assert (w > 0).sum(axis=1).max() == p["slots"] and (w < 0).sum(axis=1).max() == p["slots"]
    assert w.abs().sum(axis=1).max() <= p["gross"] + 1e-9
    slot = p["gross"] * 0.5 / p["slots"]
    assert np.allclose(w.abs().to_numpy()[w.to_numpy() != 0], slot)    # equal, constant slots
    side = np.sign(w)
    changed = side.ne(side.shift(fill_value=0.0))
    hold = changed.cumsum()                                          # one id per hold and name
    for name in w.columns[(w != 0).any()]:
        held = side[name] != 0
        assert w[name][held].groupby(hold[name][held]).nunique().max() == 1
    # no name long and short on one bar: the held frames never overlap
    r = MOD.ranks(gapped, {"si", "mom"})
    ls = MOD.side_score(r["ranks"], ["si", "mom"], {"si": 1.0, "mom": 1.0},
                        r["base"]["eligible"], 5)
    longs, shorts = MOD.select(ls, ls, gapped.close, r["base"]["shortable"], two_pass=False,
                               q_in=0.9, q_out=0.8, slots=15, max_hold=126, short_stop=1.0)
    assert not (longs & shorts).to_numpy().any()


def test_combine_book_is_half_c_si_plus_half_c_mom_netted(panel):
    mix = _run(panel, combine="book")
    a, b = _run(panel, **SI), _run(panel, **MOM)
    pd.testing.assert_frame_equal(mix["weights"], 0.5 * a["weights"] + 0.5 * b["weights"])
    pd.testing.assert_frame_equal(mix["borrow"], a["borrow"])
    assert ((a["weights"] > 0) & (b["weights"] < 0)).to_numpy().any()   # conflicts net out
    assert (mix["weights"].abs().sum(axis=1) <= 0.90 + 1e-9).all()


def test_with_identical_sleeves_two_passes_equal_one_pass(gapped):
    r = MOD.ranks(gapped, {"si", "mom"})
    score = MOD.side_score(r["ranks"], ["si", "mom"], {"si": 1.0, "mom": 1.0},
                           r["base"]["eligible"], 5)
    rules = dict(q_in=0.9, q_out=0.8, slots=15, max_hold=126, short_stop=1.0)
    one = MOD.select(score, score, gapped.close, r["base"]["shortable"], two_pass=False, **rules)
    two = MOD.select(score, score, gapped.close, r["base"]["shortable"], two_pass=True, **rules)
    assert one[0].to_numpy().sum() > 1000 and one[1].to_numpy().sum() > 1000
    pd.testing.assert_frame_equal(one[0], two[0])
    pd.testing.assert_frame_equal(one[1], two[1])


def test_the_prof_gate_only_removes_short_candidates(gapped):
    plain, gated = _run(gapped)["weights"], _run(gapped, short_gate="prof")["weights"]
    pd.testing.assert_frame_equal(plain.clip(lower=0), gated.clip(lower=0))      # longs alike
    prof = MOD.ranks(gapped, {"prof"})["ranks"]["prof"]
    junk = prof.le(MOD.GATE_MAX).shift(1, fill_value=False)       # at the deciding bar
    assert not ((gated < 0) & ~junk).to_numpy().any()             # every gated short is junk
    assert ((plain < 0) & ~junk).to_numpy().any()                 # the plain book was not
    assert not plain.clip(upper=0).equals(gated.clip(upper=0))


def test_a_target_change_on_a_bar_without_an_open_waits_for_the_next_priced_open(panel):
    w = _run(panel)["weights"]
    change = w.ne(w.shift(fill_value=0.0))
    t, j = next((t, j) for t, j in zip(*np.nonzero(change.to_numpy()), strict=True)
                if t > 400 and not change.iloc[t + 1, j] and w.iloc[t, j] != 0)
    holed = Panel(close=panel.close, open_=panel.open_.copy(), high=panel.high, low=panel.low,
                  volume=panel.volume, bar="1d", features=panel.features)
    holed.open_.iloc[t, j] = np.nan                       # the open the order would fill at
    v = _run(holed)["weights"]
    assert v.iloc[t, j] == w.iloc[t - 1, j]               # the change waits ...
    assert v.iloc[t + 1, j] == w.iloc[t + 1, j]           # ... for the next priced open
    others = [c for c in range(w.shape[1]) if c != j]
    pd.testing.assert_frame_equal(v.iloc[:, others], w.iloc[:, others])
    pd.testing.assert_frame_equal(v.drop(index=v.index[t]), w.drop(index=w.index[t]))


def test_sector_neutral_ranks_si_within_sector_and_liquidity_cells_and_mom_on_sector(panel):
    """I2: SI is a percentile inside (sector × dollar-volume group) cells, a cell under
    MIN_CELL names keeps the liquidity-group rank, and MOM's residual drops the sector."""
    sector = panel.features["sector"].copy()
    sector.iloc[:, :3] = 9.0                                   # a three-name sector
    p = Panel(close=panel.close, open_=panel.open_, high=panel.high, low=panel.low,
              volume=panel.volume, bar="1d", features={**panel.features, "sector": sector})
    neutral, plain = MOD.ranks(p, {"mom"}, neutral="sector"), MOD.ranks(p, {"mom"})
    ok = plain["base"]["eligible"]
    raw = plain["base"]["raw"]["si"].where(ok)
    groups = MOD.component("p1")._liquidity_groups(p, 5)
    t = 450
    cells = sector.iloc[t] * 100 + groups.iloc[t]
    got = neutral["ranks"]["si"].iloc[t]
    checked = 0
    for _, names in cells.groupby(cells).groups.items():
        live = raw.iloc[t][names].dropna()
        if len(live) >= MOD.MIN_CELL:
            pd.testing.assert_series_equal(got[live.index], live.rank(pct=True),
                                           check_names=False)
            checked += 1
        else:
            pd.testing.assert_series_equal(got[names], plain["ranks"]["si"].iloc[t][names],
                                           check_names=False)
    assert checked >= 15
    small = ["N000", "N001", "N002"]
    assert got[small].equals(plain["ranks"]["si"].iloc[t][small])
    parts = RM.compose(p, min_price=5.0, factors="market+sector")
    at = np.flatnonzero(parts["rebalance"].to_numpy())
    last = at[at <= t][-1]
    pd.testing.assert_series_equal(neutral["ranks"]["mom"].iloc[t],
                                   parts["rank"].iloc[last].where(ok.iloc[t]), check_names=False)
    a, b = _run(p)["weights"], _run(p, neutral="sector")["weights"]
    assert not a.equals(b) and b.abs().sum(axis=1).max() <= 0.90 + 1e-9
    pd.testing.assert_frame_equal(a, _run(p, neutral="")["weights"])


def test_every_config_param_is_a_keyword_valid_rejects_bad_books_and_the_pins_hold():
    keys = inspect.signature(XS.signals).parameters
    assert set(XS.params) <= set(keys) and set(XS.params) == set(XS.docs)
    assert XS.valid(XS.params) and XS.valid({}) and XS.fill == "open"
    assert XS.order_notional == 1000.0
    for bad in ({"long_sleeves": "si+foo"}, {"long_sleeves": ""}, {"short_sleeves": "si+si"},
                {"long_sleeves": None}, {"combine": "blend"}, {"short_gate": "junk"},
                {"combine": "book", "short_sleeves": "si"}, {"q_out": 0.9},
                {"q_in": 1.0}, {"q_out": 0.5}, {"gross": 1.2}, {"gross": 0.0}, {"slots": 0},
                {"score_smooth": 0}, {"w_mom": -1.0}, {"long_sleeves": "mom", "w_mom": 0.0},
                {"si_mid": 0.25}, {"short_stop": 0.0}, {"neutral": "industry"}):
        assert not XS.valid({**XS.params, **bad}), bad
    assert XS.valid({**XS.params, "long_sleeves": "si", "short_sleeves": "mom"})
    assert XS.valid({**XS.params, "long_sleeves": "si+mom+prof", "w_mom": 0.0})
    for key, name in MOD.COMPONENTS.items():
        path = MOD._STRATEGIES / name / "strategy.py"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == MOD.SHA256[key], name
        assert MOD.component(key).__name__ == f"_xsc_{key}"


def test_a_component_that_changed_refuses_to_load(monkeypatch):
    monkeypatch.setitem(MOD.SHA256, "ah", "0" * 64)
    MOD.component.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="changed"):
            MOD.component("ah")
    finally:
        monkeypatch.undo()
        MOD.component.cache_clear()
    assert MOD.component("ah").__name__ == "_xsc_ah"


def _research():
    """research.py, loaded by path the way the notebooks and the log's commands load it."""
    path = MOD._STRATEGIES / "xs_si_mom_composite" / "research.py"
    spec = importlib.util.spec_from_file_location("_xsc_research", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _squeezed_book():
    """T2's 2024-05-14 in miniature, on a $100 book: A is short 45% and B long 45%; A opens
    bar 2 at 30 (from 10), its stop asks for the cover (target 0) and lock_cash lets it buy
    back only 1 of 4.5 shares; C's new short on the same bar, sent after the cover as the lab's
    call order does, finds no free cash and is refused (NoCashShort). A then falls back."""
    import vectorbt as vbt
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    close = pd.DataFrame({"A": [10.0, 10.0, 25.0, 20.0, 18.0], "B": [10.0] * 5,
                          "C": [10.0, 10.0, 10.0, 9.0, 8.0]}, index=idx)
    open_ = pd.DataFrame({"A": [10.0, 10.0, 30.0, 22.0, 19.0], "B": [10.0] * 5,
                          "C": [10.0, 10.0, 10.0, 9.5, 8.5]}, index=idx)
    orders = pd.DataFrame({"A": [-0.45, np.nan, 0.0, np.nan, np.nan],
                           "B": [0.45, np.nan, np.nan, np.nan, np.nan],
                           "C": [np.nan, np.nan, -0.2, np.nan, np.nan]}, index=idx)
    seq = np.tile(np.arange(3), (5, 1))
    seq[2] = [0, 2, 1]                                   # the cover before the new short
    pf = vbt.Portfolio.from_orders(close, size=orders, size_type="targetpercent", price=open_,
                                   init_cash=100.0, group_by=True, cash_sharing=True,
                                   lock_cash=True, call_seq=seq, freq="1D", log=True)
    return pf, types.SimpleNamespace(close=close, open_=open_), orders.ffill().fillna(0.0)


def test_a_stuck_cover_and_a_refused_short_are_booked_from_their_due_open():
    """The stuck-order estimate (RESEARCH_LOG §9, corrected in §12): each gap is the position
    minus the shares the order ASKED for, held from the due bar's OPEN, when the order should
    have filled. The first version started at the due bar's close and so dropped the due bar's
    own move, the biggest piece of GME's windfall (it opened at 64.83 and closed at 48.75)."""
    rs = _research()
    pf, window, _ = _squeezed_book()
    value = pf.value().to_numpy()
    assert np.allclose(value, [100.0, 100.0, 27.5, 45.0, 52.0])
    assert np.allclose(pf.assets()["A"], [-4.5, -4.5, -3.5, -3.5, -3.5])      # 1 of 4.5 covered
    assert (pf.assets()["C"] == 0).all()                                       # short refused
    rows, gap = rs._gaps(pf, window, 100.0)
    by = {r["name"]: r for r in rows}
    assert set(by) == {"A", "C"}
    assert by["A"]["kind"] == "cover" and by["A"]["wanted"] == 4.5 and by["A"]["got"] == 1.0
    assert by["C"]["kind"] == "new short" and by["C"]["got"] == 0.0
    assert by["C"]["status info"] == 6                                         # NoCashShort
    # C asked for -0.2 shares (20% of the $10 the book was worth at the open, at $10), not
    # 20% of the close value: the gap is what the order asked, never a re-estimate
    assert by["A"]["gap shares"] == -3.5 and by["C"]["gap shares"] == 0.2
    a = -3.5 * np.array([0.0, 0.0, (25 - 30) / 100, (20 - 25) / 27.5, (18 - 20) / 45])
    c = 0.2 * np.array([0.0, 0.0, (10 - 10) / 100, (9 - 10) / 27.5, (8 - 9) / 45])
    np.testing.assert_allclose(gap.to_numpy(), a + c, atol=1e-12)
    assert by["A"]["of which the due bar"] == round(a[2], 4) == 0.175          # open -> close
    assert by["A"]["bars stuck"] == by["C"]["bars stuck"] == 3
    # removing the gap leaves the book as if both orders had filled at the due open
    shares = rs.gap_pnl(2.0, [1, 2, 3, 4, 5, 6], [1.5, 2.5, 3.5, 4.5, 5.5, 6.5], np.ones(6),
                        at=1, nxt=4)
    np.testing.assert_allclose(shares, [0.0, 1.0, 2.0, 2.0, 1.0, 0.0])
    assert shares.sum() == 2.0 * (5 - 2)                    # telescopes to open[nxt] - open[at]


def test_off_target_counts_a_position_held_against_a_zero_target():
    """The audit's off-target name-days count both failures of the squeezed book: A held short
    against a zero target (the part-filled cover) and C's target never held (the refused
    short), 3 bars each. The first count required a non-zero target and saw only C."""
    rs = _research()
    pf, window, weights = _squeezed_book()
    off = rs.off_target(pf.assets(), weights, window.close)
    assert off.sum().to_dict() == {"A": 3, "B": 0, "C": 3}
    assert off["A"].iloc[2:].all() and off["C"].iloc[2:].all() and not off.iloc[:2].any().any()
    first = (np.sign(pf.assets()) != np.sign(weights)) & (weights != 0)
    assert int(first.to_numpy().sum()) == 3
    hidden = window.close.copy()
    hidden.iloc[3:, 0] = np.nan                            # unpriced name-days are not counted
    assert int(rs.off_target(pf.assets(), weights, hidden)["A"].sum()) == 1


def test_research_refuses_the_locked_tests():
    rs = _research()
    for fn in (lambda: rs.audit(("A_U0",), ("T1", "TL")), lambda: rs.stuck(("A_U0",), ("T4",))):
        with pytest.raises(SystemExit, match="locked"):
            fn()


# --- tune.py: the optimizer's rebuild of the strategy with every knob ------------------------

def _tune():
    spec = importlib.util.spec_from_file_location(
        "xs_tune_test", strategies.STRATEGIES_DIR / "xs_si_mom_composite" / "tune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("over", [{}, SI, MOM, {"long_sleeves": "si", "short_sleeves": "mom"},
                                  {"combine": "book"}, {"neutral": "sector"},
                                  {"short_gate": "prof"},
                                  {"long_sleeves": "si+mom+prof", "short_sleeves": "si+mom+prof"}])
def test_tune_rebuilds_signals_exactly_at_the_shipped_knobs(gapped, over):
    """At version B's knobs (the shipped rules) tune.weights IS signals(): same weights, borrow
    and cap, for every ranking variant the config can express."""
    tune = _tune()
    mine = tune.weights({}, {**tune.standard(), **over}, panel=gapped)
    ref = _run(gapped, **over)
    assert (ref["weights"] != 0).sum().sum() > 1000
    pd.testing.assert_frame_equal(mine["weights"], ref["weights"])
    pd.testing.assert_frame_equal(mine["borrow"], ref["borrow"])
    assert mine["retrim_k"] == ref["retrim_k"]


def test_tune_gives_each_side_its_own_slots_bands_and_share(panel):
    """long_slots / short_slots / long_share: at most that many names a side, each at gross ×
    share ÷ slots; entries obey each side's own band; retrim_k 0 switches the cap off."""
    tune = _tune()
    k = {**tune.standard(), "long_slots": 4, "short_slots": 9, "long_share": 0.7,
         "long_entry": 0.95, "short_entry": 0.05, "retrim_k": 0.0}
    out = tune.weights({}, k, panel=panel)
    w = out["weights"]
    assert (w > 0).sum(axis=1).max() == 4 and (w < 0).sum(axis=1).max() == 9
    np.testing.assert_allclose(np.unique(w[w > 0].stack().round(12)), 0.9 * 0.7 / 4)
    np.testing.assert_allclose(np.unique(-w[w < 0].stack().round(12)), 0.9 * 0.3 / 9)
    assert out["retrim_k"] is None
    held = {(s, o) for side, s, o, _, _ in out["exits"] if side > 0}
    assert held                                              # longs were opened and closed


def test_every_standard_value_lies_in_its_own_search_space():
    """A standard value the search cannot reach would make version B a setting nobody tests
    (the repo's one-parameter-one-declaration rule)."""
    tune = _tune()
    for name, _, value, space, _ in tune.KNOBS:
        if isinstance(space, list):
            assert value in space, name
        else:
            low, high, step = space
            steps = (value - low) / step
            assert low <= value <= high and abs(steps - round(steps)) < 1e-9, name
    assert tune.valid(tune.standard())
    for name, x, k in tune.neighbours(tune.standard(), tune.NAMES):
        assert k[name] == x and sum(k[n] != v for n, v in tune.standard().items()) == 1


def test_the_search_plans_never_draw_a_setting_the_book_ignores_or_rejects():
    """Stage 1 changes one knob at a time, stage 3 two to five; neither draws an invalid book
    (two books with different signals) or a change the book ignores (a profitability weight
    without a prof sleeve, a beta cap without beta sizing, stop_from without a stop), and stage
    3 is the same sequence on every run (it never looks at results)."""
    tune = _tune()
    base = tune.standard()
    sweep = tune.sweep_plan(base, tune.NAMES)
    assert all(len(tune.changes(k, base)) == 1 and tune.usable(k, base) for k in sweep)
    changed = {next(iter(tune.changes(k, base))) for k in sweep}
    assert {"w_prof", "max_size", "stop_from"}.isdisjoint(changed)
    assert len(sweep) == sum(len(g) - 1 for n, g in tune.GRID.items()
                             if n not in {"w_prof", "max_size", "stop_from"})
    draws = [k for k, _ in zip(tune.pool_plan(base, tune.NAMES), range(300), strict=False)]
    again = [k for k, _ in zip(tune.pool_plan(base, tune.NAMES), range(300), strict=False)]
    assert draws == again
    assert all(2 <= len(tune.changes(k, base)) <= 5 and tune.usable(k, base) for k in draws)
    assert max(k["account"] for k in draws + sweep) <= 100_000


def test_combining_keeps_only_the_changes_that_raise_the_score_together():
    """Stage 2 on a toy score: four single changes help, but two of them clash. Best first, it
    keeps more slots, the stop and the bigger account, and drops the higher gross that the
    extra slots make harmful. A second value of a knob already changed is never tried."""
    tune = _tune()
    base = tune.standard()
    gain = {("long_slots", 20): 2.0, ("long_slots", 25): 1.0, ("long_stop", 5.0): 1.5,
            ("gross", 0.95): 1.0, ("account", 100_000): 0.5}

    def fit(k):
        diff = tune.changes(k, base)
        total = sum(gain.get(item, -1.0) for item in diff.items())
        return total - (5.0 if diff.get("long_slots") == 20 and "gross" in diff else 0.0)

    singles = tune.sweep_plan(base, ["long_slots", "gross", "long_stop", "account"])
    final, path = tune.improve(base, singles, fit, run=lambda k: None)
    assert tune.changes(final, base) == {"long_stop": 5.0, "long_slots": 20, "account": 100_000}
    assert [(p["change"], p["kept"]) for p in path] == [
        ("long_slots = 20", True), ("long_stop = 5.0", True), ("gross = 0.95", False),
        ("account = 100000", True)]


def _stop_case(close_a, close_b, score_a, *, trailing=False, k=3.0):
    """Three names: A ranked top (a long), B bottom (a short), C in the middle; 1% daily
    volatility; A's and B's closes as given."""
    tune = _tune()
    n = len(close_a)
    idx = pd.bdate_range("2024-01-01", periods=n, tz="UTC")
    score = pd.DataFrame({"A": score_a, "B": 0.0, "C": 0.5}, index=idx)
    close = pd.DataFrame({"A": close_a, "B": close_b, "C": 100.0}, index=idx)
    return tune._select(score, close, pd.DataFrame(True, index=idx, columns=close.columns),
                        long_entry=0.9, long_exit=0.8, short_entry=0.1, short_exit=0.2,
                        long_slots=1, short_slots=1, max_hold=10_000, short_stop=1000.0,
                        long_stop=k, short_vol_stop=k, trailing=trailing,
                        sigma=np.full((n, 3), 0.01))


def test_the_volatility_stop_sells_k_sigmas_below_entry_and_waits_for_the_zone_to_reset():
    """k = 3 at 1% volatility: A falls 5% on bar 10 (below 97) and is sold that close; it
    stays out while it still ranks in the long zone, and is bought again only after its rank
    left the zone (bar 20) and came back (bar 21). B rises 4% (above 103): covered."""
    a = [100.0] * 10 + [95.0] * 20
    b = [100.0] * 12 + [104.0] * 18
    rank = [1.0] * 20 + [0.5] + [1.0] * 9
    longs, shorts, exits = _stop_case(a, b, rank)
    causes = {(side, s): (closed, why) for side, s, _, closed, why in exits}
    assert causes[(1, "A")][1] == "stop" and causes[(-1, "B")][1] == "stop"
    assert causes[(1, "A")][0] == longs.index[10] and causes[(-1, "B")][0] == shorts.index[12]
    assert not longs["A"].iloc[10:21].any() and longs["A"].iloc[21:].all()
    assert not shorts["B"].iloc[12:].any()                  # B never left the short zone


def test_a_trailing_stop_follows_the_best_close_a_fixed_one_does_not():
    """A runs to 110 and falls back to 106: 3.6% under its best close (> 3 × 1%), 6% above
    its entry. Only the trailing stop sells it."""
    a = [100.0] * 3 + [110.0] * 3 + [106.0] * 10
    flat = [100.0] * 16
    fixed = _stop_case(a, flat, [1.0] * 16)[2]
    trail = _stop_case(a, flat, [1.0] * 16, trailing=True)[2]
    assert not any(why == "stop" for *_, why in fixed)
    assert [(s, why) for side, s, _, _, why in trail if side > 0] == [("A", "stop")]


def test_order_costs_fall_then_rise_with_the_account_size(panel):
    """The $1 minimum dominates a small account's orders, the square-root impact a large one's:
    the cost per dollar first falls with the account, then rises."""
    tune = _tune()
    costs = types.SimpleNamespace(min_commission=1.0, commission_per_share=0.005, spread_bps=2.0)
    w = pd.DataFrame(0.0, index=panel.close.index, columns=panel.close.columns)
    w.iloc[300:, :5] = 0.03
    cost = {acc: float(tune.order_fees(panel, panel, w, acc, costs).iloc[300, :5].mean())
            for acc in (10_000, 330_000, 100_000_000)}
    assert cost[10_000] > cost[330_000] < cost[100_000_000]
    assert cost[10_000] > 1.0 / 300 + 1e-4                 # $1 on a $300 order, plus spread
