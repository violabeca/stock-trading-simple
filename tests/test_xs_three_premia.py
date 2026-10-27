"""xs_three_premia (proposals lab cross-strategy): short interest × residual momentum ×
within-sector profitability in one dollar-neutral book — a thin wrapper of
xs_si_mom_composite/strategy.py, loaded by path and pinned by SHA-256. Synthetic panels, no DB."""
import hashlib
import importlib.util
import inspect
import types

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies
from trading_simple.data import Panel

XS = strategies.load("xs_three_premia")
MOD = types.SimpleNamespace(**XS.signals.__globals__)          # the loaded strategy.py
XSC = MOD.core()                                               # xs_si_mom_composite, by path
QJ = XSC.component("qj")
LABELS = {"B_U0": {}, "C_prof": {"long_sleeves": "prof", "short_sleeves": "prof"},
          "C_mix3": {"combine": "book"}, "I1": {"w_prof": 0.5}, "I2": {"w_growth": 1.0}}
W1 = {"si": 1.0, "mom": 1.0, "prof": 1.0}
RULES = {"q_in": 0.9, "q_out": 0.8, "slots": 15, "max_hold": 126, "short_stop": 1.0}


def _panel(n=620, m=150, seed=11, gaps=False):
    """``m`` names on one market factor with persistent own drifts (momentum to rank),
    days-to-cover random walks (short interest to rank), dollar volumes spread over two orders
    of magnitude, four names too crowded to short, and quarterly-stepping fundamentals in five
    sectors: every 7th name has no gross profit (ROE stands in, so some sector × ROE groups hold
    fewer than 5 names and fall back to the measure-wide rank) and a few of those have negative
    equity (no profitability at all: 0.5 in the composite). ``gaps``: some days-to-cover
    missing (ineligible) and one name listed late."""
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
    quarter = np.arange(n) // 63                         # fundamentals step once a quarter
    steps = np.cumprod(1 + rng.normal(0.0, 0.08, (quarter.max() + 1, m)), axis=0)[quarter]
    gp = rng.uniform(0.05, 0.6, m) * 1e9 * steps
    roe_only = np.arange(m) % 7 == 0
    gp[:, roe_only] = np.nan                             # no gross profit: ROE stands in
    ni = (rng.uniform(-0.1, 0.3, m) * 1e9 * np.cumprod(
        1 + rng.normal(0.0, 0.1, (quarter.max() + 1, m)), axis=0)[quarter])
    equity = np.full((n, m), 1e9)
    equity[:, roe_only & (np.arange(m) % 3 == 0)] = -1e8     # negative equity: no ROE either
    frame = lambda a: pd.DataFrame(a, index=idx, columns=names)  # noqa: E731
    feats = {"short_interest": frame(dtc), "fund_shares": frame(np.tile(shares, (n, 1))),
             "fund_gp_ttm": frame(gp), "fund_assets": frame(1e9), "fund_ni_ttm": frame(ni),
             "fund_equity": frame(equity),
             "sector": frame(np.tile((np.arange(m) % 5).astype(float), (n, 1)))}
    open_ = close * (1 + rng.normal(0, 0.002, (n, m)))
    if gaps:
        si = feats["short_interest"]
        si.iloc[300:340, 10:20] = np.nan                 # ineligible for p1, still ranked by qj
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


def _hand_prof_rank(p, t):
    """quality_junk's profitability percentile on bar ``t``, recomputed from its definition:
    GP TTM / assets, else NI TTM / equity (equity > 0), among the names with close >= $5, a
    ratio and a 252-bar volatility (half the window), ranked within sector × measure; a group
    under 5 names ranks against every name on its measure. Returns the ranks and the number of
    names ranked in own groups / by the fallback."""
    f = p.features
    gpa = (f["fund_gp_ttm"] / f["fund_assets"].where(f["fund_assets"] > 0)).iloc[t]
    roe = (f["fund_ni_ttm"] / f["fund_equity"].where(f["fund_equity"] > 0)).iloc[t]
    prof = gpa.where(gpa.notna(), roe)
    vol = p.close.pct_change(fill_method=None).rolling(252, min_periods=126).std().iloc[t]
    row = prof.where((p.close.iloc[t] >= 5.0) & prof.notna() & vol.notna())
    measure = gpa.notna()
    out, own, fallback = pd.Series(np.nan, index=row.index), 0, 0
    for (_, m), names in row.groupby([f["sector"].iloc[t], measure]).groups.items():
        live = row[names].dropna()
        if len(live) >= QJ.MIN_GROUP:
            out[live.index] = live.rank(pct=True)
            own += len(live)
        elif len(live):
            out[live.index] = row[measure == m].dropna().rank(pct=True)[live.index]
            fallback += len(live)
    return out, own, fallback


def test_prof_ranks_reproduce_quality_junk_within_sector_and_measure_with_the_fallback(panel):
    """Spec test 1: the PROF sleeve is quality_junk.compose(panel, min_price=5)['ranks']['prof']
    on the eligible cells — within sector × measure, a group under MIN_GROUP 5 on its measure."""
    r = XSC.ranks(panel, {"prof"})
    prof, ok = r["ranks"]["prof"], r["base"]["eligible"]
    ref = QJ.compose(panel, min_price=5.0)["ranks"]["prof"]
    assert ok.to_numpy().sum() > 50_000 and prof.where(ok).notna().to_numpy().sum() > 40_000
    pd.testing.assert_frame_equal(prof.where(ok), ref.where(ok))
    seen_own = seen_fallback = 0
    for t in (200, 380, 610):                            # independent re-derivation
        want, own, fallback = _hand_prof_rank(panel, t)
        pd.testing.assert_series_equal(ref.iloc[t], want, check_names=False)
        seen_own, seen_fallback = seen_own + own, seen_fallback + fallback
    assert seen_own > 300 and seen_fallback > 10        # both branches were exercised
    # C_prof's ranking sees only p1-eligible cells; a name without profitability sits at 0.5
    score = XSC.side_score(r["ranks"], ["prof"], W1, ok, 1)
    pd.testing.assert_frame_equal(score, ref.fillna(0.5).where(ok).rank(axis=1, pct=True))
    assert (ok & ref.isna()).to_numpy().sum() > 1000


def test_the_three_way_composite_is_the_mean_of_the_three_ranks_before_smoothing(gapped):
    """Spec test 2: (si + mom + prof) / 3, each missing rank at 0.5, on the eligible names —
    then averaged over 5 bars and re-ranked; and B_U0's book is the core's book on it."""
    r = XSC.ranks(gapped, {"si", "mom", "prof"})
    ok, rk = r["base"]["eligible"], r["ranks"]
    mean = ((rk["si"].fillna(0.5) + rk["mom"].fillna(0.5) + rk["prof"].fillna(0.5)) / 3).where(ok)
    for k in ("si", "mom", "prof"):                      # every sleeve has holes to fill
        assert (ok & rk[k].isna()).to_numpy().sum() > 100, k
    raw = XSC.side_score(rk, ["si", "mom", "prof"], W1, ok, 1)
    pd.testing.assert_frame_equal(raw, mean.rank(axis=1, pct=True))
    score = XSC.side_score(rk, ["si", "mom", "prof"], W1, ok, 5)
    pd.testing.assert_frame_equal(
        score, mean.rolling(5, min_periods=1).mean().where(ok).rank(axis=1, pct=True))
    longs, shorts = XSC.select(score, score, gapped.close, r["base"]["shortable"],
                                two_pass=False, **RULES)
    w = XSC.component("p1")._weights(longs, shorts, r["base"]["beta"], gross=0.90,
                                      long_share=0.5, slots=15, sizing="equal")
    w = XSC.component("ah")._on_priced_opens(w, gapped.open_)
    got = _run(gapped)
    pd.testing.assert_frame_equal(got["weights"], w)
    pd.testing.assert_frame_equal(got["borrow"], r["base"]["borrow"])
    assert got["retrim_k"] == 2.0 and (w != 0).to_numpy().sum() > 1000


def test_combine_book_with_three_sleeves_is_a_third_of_each_single_sleeve_book(panel):
    """Spec test 3: C_mix3 = ⅓ w(C_si) + ⅓ w(C_mom) + ⅓ w(C_prof), netted per name."""
    mix = _run(panel, combine="book")
    books = [_run(panel, long_sleeves=k, short_sleeves=k) for k in ("si", "mom", "prof")]
    want = sum((1.0 / 3.0) * b["weights"] for b in books)
    pd.testing.assert_frame_equal(mix["weights"], want)
    pd.testing.assert_frame_equal(mix["borrow"], books[0]["borrow"])
    w = mix["weights"]
    assert (w.abs().sum(axis=1) <= 0.90 + 1e-9).all()
    assert (w > 0).sum(axis=1).max() > 15 and (w > 0).sum(axis=1).max() <= 45
    a, b, c = (x["weights"] for x in books)
    conflict = ((a > 0) & (c < 0)) | ((b > 0) & (c < 0)) | ((c > 0) & ((a < 0) | (b < 0)))
    assert conflict.to_numpy().any()                     # opposite views net out


@pytest.mark.parametrize("where", ["mid-month", "month-end"])
def test_a_truncated_panel_reproduces_the_weights_through_its_last_bar(gapped, where):
    """Spec test 4: no bar's weights move when later bars are removed (B_U0, the controls and
    both iterations; a month end is a MOM decision bar in the full panel)."""
    idx = gapped.close.index
    month = idx.year * 12 + idx.month
    ends = np.flatnonzero(month[1:] != month[:-1])
    end = idx[ends[20] - 7] if where == "mid-month" else idx[ends[20]]
    for label, over in LABELS.items():
        full, part = _run(gapped, **over), _run(_cut(gapped, end), **over)
        assert (part["weights"].iloc[-60:] != 0).any().any(), label
        pd.testing.assert_frame_equal(full["weights"].loc[:end], part["weights"], obj=label)
        pd.testing.assert_frame_equal(full["borrow"].loc[:end], part["borrow"], obj=label)


def test_every_config_param_is_a_keyword_valid_rejects_bad_books_and_the_pins_hold():
    """Spec test 5: config ↔ signals() keywords, valid(), and the SHA-256 pins."""
    keys = inspect.signature(XS.signals).parameters
    assert set(XS.params) <= set(keys) and set(XS.params) == set(XS.docs)
    assert XS.valid(XS.params) and XS.valid({}) and XS.fill == "open"
    assert XS.order_notional == 1000.0
    assert {k: XS.params[k] for k in ("long_sleeves", "short_sleeves", "w_si", "w_mom", "w_prof",
                                      "w_growth", "combine", "gross", "slots", "q_in", "q_out")
            } == {"long_sleeves": "si+mom+prof", "short_sleeves": "si+mom+prof", "w_si": 1.0,
                  "w_mom": 1.0, "w_prof": 1.0, "w_growth": 0.0, "combine": "rank",
                  "gross": 0.9, "slots": 15, "q_in": 0.9, "q_out": 0.8}
    assert XS.data["features"] == ["short_interest", "fund_shares", "fund_gp_ttm",
                                   "fund_assets", "fund_ni_ttm", "fund_equity"]
    for bad in ({"long_sleeves": "si+growth"}, {"short_sleeves": ""}, {"w_growth": -1.0},
                {"w_growth": 1.0, "combine": "book"}, {"w_growth": "x"},
                {"combine": "book", "short_sleeves": "si"}, {"q_out": 0.9}, {"gross": 1.2},
                {"w_prof": -0.5}, {"long_sleeves": "prof", "w_prof": 0.0}):
        assert not XS.valid({**XS.params, **bad}), bad
    for good in LABELS.values():
        assert XS.valid({**XS.params, **good}), good
    core_sig = set(inspect.signature(XSC.signals).parameters) - {"panel", "_"}
    assert core_sig == set(MOD._CORE_KEYS) and set(keys) - {"panel", "_"} == core_sig | {
        "w_growth"}
    assert hashlib.sha256(MOD.CORE.read_bytes()).hexdigest() == MOD.CORE_SHA256
    assert XSC.__name__ == "_xsp_core"
    for key, name in XSC.COMPONENTS.items():           # the core's own component pins
        path = XSC._STRATEGIES / name / "strategy.py"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == XSC.SHA256[key], name


def test_a_changed_core_refuses_to_load(monkeypatch):
    g = XS.signals.__globals__                           # the loaded strategy.py's namespace
    monkeypatch.setitem(g, "CORE_SHA256", "0" * 64)
    g["core"].cache_clear()
    try:
        with pytest.raises(RuntimeError, match="changed"):
            g["core"]()
    finally:
        monkeypatch.undo()
        g["core"].cache_clear()
    assert g["core"]().__name__ == "_xsp_core"


def test_with_w_growth_off_every_label_is_the_cores_call_exactly(gapped):
    for label, over in LABELS.items():
        if over.get("w_growth"):
            continue
        params = {**XS.params, **over}
        mine = XS.signals(gapped, **params)
        theirs = XSC.signals(gapped, **{k: v for k, v in params.items() if k != "w_growth"})
        pd.testing.assert_frame_equal(mine["weights"], theirs["weights"], obj=label)
        pd.testing.assert_frame_equal(mine["borrow"], theirs["borrow"], obj=label)


def test_i2_adds_the_growth_rank_to_the_long_composite_only(gapped):
    """I2: the long score is (si + mom + prof + growth) / 4 (missing at 0.5), the short score
    the plain three-premia one; longs from the long score, shorts from the short score with the
    longs masked; growth = quality_junk.compose(w_growth=1)['ranks']['growth']."""
    r = XSC.ranks(gapped, {"si", "mom", "prof"})
    ok, base = r["base"]["eligible"], r["base"]
    growth = QJ.compose(gapped, min_price=5.0, w_growth=1.0)["ranks"]["growth"]
    assert growth.where(ok).notna().to_numpy().sum() > 10_000
    rank = {**r["ranks"], "growth": growth}
    w = {**W1, "growth": 1.0}
    long_score = XSC.side_score(rank, ["si", "mom", "prof", "growth"], w, ok, 5)
    short_score = XSC.side_score(rank, ["si", "mom", "prof"], w, ok, 5)
    mean4 = ((rank["si"].fillna(0.5) + rank["mom"].fillna(0.5) + rank["prof"].fillna(0.5)
              + growth.fillna(0.5)) / 4).where(ok)
    pd.testing.assert_frame_equal(
        long_score, mean4.rolling(5, min_periods=1).mean().where(ok).rank(axis=1, pct=True))
    longs, _, _ = XSC.component("p1")._select(long_score, gapped.close, base["shortable"],
                                               sides="long", **RULES)
    _, shorts, _ = XSC.component("p1")._select(short_score.mask(longs), gapped.close,
                                                base["shortable"], sides="short", **RULES)
    want = XSC.component("p1")._weights(longs, shorts, base["beta"], gross=0.90,
                                         long_share=0.5, slots=15, sizing="equal")
    want = XSC.component("ah")._on_priced_opens(want, gapped.open_)
    got = _run(gapped, w_growth=1.0)["weights"]
    pd.testing.assert_frame_equal(got, want)
    plain = _run(gapped)["weights"]
    assert not got.clip(lower=0).equals(plain.clip(lower=0))   # the long book moved


@pytest.mark.parametrize("label", ["B_U0", "C_prof", "I1", "I2"])
def test_book_invariants_sides_gross_and_one_weight_per_hold(gapped, label):
    p = {**XS.params, **LABELS[label]}
    w = _run(gapped, **LABELS[label])["weights"]
    assert (w > 0).sum(axis=1).max() == p["slots"] and (w < 0).sum(axis=1).max() == p["slots"]
    assert w.abs().sum(axis=1).max() <= p["gross"] + 1e-9
    slot = p["gross"] * 0.5 / p["slots"]
    assert np.allclose(w.abs().to_numpy()[w.to_numpy() != 0], slot)    # equal, constant slots
    side = np.sign(w)
    hold = side.ne(side.shift(fill_value=0.0)).cumsum()             # one id per hold and name
    for name in w.columns[(w != 0).any()]:
        held = side[name] != 0
        assert w[name][held].groupby(hold[name][held]).nunique().max() == 1
    assert not ((w > 0) & (w.shift(1) < 0)).to_numpy().any()          # never flipped in a bar


def _research():
    """research.py, loaded by path the way the log's commands load it."""
    path = MOD.CORE.parent.parent / "xs_three_premia" / "research.py"
    spec = importlib.util.spec_from_file_location("_xsp_research", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _squeezed_book():
    """xs_si_mom_composite's T2 2024-05-14 in miniature, on a $100 book: A is short 45% and B
    long 45%; A opens bar 2 at 30 (from 10), its stop asks for the cover (target 0) and
    lock_cash lets it buy back only 1 of 4.5 shares; C's new short on the same bar, sent after
    the cover as the lab's call order does, finds no free cash and is refused (NoCashShort)."""
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
    """The reconciliation's stuck-order P&L (xs_si_mom_composite's review fix): each gap is the
    position minus the shares the order ASKED for, held from the due bar's OPEN."""
    rs = _research()
    pf, window, weights = _squeezed_book()
    assert np.allclose(pf.value().to_numpy(), [100.0, 100.0, 27.5, 45.0, 52.0])
    assert np.allclose(pf.assets()["A"], [-4.5, -4.5, -3.5, -3.5, -3.5])      # 1 of 4.5 covered
    assert (pf.assets()["C"] == 0).all()                                       # short refused
    rows, gap = rs._gaps(pf, window, 100.0)
    by = {r["name"]: r for r in rows}
    assert set(by) == {"A", "C"}
    assert by["A"]["kind"] == "cover" and by["A"]["wanted"] == 4.5 and by["A"]["got"] == 1.0
    assert by["C"]["kind"] == "new short" and by["C"]["got"] == 0.0
    assert by["C"]["status info"] == 6                                         # NoCashShort
    assert by["A"]["gap shares"] == -3.5 and by["C"]["gap shares"] == 0.2
    a = -3.5 * np.array([0.0, 0.0, (25 - 30) / 100, (20 - 25) / 27.5, (18 - 20) / 45])
    c = 0.2 * np.array([0.0, 0.0, (10 - 10) / 100, (9 - 10) / 27.5, (8 - 9) / 45])
    np.testing.assert_allclose(gap.to_numpy(), a + c, atol=1e-12)
    assert by["A"]["of which the due bar"] == round(a[2], 4) == 0.175          # open -> close
    assert by["A"]["bars stuck"] == by["C"]["bars stuck"] == 3
    shares = rs.gap_pnl(2.0, [1, 2, 3, 4, 5, 6], [1.5, 2.5, 3.5, 4.5, 5.5, 6.5], np.ones(6),
                        at=1, nxt=4)
    np.testing.assert_allclose(shares, [0.0, 1.0, 2.0, 2.0, 1.0, 0.0])
    # the frictionless preview fills both orders: on the due bar the engine trails it by
    # exactly the stuck gap's P&L (both books start the bar from the same $100 and positions)
    pv = rs.preview(window, weights, 100.0)
    engine = pf.value() / pf.value().shift(1).fillna(100.0) - 1.0
    np.testing.assert_allclose((engine - pv).to_numpy()[:3], [0.0, 0.0, 0.175], atol=1e-12)


def test_off_target_counts_a_position_held_against_a_zero_target():
    """Off-target name-days count both failures of the squeezed book: A held short against a
    zero target (the part-filled cover) and C's target never held (the refused short)."""
    rs = _research()
    pf, window, weights = _squeezed_book()
    off = rs.off_target(pf.assets(), weights, window.close)
    assert off.sum().to_dict() == {"A": 3, "B": 0, "C": 3}
    hidden = window.close.copy()
    hidden.iloc[3:, 0] = np.nan                            # unpriced name-days are not counted
    assert int(rs.off_target(pf.assets(), weights, hidden)["A"].sum()) == 1


def test_the_preview_is_the_frictionless_engine_book_retrim_included():
    """With no costs and no binding cash limit, vectorbt's book on the lab's event-driven
    targets IS the research preview, bar for bar — including a long that triples and is
    trimmed back by the engine's retrim loop, and a bar without an open."""
    import vectorbt as vbt

    from trading_simple import backtest as bt
    rs = _research()
    rng = np.random.default_rng(3)
    n, m = 120, 8
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    names = [f"S{i}" for i in range(m)]
    rets = rng.normal(0.0, 0.02, (n, m))
    rets[30:50, 2] = 0.06                                 # a runaway long: 2x its slot by bar ~42
    close = pd.DataFrame(50.0 * np.cumprod(1 + rets, axis=0), index=idx, columns=names)
    open_ = close.shift(1).fillna(50.0) * (1 + rng.normal(0, 0.003, (n, m)))
    open_.iloc[70, 5] = np.nan                            # a bar without an open
    side = np.repeat(rng.choice([-1.0, 0.0, 1.0], (n // 15 + 1, m)), 15, axis=0)[:n]
    side[20:60, 2] = 1.0                                  # S2 held long through its run
    weights = pd.DataFrame(side * 0.1, index=idx, columns=names)
    weights = XSC.component("ah")._on_priced_opens(weights, open_)
    targets = bt.rotation_targets(weights)

    def build(tg):
        return vbt.Portfolio.from_orders(close, size=tg, size_type="targetpercent",
                                         price=open_, init_cash=1000.0, group_by=True,
                                         cash_sharing=True, call_seq="auto", freq="1D")

    pf = bt._retrim_loop(build, targets, weights, 2.0 * weights)
    trims = (pf.orders.records_readable["Column"] == "S2").sum()
    assert trims >= 3                                     # entry, a retrim (or more), exit
    engine = pf.value() / pf.value().shift(1).fillna(1000.0) - 1.0
    pv = rs.preview(types.SimpleNamespace(close=close, open_=open_), weights, 1000.0, 2.0)
    np.testing.assert_allclose(engine.to_numpy(), pv.to_numpy(), atol=1e-10)
    plain = rs.preview(types.SimpleNamespace(close=close, open_=open_), weights, 1000.0)
    assert np.abs(plain.to_numpy() - pv.to_numpy()).max() > 1e-4    # the trim mattered


def test_research_refuses_the_locked_tests():
    rs = _research()
    for fn in (lambda: rs.audit(("B_U0",), ("T1", "TL")), lambda: rs.stuck(("B_U0",), ("T4",)),
               lambda: rs.premise(("TB", "T4")), lambda: rs.delta("B_U0", "C_mix3", ("TL",))):
        with pytest.raises(SystemExit, match="locked"):
            fn()
    assert set(rs.LABELS) >= set(LABELS) and rs.LABELS["C_mix3"] == {"combine": "book"}
