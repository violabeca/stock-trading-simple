"""xs_cross_leg (proposals lab cross-strategy): short-interest longs, residual-momentum shorts.

A thin wrapper of xs_si_mom_composite's strategy.py (loaded by path, pinned by SHA-256) with
``long_sleeves: si`` / ``short_sleeves: mom``; the reverse cross (``mom`` / ``si``) is the
falsification control. Synthetic panels, no DB."""
import hashlib
import importlib.util
import inspect

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies
from trading_simple.data import Panel

XS = strategies.load("xs_cross_leg")
CORE = XS.signals.__globals__["core"].__dict__      # the composite's strategy.py, as loaded
P1 = CORE["component"]("p1")
AH = CORE["component"]("ah")
REV = {"long_sleeves": "mom", "short_sleeves": "si"}
RULES = dict(q_in=0.9, q_out=0.8, slots=15, max_hold=126, short_stop=1.0)


def _wrapper():
    """strategies/xs_cross_leg/strategy.py itself (the loader keeps only its callables)."""
    path = XS.dir / "strategy.py"
    spec = importlib.util.spec_from_file_location("_xsl_test_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _panel(n=640, m=160, seed=11, gaps=False):
    """``m`` names on one market factor with persistent own drifts (residual momentum to rank),
    days-to-cover random walks (short interest to rank), dollar volumes spread over two orders
    of magnitude and a few names too crowded to short. ``gaps``: some days-to-cover missing
    (ineligible for p1) and one name listed late."""
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
    dtc = np.exp(np.log(rng.uniform(1.0, 6.0, m)) + np.cumsum(rng.normal(0, 0.05, (n, m)),
                                                               axis=0))
    shares = np.full(m, 3e9)
    shares[:4] = 1e5                                     # >30% of shares short: never shorted
    feats = {"short_interest": pd.DataFrame(dtc, index=idx, columns=names),
             "fund_shares": pd.DataFrame(np.tile(shares, (n, 1)), index=idx, columns=names)}
    open_ = close * (1 + rng.normal(0, 0.002, (n, m)))
    if gaps:
        feats["short_interest"].iloc[320:360, 10:20] = np.nan    # p1-ineligible for 40 bars
        close.iloc[:220, 30] = np.nan                              # a name listed late
        open_.iloc[:220, 30] = np.nan
    return Panel(close=close, open_=open_, high=close * 1.01, low=close * 0.99, volume=volume,
                 bar="1d", features=feats)


def _cut(p, end):
    keep = p.close.index <= end
    return Panel(close=p.close[keep], open_=p.open_[keep], high=p.high[keep], low=p.low[keep],
                 volume=p.volume[keep], bar="1d",
                 features={k: v[keep] for k, v in p.features.items()})


def _run(p, **over):
    return XS.signals(p, **{**XS.params, **over})


def _scores(p):
    """The SI and the MOM side scores as the core builds them (5-bar average, re-ranked among
    p1-eligible names), and p1's base frames."""
    r = CORE["ranks"](p, {"si", "mom"})
    w = {"si": 1.0, "mom": 1.0, "prof": 1.0}
    ok = r["base"]["eligible"]
    si = CORE["side_score"](r["ranks"], ["si"], w, ok, 5)
    mom = CORE["side_score"](r["ranks"], ["mom"], w, ok, 5)
    return si, mom, r["base"]


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def gapped():
    return _panel(gaps=True)


@pytest.mark.parametrize("sleeve", ["si", "mom"])
def test_with_identical_sleeves_the_two_pass_selection_equals_p1s_single_pass(gapped, sleeve):
    si, mom, base = _scores(gapped)
    score = {"si": si, "mom": mom}[sleeve]
    one = P1._select(score, gapped.close, base["shortable"], sides="both", **RULES)
    two = CORE["select"](score, score, gapped.close, base["shortable"], two_pass=True, **RULES)
    assert one[0].to_numpy().sum() > 1000 and one[1].to_numpy().sum() > 1000   # real books
    pd.testing.assert_frame_equal(one[0], two[0])
    pd.testing.assert_frame_equal(one[1], two[1])


def test_a_short_that_becomes_long_is_covered_on_that_bar():
    """Name A ranks in the short zone from bar 0 and reaches the long zone at bar 10: the long
    pass takes it at bar 10, its short score is masked there (NaN), so the short pass exits it
    on the same bar ("ineligible") and never re-enters it while it is held long. The weight
    flips from short to long in one order at bar 11's open."""
    idx = pd.date_range("2020-01-01", periods=30, freq="B", tz="UTC")
    cols = [f"S{i}" for i in range(20)]
    long_score = pd.DataFrame(0.5, index=idx, columns=cols)
    short_score = pd.DataFrame(0.5, index=idx, columns=cols)
    short_score["S0"] = 0.05
    long_score.iloc[10:, 0] = 0.95
    close = pd.DataFrame(10.0, index=idx, columns=cols)
    ok = pd.DataFrame(True, index=idx, columns=cols)
    rules = {**RULES, "slots": 2}
    longs, shorts = CORE["select"](long_score, short_score, close, ok, two_pass=True, **rules)
    assert shorts["S0"].iloc[:10].all() and not shorts["S0"].iloc[10:].any()
    assert longs["S0"].iloc[10:].all() and not longs["S0"].iloc[:10].any()
    assert not (longs & shorts).to_numpy().any()
    _, _, exits = P1._select(short_score.mask(longs), close, ok, sides="short", **rules)
    assert ("S0", idx[10], "ineligible") in {(s, z, c) for _, s, _, z, c in exits}
    w = P1._weights(longs, shorts, None, gross=0.9, long_share=0.5, slots=2, sizing="equal")
    assert w["S0"].iloc[10] == pytest.approx(-0.225) and w["S0"].iloc[11] == pytest.approx(0.225)


@pytest.mark.parametrize("over", [{}, REV])
def test_no_name_is_long_and_short_on_one_bar_and_flips_cover_on_the_flip_bar(gapped, over):
    si, mom, base = _scores(gapped)
    lo, sh = (si, mom) if not over else (mom, si)
    longs, shorts = CORE["select"](lo, sh, gapped.close, base["shortable"], two_pass=True,
                                   **RULES)
    assert not (longs & shorts).to_numpy().any()
    assert ((sh <= 0.1 + 1e-9) & longs).to_numpy().sum() > 100   # the long mask binds
    was_short = shorts.shift(1, fill_value=False)
    flips = was_short & longs                       # short yesterday, long today ...
    assert flips.to_numpy().any()
    assert not (flips & shorts).to_numpy().any()    # ... and no longer short today
    w = _run(gapped, **over)["weights"]
    assert ((w > 0).sum(axis=1) <= 15).all() and ((w < 0).sum(axis=1) <= 15).all()


@pytest.mark.parametrize("over", [{}, REV])
def test_the_book_is_15_plus_15_equal_constant_slots_within_gross(gapped, over):
    w = _run(gapped, **over)["weights"]
    assert (w > 0).sum(axis=1).max() == 15 and (w < 0).sum(axis=1).max() == 15
    assert w.abs().sum(axis=1).max() <= 0.90 + 1e-9
    assert np.allclose(w.abs().to_numpy()[w.to_numpy() != 0], 0.03)
    side = np.sign(w)
    hold = side.ne(side.shift(fill_value=0.0)).cumsum()
    for name in w.columns[(w != 0).any()]:
        held = side[name] != 0
        assert w[name][held].groupby(hold[name][held]).nunique().max() == 1


def test_c_rev_runs_and_mirrors_the_legs(panel):
    """The SI score drives C_U0's long leg and C_rev's short leg; the MOM score drives C_U0's
    short leg and C_rev's long leg. Each book is exactly: longs from its long score (p1's long
    pass), shorts from its short score masked where it holds a long (p1's short pass)."""
    si, mom, base = _scores(panel)
    books = {}
    for name, over, lo, sh in (("C_U0", {}, si, mom), ("C_rev", REV, mom, si)):
        got = _run(panel, **over)
        longs = P1._select(lo, panel.close, base["shortable"], sides="long", **RULES)[0]
        shorts = P1._select(sh.mask(longs), panel.close, base["shortable"], sides="short",
                            **RULES)[1]
        want = AH._on_priced_opens(
            P1._weights(longs, shorts, base["beta"], gross=0.9, long_share=0.5, slots=15,
                        sizing="equal"), panel.open_)
        pd.testing.assert_frame_equal(got["weights"], want)
        pd.testing.assert_frame_equal(got["borrow"], base["borrow"])
        assert got["retrim_k"] == 2.0
        books[name] = (longs, shorts)
    (u_l, u_s), (r_l, r_s) = books["C_U0"], books["C_rev"]
    assert u_l.to_numpy().sum() > 1000 and r_s.to_numpy().sum() > 1000
    # C_U0's longs ARE the SI long pass and C_rev's longs the MOM long pass (mirror images) ...
    pd.testing.assert_frame_equal(u_l, P1._select(si, panel.close, base["shortable"],
                                                  sides="long", **RULES)[0])
    pd.testing.assert_frame_equal(r_l, P1._select(mom, panel.close, base["shortable"],
                                                  sides="long", **RULES)[0])
    # ... and the two crosses are different books
    assert not u_l.equals(r_l) and not u_s.equals(r_s)


def test_the_wrapper_runs_the_composites_own_code(panel):
    """The controls C_si / C_mom / A_U0 are read from xs_si_mom_composite's saved results: the
    same strategy.py, so its loaded signals() with these sleeves gives the same book."""
    comp = strategies.load("xs_si_mom_composite")
    for over in ({}, REV):
        mine = _run(panel, **over)
        theirs = comp.signals(panel, **{**comp.params, "long_sleeves": "si",
                                        "short_sleeves": "mom", **over})
        pd.testing.assert_frame_equal(mine["weights"], theirs["weights"])
        pd.testing.assert_frame_equal(mine["borrow"], theirs["borrow"])


@pytest.mark.parametrize("where", ["mid-month", "month-end"])
def test_a_truncated_panel_reproduces_the_weights_through_its_last_bar(gapped, where):
    idx = gapped.close.index
    month = idx.year * 12 + idx.month
    ends = np.flatnonzero(month[1:] != month[:-1])
    end = idx[ends[21] - 7] if where == "mid-month" else idx[ends[21]]   # month-end: a MOM
    for over in ({}, REV):                                                # decision in full
        full, part = _run(gapped, **over), _run(_cut(gapped, end), **over)
        assert (part["weights"].iloc[-60:] < 0).any().any()               # MOM ranks live
        pd.testing.assert_frame_equal(full["weights"].loc[:end], part["weights"])
        pd.testing.assert_frame_equal(full["borrow"].loc[:end], part["borrow"])


def test_every_config_param_is_a_keyword_and_the_sha256_pin_holds():
    keys = inspect.signature(XS.signals).parameters
    assert set(XS.params) <= set(keys) and set(XS.params) == set(XS.docs)
    assert XS.params["long_sleeves"] == "si" and XS.params["short_sleeves"] == "mom"
    assert XS.valid(XS.params) and XS.valid({**XS.params, **REV})
    assert XS.valid({**XS.params, "short_sleeves": "si+mom"})              # iteration I1
    assert not XS.valid({**XS.params, "combine": "book"})                  # needs equal sleeves
    assert XS.fill == "open" and XS.order_notional == 1000.0
    assert XS.optuna["n_trials"] == 1 and XS.optuna["space"] == {}
    assert XS.optuna["objectives"] == ["alpha_cagr"]
    assert XS.data["features"] == ["short_interest", "fund_shares"]
    w = _wrapper()
    assert hashlib.sha256(w.CORE.read_bytes()).hexdigest() == w.SHA256
    assert w.core.__name__ == "_xsl_core" and CORE["__name__"] == "_xsl_core"
    for key, name in CORE["COMPONENTS"].items():                           # the core's own pins
        path = CORE["_STRATEGIES"] / name / "strategy.py"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == CORE["SHA256"][key], name


def test_a_changed_core_refuses_to_load(monkeypatch):
    w = _wrapper()
    monkeypatch.setattr(w, "SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="changed"):
        w.load_core()


def test_the_config_is_the_composites_but_for_the_sleeves():
    """C_si, C_mom and A_U0 are read from xs_si_mom_composite's saved results: they are fair
    controls only if every shared knob ships the same value here."""
    comp = strategies.load("xs_si_mom_composite")
    assert set(XS.params) - set(comp.params) == {"short_sizing"}           # I2b's knob, off
    assert set(comp.params) <= set(XS.params) and XS.params["short_sizing"] == "equal"
    differ = {k for k in comp.params if XS.params[k] != comp.params[k]}
    assert differ == {"long_sleeves", "short_sleeves"}
    assert (XS.fill, XS.order_notional) == (comp.fill, comp.order_notional)


def test_beta_short_sizing_resizes_only_the_shorts_at_entry_and_freezes_them(gapped):
    """I2b (RESEARCH_LOG §0.7): the same longs and the same short names as C_U0; each short
    is sized at its entry by median beta ÷ its beta (p1's 126-bar beta, known at the deciding
    close), clipped to 1/1.5 .. 1.5 of a 3% slot, frozen while held, and the short side's
    nominal gross never exceeds 0.45."""
    base = _run(gapped)["weights"]
    got = _run(gapped, short_sizing="beta")
    w = got["weights"]
    pd.testing.assert_frame_equal(w.clip(lower=0), base.clip(lower=0))     # longs untouched
    pd.testing.assert_frame_equal(w < 0, base < 0)                         # the same shorts
    sizes = -w[w < 0].stack()
    assert sizes.min() > 0 and sizes.max() <= 0.03 * 1.5 + 1e-12
    assert sizes.std() > 1e-3                                              # really re-sized
    short_gross = -w.clip(upper=0).sum(axis=1)
    assert short_gross.max() <= 0.45 + 1e-9 and w.abs().sum(axis=1).max() <= 0.90 + 1e-9
    side = np.sign(w)
    hold = side.ne(side.shift(fill_value=0.0)).cumsum()
    for name in w.columns[(w < 0).any()]:
        held = side[name] < 0
        assert w[name][held].groupby(hold[name][held]).nunique().max() == 1   # frozen
    # entries by hand, on fill bars where the cap did not bind (the short side under 0.45):
    # median beta / the name's beta at the deciding bar (one bar before the fill), clipped
    beta = _scores(gapped)[2]["beta"]
    starts = (w < 0) & ~(w.shift(1, fill_value=0.0) < 0)
    checked = 0
    for t in w.index[starts.any(axis=1) & (short_gross < 0.45 - 1e-9)][1:]:
        b = beta.iloc[w.index.get_loc(t) - 1]
        for name in starts.columns[starts.loc[t].to_numpy()]:
            if np.isfinite(b[name]) and b[name] > 0:
                ratio = float(np.clip(b[b > 0].median() / b[name], 1 / 1.5, 1.5))
                assert -w.loc[t, name] == pytest.approx(0.03 * ratio, rel=1e-12)
                checked += 1
    assert checked > 20
    pd.testing.assert_frame_equal(got["borrow"], _run(gapped)["borrow"])
    assert XS.valid({**XS.params, "short_sizing": "beta"})
    assert not XS.valid({**XS.params, "short_sizing": "volatility"})
    assert not XS.valid({**XS.params, "short_sizing": "beta", "short_gate": "prof"})


# --- research.py ---------------------------------------------------------------------------

def _research():
    """research.py, loaded by path the way the log's commands load it."""
    path = XS.dir / "research.py"
    spec = importlib.util.spec_from_file_location("_xsl_research", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lab():
    path = XS.dir.parent / "proposals_lab" / "lab.py"
    spec = importlib.util.spec_from_file_location("_xsl_lab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _squeezed_book():
    """A squeeze in miniature, on a $100 book: A is short 45% and B long 45%; A opens bar 2 at
    30 (from 10), the cover (target 0) can spend only the locked proceeds plus an equal margin
    and buys back 1 of 4.5 shares; C's new short on the same bar, sent after the cover as the
    lab's call order does, finds no free cash and is refused (NoCashShort). A then falls."""
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
    window = Panel(close=close, open_=open_, high=close, low=close, volume=close * 0 + 1e6,
                   bar="1d")
    return pf, window, orders.ffill().fillna(0.0)


def test_a_stuck_cover_and_a_refused_short_are_booked_from_their_due_open():
    """Each gap is the position minus the shares the order ASKED for, held from the due bar's
    OPEN (xs_si_mom_composite's review fix: its first version started at the due close)."""
    rs = _research()
    pf, window, _ = _squeezed_book()
    assert np.allclose(pf.value().to_numpy(), [100.0, 100.0, 27.5, 45.0, 52.0])
    assert np.allclose(pf.assets()["A"], [-4.5, -4.5, -3.5, -3.5, -3.5])      # 1 of 4.5 covered
    assert (pf.assets()["C"] == 0).all()                                       # short refused
    rows, gap = rs._gaps(pf, window, 100.0)
    by = {r["name"]: r for r in rows}
    assert by["A"]["kind"] == "cover" and by["A"]["wanted"] == 4.5 and by["A"]["got"] == 1.0
    assert by["C"]["kind"] == "new short" and by["C"]["got"] == 0.0
    assert by["C"]["status info"] == 6                                         # NoCashShort
    assert by["A"]["gap shares"] == -3.5 and by["C"]["gap shares"] == 0.2
    a = -3.5 * np.array([0.0, 0.0, (25 - 30) / 100, (20 - 25) / 27.5, (18 - 20) / 45])
    c = 0.2 * np.array([0.0, 0.0, (10 - 10) / 100, (9 - 10) / 27.5, (8 - 9) / 45])
    np.testing.assert_allclose(gap.to_numpy(), a + c, atol=1e-12)
    assert by["A"]["bars stuck"] == by["C"]["bars stuck"] == 3
    moved = rs.gap_pnl(2.0, [1, 2, 3, 4, 5, 6], [1.5, 2.5, 3.5, 4.5, 5.5, 6.5], np.ones(6),
                       at=1, nxt=4)
    np.testing.assert_allclose(moved, [0.0, 1.0, 2.0, 2.0, 1.0, 0.0])         # open[4] - open[1]


def test_off_target_counts_both_a_refused_entry_and_a_stuck_exit():
    rs = _research()
    pf, window, weights = _squeezed_book()
    off = rs.off_target(pf.assets(), weights, window.close)
    assert off.sum().to_dict() == {"A": 3, "B": 0, "C": 3}
    hidden = window.close.copy()
    hidden.iloc[3:, 0] = np.nan                            # unpriced name-days are not counted
    assert int(rs.off_target(pf.assets(), weights, hidden)["A"].sum()) == 1


def test_the_preview_is_the_lab_runner_without_costs_and_flags_a_failed_order():
    """The research preview replays the targets outside vectorbt. Without costs it equals
    lab.run_portfolio (retrim cap included); the squeezed book's stuck cover and refused short
    are exactly the difference, and an order due at a missing open waits for the next one."""
    rs, lab = _research(), _lab()
    rng = np.random.default_rng(3)
    n, m = 120, 12
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    cols = [f"S{i}" for i in range(m)]
    close = pd.DataFrame(50 * np.cumprod(1 + rng.normal(0.001, 0.03, (n, m)), axis=0),
                         index=idx, columns=cols)
    close["S0"] = 20 * np.cumprod(1 + np.r_[np.zeros(10), np.full(n - 10, 0.02)])  # a runaway
    open_ = close.shift(1) * (1 + rng.normal(0, 0.01, (n, m)))
    open_.iloc[0] = close.iloc[0]
    open_.iloc[40:43, 5] = np.nan                         # a held short without an open
    w = pd.DataFrame(0.0, index=idx, columns=cols)
    w.iloc[:, :3], w.iloc[:, 3:6] = 0.1, -0.1
    w.iloc[60:, 4], w.iloc[60:, 7], w.iloc[80:, 1], w.iloc[80:, 8] = 0.0, -0.1, 0.0, 0.1
    p = Panel(close=close, open_=open_, high=close, low=close, volume=close * 0 + 1e6, bar="1d")
    for k in (None, 2.0):
        pf = lab.run_portfolio(p, w, price=open_, fees=0.0, slippage=0.0, init_cash=1e4,
                               retrim_k=k)
        v = pf.value()
        engine = v / v.shift(1).fillna(1e4) - 1.0
        np.testing.assert_allclose(engine, rs.preview(p, w, k), atol=1e-12)
    assert (rs.preview(p, w, 2.0) - rs.preview(p, w, None)).abs().max() > 1e-3   # trims bind
    pf, window, weights = _squeezed_book()
    v = pf.value()
    engine = v / v.shift(1).fillna(100.0) - 1.0
    want = rs.preview(window, weights)
    assert (engine - want).abs().iloc[:2].max() < 1e-12
    assert (engine - want).abs().iloc[2:].max() > 0.05     # the failed orders show up ...
    late = w.copy()
    late.iloc[41:, 5] = 0.0                                # a cover due at a missing open
    moved = w.copy()
    moved.iloc[43:, 5] = 0.0                               # ... fills at the next priced open
    np.testing.assert_allclose(rs.preview(p, late), rs.preview(p, moved), atol=1e-12)


def test_research_refuses_the_locked_tests():
    rs = _research()
    for fn in (lambda: rs.audit(("C_U0",), ("T1", "TL")), lambda: rs.stuck(("C_U0",), ("T4",)),
               lambda: rs.delta("C_U0", "C_rev", ("T4",)), lambda: rs.legs("C_U0", ("TL",))):
        with pytest.raises(SystemExit, match="locked"):
            fn()
