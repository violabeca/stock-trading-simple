"""conviction_flow selectivity levers on synthetic data (no DB, no network).

conviction_flow is informed_flow plus five open/close levers, each OFF by default. The
load-bearing test is :func:`test_off_defaults_match_informed_flow` — at default params the
two strategies must produce byte-identical weights, so every lever is a pure opt-in.
"""

import numpy as np
import pandas as pd

from trading_simple import strategies
from trading_simple.data import Panel

#: Flow params shared across tests (fundamentals levers overridden per test).
_PARAMS = {"insider_window": 10, "rating_halflife": 3, "ptrev_halflife": 5,
           "w_insider": 0.5, "w_rating": 0.3, "w_ptrev": 0.0, "w_dtc": 0.2,
           "q_in": 0.8, "q_out": 0.6, "max_hold": 40, "min_events": 1,
           "max_positions": 5, "displace": False}


def _cf(panel, **overrides):
    strat = strategies.load("conviction_flow")
    return strat.signals(panel, **{**strat.params, **_PARAMS, **overrides})


def _held(panel, **overrides):
    """Set of symbols the book wants to own at some point, from the weight frame."""
    w = _cf(panel, **overrides)["weights"]
    return set(w.columns[(w > 0).any()])


def _panel(symbols=("AAA", "BBB", "CCC", "DDD"), n=90, fundamentals=False):
    """A daily panel with flat prices and empty flow features to fill per test.

    When ``fundamentals`` is set, attaches neutral (equal) ``fund_*`` matrices so the
    quality gate is *live but flat* — every name gets the same quality, so the gate never
    changes selection until a test perturbs the fundamentals.
    """
    index = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    close = pd.DataFrame(100.0, index=index, columns=list(symbols))
    volume = pd.DataFrame(1_000_000.0, index=index, columns=list(symbols))
    panel = Panel(close=close, open_=close * 0.999, high=close * 1.01,
                  low=close * 0.99, volume=volume, bar="1d")
    zero = pd.DataFrame(0.0, index=index, columns=list(symbols))
    panel.features = {"insider": zero.copy(), "ratings": zero.copy()}
    if fundamentals:
        one = pd.DataFrame(1.0, index=index, columns=list(symbols))
        panel.features.update(fund_shares=one.copy(), fund_equity=one.copy(),
                              fund_ni_ttm=one.copy(), fund_gp_ttm=one.copy(),
                              fund_assets=one.copy())
    return panel


def _rank_panel(n=90):
    """5 names with stable, distinct rating levels -> stable, distinct flow ranks.

    Rating levels AAA=5 … EEE=1 give cross-sectional ranks AAA=1.0, EEE=0.8, DDD=0.6,
    CCC=0.4, BBB=0.2 (rank of 5 items in steps of 0.2). The tests perturb one name to
    exercise displacement, decay and the conviction bar with known rank gaps.
    """
    syms = ("AAA", "BBB", "CCC", "DDD", "EEE")
    index = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    close = pd.DataFrame(100.0, index=index, columns=list(syms))
    volume = pd.DataFrame(1_000_000.0, index=index, columns=list(syms))
    panel = Panel(close=close, open_=close * 0.999, high=close * 1.01,
                  low=close * 0.99, volume=volume, bar="1d")
    rat = pd.DataFrame({"AAA": 5.0, "BBB": 1.0, "CCC": 2.0, "DDD": 3.0, "EEE": 4.0},
                       index=index)
    panel.features = {"ratings": rat}
    return panel


def _ratings_only(**overrides):
    """Params isolating the single ratings sleeve (so composite == the ratings rank)."""
    return {"w_insider": 0.0, "w_rating": 1.0, "w_ptrev": 0.0, "w_dtc": 0.0,
            "rating_halflife": 2, "min_events": 1, **overrides}


# --- the load-bearing pin: at defaults, conviction_flow IS informed_flow ------
def test_off_defaults_match_informed_flow():
    """Every lever off => byte-identical weights to informed_flow, even with fund_* loaded."""
    p = _panel(fundamentals=True)
    p.features["insider"].iloc[30, 0] = 5_000_000.0
    p.features["insider"].iloc[30, 1] = 1_000_000.0
    p.features["ratings"].iloc[35, 2] = 1.0
    p.features["insider"].iloc[60, 3] = 3_000_000.0
    cf = strategies.load("conviction_flow")
    iff = strategies.load("informed_flow")
    # Only pass params informed_flow knows; conviction_flow's new levers stay at defaults.
    # TWO translations are needed, and both are deliberate divergences, not drift:
    #
    # 1. The two spell "never displace" differently: conviction_flow still carries the
    #    `displace` boolean, while informed_flow collapsed it into `displace_margin` (1.0
    #    is unreachable for a percentile rank, so it is the same instruction).
    # 2. informed_flow v21 grew a `recency` sleeve (bars since the last insider/rating
    #    event) to replace the `min_events` cliff that had become its dominant exit rule.
    #    conviction_flow is a FORK, with its own config and study journal, and has NOT
    #    been ported — so ablate the sleeve to compare like with like. Until it is ported,
    #    conviction_flow still exits on the eligibility cliff this sleeve exists to
    #    remove, and the two strategies are equivalent only with `w_recency = 0`.
    #
    # Without the translations the two would diverge for reasons that have nothing to do
    # with the fundamentals levers this test is actually pinning.
    shared = {k: v for k, v in _PARAMS.items() if k != "displace"}
    w_cf = cf.signals(p, **{**cf.params, **_PARAMS})["weights"]
    w_if = iff.signals(p, **{**iff.params, **shared,
                             "displace_margin": 1.0, "w_recency": 0.0})["weights"]
    assert w_cf.equals(w_if)


# --- fundamentals quality gate -----------------------------------------------
def test_quality_gate_vetoes_a_fundamentally_weak_name():
    """A name hot on flow but poor on fundamentals is not opened when the gate is on."""
    p = _panel(symbols=("AAA", "BBB", "CCC", "DDD", "EEE"))
    p.features["ratings"] = pd.DataFrame(
        {"AAA": 5.0, "BBB": 4.0, "CCC": 3.0, "DDD": 2.0, "EEE": 1.0}, index=p.close.index)
    # Fundamentals INVERT the flow order: AAA (top flow) has the worst book-to-price.
    one = pd.DataFrame(1e6, index=p.close.index, columns=p.symbols)
    equity = pd.DataFrame({"AAA": 1e5, "BBB": 2e5, "CCC": 3e5, "DDD": 4e5, "EEE": 5e5},
                          index=p.close.index)
    p.features.update(fund_shares=one, fund_equity=equity, fund_ni_ttm=equity.copy(),
                      fund_gp_ttm=equity.copy(), fund_assets=one.copy())
    flow = _ratings_only(q_in=0.6, q_out=0.4, max_positions=2, max_hold=200)
    # No gate: holds the top-2 by flow (AAA, BBB).
    assert _held(p, **flow) == {"AAA", "BBB"}
    # Quality floor 0.6: AAA and BBB (bottom-40% fundamentals) can't open; only CCC still
    # clears BOTH q_in (flow rank >= 0.6) and the quality floor.
    assert "AAA" not in _held(p, **flow, min_quality_pct=0.6)


def test_quality_gate_self_disables_without_fundamentals():
    """No fund_* features => the gate is inert (falls back to informed_flow behavior)."""
    p = _panel()  # alt-data only, no fundamentals
    p.features["insider"].iloc[30, 0] = 5_000_000.0
    p.features["insider"].iloc[30, 1] = 1_000_000.0
    # A high floor cannot block anything when there is no fundamentals data to judge on.
    assert _held(p, min_quality_pct=0.9) == _held(p, min_quality_pct=0.0)


# --- signal-agreement gate ---------------------------------------------------
def test_agreement_gate_requires_multiple_sleeves():
    """min_agree counts INDEPENDENT top-quantile sleeves, not the blended composite.

    AAA is top on BOTH insider and ratings (agrees 2). BBB is top on ratings but only mid
    on insider — its blended composite still clears ``q_in``, but only ONE sleeve agrees.
    So ``min_agree=2`` opens AAA and blocks BBB, even though both qualify on the composite.
    """
    p = _panel(symbols=("AAA", "BBB", "CCC"))
    # A wide insider_window keeps the single events in the trailing sum, so ranks are
    # stable. insider: AAA > CCC > BBB(=0); ratings: BBB > AAA > CCC. So AAA is top-quantile
    # on BOTH sleeves, while BBB tops only ratings (bottom on insider).
    p.features["insider"].iloc[30, 0] = 5_000_000.0
    p.features["insider"].iloc[30, 2] = 1_000_000.0
    p.features["ratings"].iloc[30:, 0] = 2.0
    p.features["ratings"].iloc[30:, 1] = 3.0
    p.features["ratings"].iloc[30:, 2] = 1.0
    both = {"insider_window": 200, "w_insider": 1.0, "w_rating": 1.0, "w_ptrev": 0.0,
            "w_dtc": 0.0, "rating_halflife": 2, "q_in": 0.6, "q_out": 0.4,
            "max_positions": 2, "max_hold": 200}
    no_gate = _held(p, **both)
    assert {"AAA", "BBB"} <= no_gate                  # both qualify on the composite
    gated = _held(p, **both, min_agree=2)
    assert "AAA" in gated and "BBB" not in gated      # only AAA has two agreeing sleeves


# --- higher conviction bar (composite level) ---------------------------------
def test_conviction_bar_blocks_a_weak_field():
    """min_score gates on the composite LEVEL: below it, a name cannot open."""
    p = _rank_panel()
    flow = _ratings_only(q_in=0.6, q_out=0.4, max_positions=5, max_hold=200)
    # Composite == ratings rank here (single sleeve), so levels are 0.2..1.0. A 0.9 floor
    # admits only the 1.0-rank name; off, the whole qualifying set (rank >= q_in) opens.
    strong = _held(p, **flow, min_score=0.9)
    assert strong == {"AAA"}                        # only the top name clears the 0.9 level
    assert len(_held(p, **flow, min_score=0.0)) > 1


# --- margin displacement -----------------------------------------------------
def test_displace_margin_requires_a_decisive_edge():
    """A challenger only 0.2 of rank better than the holding evicts it at margin 0, not 0.3."""
    p = _rank_panel()
    # AAA leads and is bought (1 slot). At bar 40 BBB storms to the top; AAA falls to the
    # 0.8 rank (gap 0.2), still above q_out so only displacement can remove it.
    p.features["ratings"].loc[p.close.index[40:], "BBB"] = 6.0
    flow = _ratings_only(q_in=0.6, q_out=0.5, max_positions=1, displace=True, max_hold=200)
    late_strict = _cf(p, **flow, displace_margin=0.0)["weights"].iloc[70]
    late_margin = _cf(p, **flow, displace_margin=0.3)["weights"].iloc[70]
    assert late_strict["BBB"] > 0 and late_strict["AAA"] == 0    # 0.2 gap > 0 margin: rotate
    assert late_margin["AAA"] > 0 and late_margin["BBB"] == 0    # 0.2 gap < 0.3 margin: hold


# --- rank-decay exit ---------------------------------------------------------
def test_decay_exit_sells_a_faded_holding():
    """A holding whose rank falls far from its peak is sold, even while still above q_out."""
    p = _rank_panel()
    # AAA enters at rank 1.0, then at bar 40 drops to the MIDDLE (rank 0.6) — a 0.4 decay,
    # but 0.6 >= q_out (0.5), so only the decay exit can remove it.
    p.features["ratings"].loc[p.close.index[40:], "AAA"] = 2.5
    flow = _ratings_only(q_in=0.8, q_out=0.5, max_positions=1, max_hold=200)
    w = _cf(p, **flow, exit_decay=0.3)["weights"]["AAA"]        # 0.4 decay > 0.3 -> sold
    sold = np.where((w == 0) & (w.shift(1).fillna(0.0) > 0))[0]
    assert sold.size and 40 <= sold[0] <= 50
    # With the decay exit off, AAA is held straight through (0.6 stays above q_out).
    assert (_cf(p, **flow, exit_decay=1.0)["weights"]["AAA"].iloc[45:] > 0).any()


# --- valid() -----------------------------------------------------------------
def test_valid_rules():
    strat = strategies.load("conviction_flow")
    base = {"q_in": 0.8, "q_out": 0.6, "w_insider": 0.4, "w_rating": 0.3, "w_dtc": 0.3,
            "max_positions": 10}
    assert strat.valid(base)                                    # informed_flow core holds
    assert strat.valid({**base, "min_quality_pct": 0.9, "min_agree": 3,
                        "min_score": 0.6, "displace_margin": 0.3, "exit_decay": 0.5})
    assert not strat.valid({**base, "min_quality_pct": 1.0})    # quantile must be < 1
    assert not strat.valid({**base, "min_score": 1.0})          # a level in [0,1)
    assert not strat.valid({**base, "exit_decay": 0.0})         # decay must be > 0
    assert not strat.valid({**base, "displace_margin": -0.1})   # margin non-negative
    assert not strat.valid({**base, "q_in": 0.6, "q_out": 0.8})  # band inverted (inherited)
