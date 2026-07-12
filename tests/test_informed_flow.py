"""informed_flow signal logic + feature alignment on synthetic data (no DB, no network)."""

import numpy as np
import pandas as pd
import pytest

from trading_simple import features as F
from trading_simple import strategies
from trading_simple.data import Panel

_PARAMS = {"insider_window": 10, "rating_halflife": 5, "ptrev_halflife": 5,
           "w_insider": 0.5, "w_rating": 0.3, "w_ptrev": 0.0, "w_dtc": 0.2,
           "q_in": 0.8, "q_out": 0.6, "max_hold": 40, "min_events": 1,
           # Enough slots that the panel's whole universe fits, so tests about *ranking*
           # are not silently also testing the capital constraint. The tests that DO
           # exercise the slot limit set it themselves.
           "max_positions": 5, "displace": False}


def _if(panel, **overrides):
    strat = strategies.load("informed_flow")
    return strat.signals(panel, **{**strat.params, **_PARAMS, **overrides})


def _module():
    """The informed_flow strategy module (the loader binds only signals/valid)."""
    return strategies.load("informed_flow").signals.__globals__


def _explain(panel, symbol, **overrides):
    return _module()["explain"](panel, symbol, **{**_PARAMS, **overrides})


def _bought(sig, sym) -> np.ndarray:
    """Bars where the book starts wanting a name (target weight 0 -> positive)."""
    weight = sig["weights"][sym]
    return np.where((weight > 0) & (weight.shift(1).fillna(0.0) == 0))[0]


def _sold(sig, sym) -> np.ndarray:
    """Bars where the book stops wanting a name (target weight positive -> 0)."""
    weight = sig["weights"][sym]
    return np.where((weight == 0) & (weight.shift(1).fillna(0.0) > 0))[0]


def _panel(symbols=("AAA", "BBB", "CCC", "DDD"), n=90):
    """A daily panel with flat prices and empty feature frames to fill per test."""
    index = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    close = pd.DataFrame(100.0, index=index, columns=list(symbols))
    volume = pd.DataFrame(1_000_000.0, index=index, columns=list(symbols))
    panel = Panel(close=close, open_=close * 0.999, high=close * 1.01,
                  low=close * 0.99, volume=volume, bar="1d")
    zero = pd.DataFrame(0.0, index=index, columns=list(symbols))
    panel.features = {"insider": zero.copy(), "ratings": zero.copy()}
    return panel


def test_insider_cluster_enters_next_bar():
    p = _panel()
    p.features["insider"].iloc[30, 0] = 5_000_000.0  # a big AAA buy at bar 30
    sig = _if(p)
    # Signal at bar 30 -> execution at bar 31 (next-bar open), never same bar.
    assert sig["weights"].iloc[30].sum() == 0
    assert _bought(sig, "AAA")[0] == 31
    # Names with no alt-data activity are ineligible and are never bought.
    assert not (sig["weights"]["CCC"] > 0).any()


def test_eligibility_mask_blocks_thin_names():
    p = _panel()
    p.features["insider"].iloc[30, 0] = 5_000_000.0  # exactly ONE AAA event
    # With min_events=2, AAA is ineligible despite ranking top -> the book stays empty.
    assert _if(p, min_events=2)["weights"].to_numpy().sum() == 0
    # With min_events=1 it trades.
    assert _if(p, min_events=1)["weights"].to_numpy().sum() > 0


def test_ineligible_names_leave_the_ranked_universe():
    """A name with too little alt-data is DROPPED from the cross-section, not just blocked.

    Ranking it and then refusing to buy it would be the worst of both worlds: it could
    never be held, yet it would still occupy rank space and shift everyone else's
    percentile. So its score is NaN, and the eligible names are ranked among themselves.
    """
    p = _panel()  # AAA, BBB, CCC, DDD
    p.features["insider"].iloc[30, 0] = 5_000_000.0   # AAA: an event -> eligible
    p.features["insider"].iloc[30, 1] = 1_000_000.0   # BBB: smaller -> eligible, ranks below
    parts = _module()["_compose"](p, **_PARAMS)       # CCC, DDD: no events -> ineligible
    bar = p.close.index[31]

    assert parts["eligible"].loc[bar].tolist() == [True, True, False, False]
    assert parts["n_ranked"].loc[bar] == 2            # ranked against each other only
    score = parts["score_pct"].loc[bar]
    assert score["AAA"] == 1.0 and score["BBB"] == 0.5
    assert score[["CCC", "DDD"]].isna().all()         # no rank at all — not a neutral 0.5


def test_held_name_that_becomes_ineligible_is_exited():
    """If a name falls out of the ranked universe, close it — do not carry it unscored."""
    p = _panel(symbols=("AAA", "BBB"))
    p.features["insider"].iloc[30, 0] = 5_000_000.0   # AAA's only event
    p.features["insider"].iloc[30, 1] = 1_000_000.0   # keeps BBB in the universe too
    p.features["ratings"].iloc[30:, 1] = 1.0          # BBB stays eligible; AAA does not
    sig = _if(p, max_hold=200)                        # rule out the time-stop
    # AAA's event ages out of the 10-bar window at bar 40 -> ineligible -> sold.
    assert _bought(sig, "AAA").size
    assert 40 <= _sold(sig, "AAA")[0] <= 43


def test_hysteresis_exit_when_rank_drops():
    p = _panel(symbols=("AAA", "BBB"))
    p.features["insider"].iloc[30, 0] = 5_000_000.0   # AAA leads from bar 30
    p.features["insider"].iloc[40, 1] = 50_000_000.0  # BBB overtakes at bar 40
    sig = _if(p, max_hold=200)  # disable the time-stop so only the band can sell
    assert _bought(sig, "AAA")[0] == 31
    # AAA's rank falls away once BBB dominates -> it is sold (~bar 41).
    assert 40 <= _sold(sig, "AAA")[0] <= 43


def _band_panel(n=90):
    """5 names with constant rating streams -> stable, distinct cross-sectional ranks.

    AAA leads (so it enters), then from bar 40 drops to the MIDDLE of the pack: rank
    0.6, which is >= q_out (0.6) but < q_in (0.8) — parked in the hysteresis band.
    """
    syms = ("AAA", "BBB", "CCC", "DDD", "EEE")
    index = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    close = pd.DataFrame(100.0, index=index, columns=list(syms))
    volume = pd.DataFrame(1_000_000.0, index=index, columns=list(syms))
    panel = Panel(close=close, open_=close * 0.999, high=close * 1.01,
                  low=close * 0.99, volume=volume, bar="1d")
    rat = pd.DataFrame({"AAA": 5.0, "BBB": 4.0, "CCC": 3.0, "DDD": 2.0, "EEE": 1.0},
                       index=index)
    rat.loc[index[40:], "AAA"] = 2.5      # 3rd of 5 -> rank 0.6 -> the band
    panel.features = {"ratings": rat}
    return panel


def test_max_hold_evicts_a_name_parked_in_the_band():
    """The time-stop's one real job: a name aging in the hysteresis band.

    It is too weak to re-qualify (never back at/above q_in) and not weak enough to trip
    q_out, so nothing else can ever sell it. Only max_hold can.
    """
    p = _band_panel()
    sig = _if(p, w_insider=0.0, w_rating=1.0, w_dtc=0.0, rating_halflife=2, max_hold=10)
    assert _bought(sig, "AAA")[0] == 1               # AAA leads from the start
    # AAA's rank sinks into the band from ~bar 40, so the clock stops resetting and the
    # time-stop fires ~10 bars later.
    sold = _sold(sig, "AAA")
    assert sold.size and 42 <= sold[0] <= 60


def test_time_stop_cannot_evict_a_name_that_is_re_qualifying():
    """A name back in the top quantile has a FRESH thesis; an old clock must not evict it.

    AAA's rating stream never fades, so it is at/above q_in on every bar, its entry clock
    resets every bar, and the time-stop can never mature. This is not a nicety: without
    the reset the timer would evict AAA and the very next rule would buy it straight
    back, paying two fees for a book that did not change — a pure wash trade.
    """
    p = _band_panel()
    p.features["ratings"].loc[:, "AAA"] = 5.0        # never drops out of the top
    sig = _if(p, w_insider=0.0, w_rating=1.0, w_dtc=0.0, max_hold=5)
    assert _bought(sig, "AAA").size
    assert not _sold(sig, "AAA").size                # no rank exit, no time-stop, no wash


def test_the_book_is_always_a_valid_portfolio():
    """Structural invariant: never oversubscribed, never levered, always whole slots.

    Every weight is either zero or exactly one slot (1/max_positions), no more than
    max_positions names are held at once, and the row never sums above 1.0 — the book
    cannot spend money it does not have.
    """
    p = _panel()
    p.features["insider"].iloc[30, 0] = 5_000_000.0
    p.features["insider"].iloc[60, 0] = 7_000_000.0
    p.features["ratings"].iloc[35, 1] = 1.0
    for max_positions in (1, 2, 4):
        weights = _if(p, max_positions=max_positions)["weights"]
        slot = 1.0 / max_positions
        held = (weights > 0).sum(axis=1)
        assert held.max() <= max_positions
        assert weights.sum(axis=1).max() <= 1.0 + 1e-12
        nonzero = weights.to_numpy()[weights.to_numpy() > 0]
        assert np.allclose(nonzero, slot)


def test_missing_short_interest_degrades_gracefully():
    p = _panel()
    p.features["insider"].iloc[30, 0] = 5_000_000.0
    p.features["short_interest"] = pd.DataFrame(np.nan, index=p.close.index, columns=p.symbols)
    # An all-NaN feature (never ingested) is dropped; the strategy still trades.
    assert _if(p, w_dtc=0.9)["weights"].to_numpy().sum() > 0


def test_valid_rules():
    strat = strategies.load("informed_flow")
    assert strat.valid({"q_in": 0.8, "q_out": 0.6, "w_insider": 0.4,
                        "w_rating": 0.3, "w_dtc": 0.3})
    assert not strat.valid({"q_in": 0.6, "q_out": 0.8})               # band inverted
    assert not strat.valid({"q_in": 0.8, "q_out": 0.6, "w_insider": 0.0,
                            "w_rating": 0.0, "w_dtc": 0.0})            # no weight
    assert not strat.valid({"q_in": 0.8, "q_out": 0.6, "w_insider": 0.4,
                            "max_positions": 0})                       # a book with no slots


# --- the capital constraint: slots, waiting, and displacement -----------------
_DISPLACE = {"w_insider": 0.0, "w_rating": 1.0, "w_dtc": 0.0, "rating_halflife": 2,
             "q_in": 0.6, "q_out": 0.4, "max_hold": 200, "max_positions": 2}
#: `_band_panel` starts AAA 5, BBB 4, CCC 3, DDD 2, EEE 1 and drops AAA to 2.5 at bar 40.
#: Sending EEE to 10 there leaves a late ranking of EEE 1.0, BBB 0.8, CCC 0.6, AAA 0.4,
#: DDD 0.2. So the book (AAA, BBB) is full, EEE outranks everything, and **AAA is the
#: weakest holding** — clinging on at exactly q_out. Whether EEE gets in is entirely a
#: question of `displace`.


def test_a_qualified_name_waits_for_a_vacancy():
    """With displace off, the best name in the market cannot buy in — the money is spent.

    This state simply did not exist under per-symbol cash accounts, where every name had
    its own money and a qualifying name was ALWAYS bought.
    """
    p = _band_panel()
    p.features["ratings"].loc[p.close.index[40:], "EEE"] = 10.0   # EEE storms to the top
    late = _if(p, **_DISPLACE, displace=False)["weights"].iloc[70]
    assert late["AAA"] > 0 and late["BBB"] > 0         # both original holdings keep their slots
    assert late["EEE"] == 0                            # the best name in the market: unbought

    exp = _explain(p, "EEE", **_DISPLACE, displace=False)
    bar = exp.index[69]                                # the signal bar behind weights[70]
    assert exp.loc[bar, "qualified"] and exp.loc[bar, "slot_blocked"]
    assert exp.loc[bar, "state"] == "slot_blocked"     # and the diagnostic says exactly why


def test_displace_evicts_the_weakest_holding():
    """With displace on, the same newcomer takes the WEAKEST holding's slot — not any slot."""
    p = _band_panel()
    p.features["ratings"].loc[p.close.index[40:], "EEE"] = 10.0
    late = _if(p, **_DISPLACE, displace=True)["weights"].iloc[70]
    assert late["EEE"] > 0                             # EEE is in...
    assert late["AAA"] == 0                            # ...and AAA, the weaker holding, is out
    assert late["BBB"] > 0                             # the stronger holding is untouched
    assert (late > 0).sum() == 2                       # still exactly two slots


def test_thin_selection_leaves_cash_rather_than_concentrating():
    """One name qualifying does not mean betting the whole book on it.

    A slot is a FIXED 1/max_positions. If only one of ten slots can be filled, the book is
    10% invested and 90% cash — it does not quietly become a 100% single-stock position.
    """
    p = _panel()
    p.features["insider"].iloc[30, 0] = 5_000_000.0    # only AAA has anything to say
    weights = _if(p, max_positions=10)["weights"]
    invested = weights.sum(axis=1)
    assert weights["AAA"].max() == pytest.approx(0.1)
    assert invested.max() == pytest.approx(0.1)        # 10% invested, 90% in cash


def test_low_coverage_sleeve_is_dropped():
    """A sleeve covering a handful of names must not become a tilt on those names.

    Regression test for a real bug: short_interest shipped covering 3 of 100
    symbols. `notna().any().any()` said "live", so the 3 got real percentile ranks
    while the other 97 took the neutral .fillna(0.5) — a large arbitrary shove on
    three names and pure noise for everyone else.
    """
    p = _panel()  # AAA, BBB, CCC, DDD
    p.features["insider"].iloc[30, 0] = 5_000_000.0
    dtc = pd.DataFrame(np.nan, index=p.close.index, columns=p.symbols)
    dtc["AAA"] = 1.0  # only 1 of 4 names covered = 25%, below the 50% floor
    p.features["short_interest"] = dtc
    parts = _module()["_compose"](p, **{**_PARAMS, "w_dtc": 0.9})
    assert "short_interest" not in parts["present"]
    # With enough coverage it is kept, and its weight bites.
    dtc["BBB"] = 2.0
    dtc["CCC"] = 3.0
    parts = _module()["_compose"](p, **{**_PARAMS, "w_dtc": 0.9})
    assert "short_interest" in parts["present"]


def test_explain_reproduces_signals():
    """The diagnostic must report exactly the book the backtest traded."""
    p = _panel()
    p.features["insider"].iloc[30, 0] = 5_000_000.0
    p.features["insider"].iloc[50, 1] = 9_000_000.0
    sig = _if(p)
    exp = _explain(p, "AAA")
    assert exp["target_weight"].equals(sig["weights"]["AAA"])


def test_contributions_sum_to_composite():
    p = _panel()
    p.features["insider"].iloc[30, 0] = 5_000_000.0
    exp = _explain(p, "AAA")
    contrib = exp[[c for c in exp.columns if c.startswith("contrib_")]].sum(axis=1)
    assert (contrib - exp["composite"]).abs().max() < 1e-12


def test_own_peer_split_is_exact():
    """own + peer must telescope to the rank change with no residual."""
    p = _panel()
    p.features["insider"].iloc[30, 0] = 5_000_000.0
    p.features["insider"].iloc[40, 1] = 9_000_000.0
    exp = _explain(p, "AAA").dropna(subset=["d_score"])
    assert (exp["own"] + exp["peer"] - exp["d_score"]).abs().max() < 1e-12


def test_peer_drift_moves_a_name_with_no_event():
    """A name can lose its rank on a bar where nothing at all happened to it.

    AAA buys once at bar 30 and never again. BBB buys hugely at bar 35, while AAA's
    event is still inside the 10-bar trailing window (so AAA's own raw value is
    unchanged: same trailing sum, and flat prices/volume keep the dollar-volume
    denominator constant). AAA's rank still falls. The whole move must be attributed
    to `peer`, and `own_event` must be False: nothing happened to AAA.
    """
    p = _panel(symbols=("AAA", "BBB"))
    p.features["insider"].iloc[30, 0] = 5_000_000.0
    p.features["insider"].iloc[35, 1] = 50_000_000.0
    exp = _explain(p, "AAA", max_hold=200)
    bar = exp.index[35]
    assert exp.loc[bar, "d_score"] < 0                 # AAA's rank fell
    assert abs(exp.loc[bar, "own"]) < 1e-12            # its own value did not move
    assert exp.loc[bar, "peer"] == exp.loc[bar, "d_score"]  # peers did all of it
    assert exp.loc[bar, "driver"] == "peer"
    assert not exp.loc[bar, "own_event"]               # and nothing happened to AAA


def test_neutral_fill_is_flagged():
    """An ELIGIBLE name missing ONE sleeve gets the neutral 0.5 there — and must say so.

    (A name missing alt-data altogether is now dropped from the ranking outright, so
    the neutral fill only ever applies to a ranked name with a hole in one sleeve —
    e.g. it has insider/rating events but no short-interest coverage.)
    """
    p = _panel()
    for col in range(4):                                # every name is eligible
        p.features["insider"].iloc[30, col] = 1_000_000.0 * (col + 1)
    dtc = pd.DataFrame(1.0, index=p.close.index, columns=p.symbols)
    dtc["DDD"] = np.nan                                 # 3/4 covered -> sleeve survives
    p.features["short_interest"] = dtc
    exp = _explain(p, "DDD", w_dtc=0.5)
    ranked = exp[exp["eligible"]]                       # the bars DDD is in the universe
    assert len(ranked)

    assert ranked["filled_short_interest"].all()        # flagged on every ranked bar
    assert ranked["rank_short_interest"].isna().all()   # no rank -> it was imputed
    contrib = ranked["contrib_short_interest"]
    assert np.allclose(contrib, contrib.iloc[0])        # a flat, information-free 0.5


# --- features._to_grid (pure alignment + PIT lag) ---------------------------
def test_to_grid_flow_shifts_one_bar_and_zero_fills():
    index = pd.date_range("2024-01-02", periods=6, freq="B", tz="UTC")
    long = pd.DataFrame({"symbol": ["AAA"], "d": [pd.Timestamp("2024-01-03").date()],
                         "value": [7.0]})
    grid = F._to_grid(long, index, fill="zero")
    assert grid.index.equals(index)
    # Event on 01-03 (bar 1) is only knowable at the NEXT bar 01-04 (bar 2).
    assert grid["AAA"].iloc[2] == 7.0
    assert grid["AAA"].iloc[3] == 0.0                 # no-event day = 0, not NaN
    assert np.isnan(grid["AAA"].iloc[0])              # shift boundary


def test_to_grid_level_forward_fills():
    index = pd.date_range("2024-01-02", periods=6, freq="B", tz="UTC")
    long = pd.DataFrame({"symbol": ["AAA"], "d": [pd.Timestamp("2024-01-03").date()],
                         "value": [3.5]})
    grid = F._to_grid(long, index, fill="ffill")
    assert grid["AAA"].iloc[2] == 3.5 and grid["AAA"].iloc[4] == 3.5   # level persists
    assert np.isnan(grid["AAA"].iloc[1])              # unknown before first observation


def test_to_grid_empty_is_index_only():
    index = pd.date_range("2024-01-02", periods=4, freq="B", tz="UTC")
    grid = F._to_grid(pd.DataFrame(columns=["symbol", "d", "value"]), index, fill="zero")
    assert grid.index.equals(index) and grid.shape[1] == 0
