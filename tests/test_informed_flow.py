"""informed_flow signal logic + feature alignment on synthetic data (no DB, no network)."""

import numpy as np
import pandas as pd

from trading_simple import features as F
from trading_simple import strategies
from trading_simple.data import Panel

_PARAMS = {"insider_window": 10, "rating_halflife": 5, "w_insider": 0.5,
           "w_rating": 0.3, "w_dtc": 0.2, "q_in": 0.8, "q_out": 0.6,
           "max_hold": 40, "min_events": 1}


def _if(panel, **overrides):
    strat = strategies.load("informed_flow")
    return strat.signals(panel, **{**strat.params, **_PARAMS, **overrides})


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
    assert not sig["entries"].iloc[30].any()
    assert list(np.where(sig["entries"]["AAA"].to_numpy())[0])[0] == 31
    # Names with no alt-data activity are ineligible and never enter.
    assert not sig["entries"]["CCC"].any()


def test_eligibility_mask_blocks_thin_names():
    p = _panel()
    p.features["insider"].iloc[30, 0] = 5_000_000.0  # exactly ONE AAA event
    # With min_events=2, AAA is ineligible despite ranking top -> no entries at all.
    sig = _if(p, min_events=2)
    assert sig["entries"].to_numpy().sum() == 0
    # With min_events=1 it trades.
    assert _if(p, min_events=1)["entries"].to_numpy().sum() > 0


def test_hysteresis_exit_when_rank_drops():
    p = _panel(symbols=("AAA", "BBB"))
    p.features["insider"].iloc[30, 0] = 5_000_000.0   # AAA leads from bar 30
    p.features["insider"].iloc[40, 1] = 50_000_000.0  # BBB overtakes at bar 40
    sig = _if(p, max_hold=200)  # disable the time-stop so only the band can exit
    assert list(np.where(sig["entries"]["AAA"].to_numpy())[0])[0] == 31
    # AAA's rank falls below q_out once BBB dominates -> a band exit fires (~bar 41).
    aaa_exits = np.where(sig["exits"]["AAA"].to_numpy())[0]
    assert aaa_exits.size and 40 <= aaa_exits[0] <= 43


def test_max_hold_backstop():
    p = _panel(symbols=("AAA", "BBB"))
    # AAA buys every day so its rank never drops -> only the time-stop can exit.
    p.features["insider"].iloc[30:60, 0] = 1_000_000.0
    sig = _if(p, max_hold=5)
    first_entry = np.where(sig["entries"]["AAA"].to_numpy())[0][0]
    assert sig["exits"]["AAA"].to_numpy()[first_entry + 5]  # exit_after fires at entry+max_hold


def test_missing_short_interest_degrades_gracefully():
    p = _panel()
    p.features["insider"].iloc[30, 0] = 5_000_000.0
    p.features["short_interest"] = pd.DataFrame(np.nan, index=p.close.index, columns=p.symbols)
    # An all-NaN feature (never ingested) is dropped; the strategy still trades.
    sig = _if(p, w_dtc=0.9)
    assert sig["entries"].to_numpy().sum() > 0


def test_valid_rules():
    strat = strategies.load("informed_flow")
    assert strat.valid({"q_in": 0.8, "q_out": 0.6, "w_insider": 0.4,
                        "w_rating": 0.3, "w_dtc": 0.3})
    assert not strat.valid({"q_in": 0.6, "q_out": 0.8})               # band inverted
    assert not strat.valid({"q_in": 0.8, "q_out": 0.6, "w_insider": 0.0,
                            "w_rating": 0.0, "w_dtc": 0.0})            # no weight


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
