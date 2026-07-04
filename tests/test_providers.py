"""Pure provider parts: parsing, request planning, pacing — no network."""

from datetime import date

import pandas as pd
import pytest

from trading_simple.exchanges import REGISTRY
from trading_simple.providers import ibkr, yf


# --- yfinance ---------------------------------------------------------------
def _yf_frame(adj: bool = True) -> pd.DataFrame:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])  # naive, like yfinance daily
    data = {"Open": [10.0, 11.0], "High": [10.5, 11.5], "Low": [9.5, 10.5],
            "Close": [10.2, 11.2], "Volume": [1000.0, float("nan")]}
    if adj:
        data["Adj Close"] = [5.1, 5.6]  # a 2:1 split happened later
    return pd.DataFrame(data, index=idx)


def test_yf_frame_to_bars_utc_and_adj():
    bars = yf._frame_to_bars("AAPL", _yf_frame())
    assert [b.ts.isoformat() for b in bars] == ["2024-01-02T00:00:00+00:00",
                                                "2024-01-03T00:00:00+00:00"]
    assert bars[0].adj_close == 5.1 and bars[0].close == 10.2
    assert bars[1].volume is None            # NaN -> None
    assert all(b.rth for b in bars)          # daily = regular candle


def test_yf_adj_close_falls_back_to_close():
    bars = yf._frame_to_bars("AAPL", _yf_frame(adj=False))
    assert bars[0].adj_close == bars[0].close == 10.2


# --- IBKR: request planning -------------------------------------------------
def test_request_windows_daily_single_request():
    ((end_dt, duration),) = ibkr.request_windows("1d", date(2024, 1, 1), date(2024, 3, 1))
    assert duration == "61 D" and end_dt.hour == 23


def test_request_windows_daily_caps_to_years():
    # 9 years + 1 day = 3288 days -> ceil(3288/365) = 10 years requested.
    ((_, duration),) = ibkr.request_windows("1d", date(2015, 1, 1), date(2024, 1, 1))
    assert duration == "10 Y"


def test_request_windows_1m_pages_backwards_in_10d_steps():
    windows = ibkr.request_windows("1m", date(2024, 1, 1), date(2024, 1, 25))
    assert [d for _, d in windows] == ["10 D", "10 D", "5 D"]
    ends = [e.date() for e, _ in windows]
    assert ends == sorted(ends, reverse=True)  # newest first


def test_request_windows_rejects_inverted_range():
    with pytest.raises(ValueError, match="on or before"):
        ibkr.request_windows("1m", date(2024, 2, 1), date(2024, 1, 1))


# --- IBKR: pacer ------------------------------------------------------------
def test_pacer_only_blocks_at_the_cap():
    pacer = ibkr.Pacer(max_per_window=3, window_s=600, min_gap_s=0)
    clock, sleeps = iter(range(100)).__next__, []
    for _ in range(3):
        pacer.wait(clock=lambda: float(clock()), sleep=sleeps.append)
    assert sleeps == []                       # under the cap: full speed
    pacer.wait(clock=lambda: float(clock()), sleep=sleeps.append)
    assert len(sleeps) == 1 and sleeps[0] > 500  # 4th must wait for the window


def test_pacer_enforces_min_gap():
    pacer = ibkr.Pacer(max_per_window=100, window_s=600, min_gap_s=0.5)
    now, sleeps = [0.0], []
    pacer.wait(clock=lambda: now[0], sleep=sleeps.append)
    now[0] = 0.1                              # only 100ms later
    pacer.wait(clock=lambda: now[0], sleep=sleeps.append)
    assert sleeps and abs(sleeps[0] - 0.4) < 1e-9


# --- IBKR: contracts + parsing ----------------------------------------------
def test_contract_args_by_venue():
    assert ibkr.contract_args("AAPL", REGISTRY["US"]) == ("AAPL", "SMART", "USD")
    assert ibkr.contract_args("NESN", REGISTRY["SIX"]) == ("NESN", "EBS", "CHF")


def test_ib_frame_to_bars_rth_tagging_and_sentinels():
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2026-07-06 13:30", tz="UTC"),   # 09:30 NY = regular
                 pd.Timestamp("2026-07-06 20:05", tz="UTC")],  # 16:05 NY = extended
        "open": [10.0, 10.5], "high": [10.6, 10.7], "low": [9.9, 10.4],
        "close": [10.5, 10.6], "volume": [500.0, -1.0],
    })
    bars = ibkr.frame_to_bars("AAPL", "1m", frame, REGISTRY["US"])
    assert [b.rth for b in bars] == [True, False]
    assert bars[0].adj_close == bars[0].close   # IBKR has no adjusted close
    assert bars[1].volume is None               # -1 sentinel -> None
