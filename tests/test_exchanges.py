"""Exchange resolution and regular-trading-hours tagging."""

import pandas as pd
import pytest

from trading_simple.exchanges import REGISTRY, is_rth, resolve_exchange


def test_resolution_priority():
    assert resolve_exchange("AAPL").code == "US"           # default
    assert resolve_exchange("NESN").code == "SIX"          # built-in Swiss list
    assert resolve_exchange("nesn").code == "SIX"          # case-insensitive
    assert resolve_exchange("AAPL", {"aapl": "SIX"}).code == "SIX"  # override wins


def test_unknown_exchange_code_raises():
    with pytest.raises(KeyError, match="Unknown exchange"):
        resolve_exchange("AAPL", {"AAPL": "NOPE"})


def test_daily_bars_are_always_rth():
    ts = pd.Timestamp("2026-01-05 02:00", tz="UTC")  # deep night in New York
    assert is_rth(ts, REGISTRY["US"], "1d") is True


def test_us_session_boundaries_dst_safe():
    us = REGISTRY["US"]
    # July (EDT, UTC-4): 09:30 New York == 13:30 UTC
    assert is_rth(pd.Timestamp("2026-07-06 13:30", tz="UTC"), us, "1m")
    assert not is_rth(pd.Timestamp("2026-07-06 13:29", tz="UTC"), us, "1m")
    # January (EST, UTC-5): 09:30 New York == 14:30 UTC
    assert is_rth(pd.Timestamp("2026-01-05 14:30", tz="UTC"), us, "1m")
    assert not is_rth(pd.Timestamp("2026-01-05 14:29", tz="UTC"), us, "1m")
    # Close is exclusive: 16:00 EDT == 20:00 UTC is post-market
    assert not is_rth(pd.Timestamp("2026-07-06 20:00", tz="UTC"), us, "1m")


def test_six_session_and_suffix():
    six = REGISTRY["SIX"]
    assert six.yf_suffix == ".SW" and six.currency == "CHF"
    # 09:00 Zurich in July == 07:00 UTC
    assert is_rth(pd.Timestamp("2026-07-06 07:00", tz="UTC"), six, "1m")
    assert not is_rth(pd.Timestamp("2026-07-06 06:59", tz="UTC"), six, "1m")
