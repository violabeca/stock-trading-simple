"""Pure provider parts: parsing, request planning, pacing — no network."""

from datetime import date

import pandas as pd
import pytest

from trading_simple.config import IBKRSettings
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


# --- IBKR: reconnect on a dropped connection ---------------------------------
class _FakeIB:
    """Duck-typed stand-in for ``ib_async.IB`` — no network, no event loop."""

    def __init__(self, fail_first: int = 0):
        self.fail_first = fail_first  # how many connect() calls should raise first
        self.connect_calls = 0
        self.sleeps: list[float] = []

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)

    def disconnect(self) -> None:
        pass

    def connect(self, host, port, clientId, timeout) -> None:  # noqa: N803 - matches ib_async
        self.connect_calls += 1
        if self.connect_calls <= self.fail_first:
            raise ConnectionError("gateway not back up yet")


def _settings() -> IBKRSettings:
    return IBKRSettings(host="127.0.0.1", port=4002, client_id=10)


def test_reconnect_succeeds_on_first_attempt():
    ib = _FakeIB(fail_first=0)
    assert ibkr._reconnect(ib, _settings(), attempts=4, wait_s=0) is True
    assert ib.connect_calls == 1 and ib.sleeps == [0]


def test_reconnect_retries_then_succeeds():
    ib = _FakeIB(fail_first=2)
    assert ibkr._reconnect(ib, _settings(), attempts=4, wait_s=0) is True
    assert ib.connect_calls == 3


def test_reconnect_gives_up_after_all_attempts():
    ib = _FakeIB(fail_first=10)
    assert ibkr._reconnect(ib, _settings(), attempts=3, wait_s=0) is False
    assert ib.connect_calls == 3


# --- IBKR: a slow answer is not the end of history ---------------------------
class _ScriptedIB:
    """Answers history requests from a script of ``(bars, seconds the answer takes)`` on a fake
    clock; ``sleep`` is free."""

    def __init__(self, script):
        self.script, self.now, self.timeouts = list(script), 0.0, []

    def sleep(self, s: float = 0.0) -> None:
        self.now += s

    def reqHistoricalData(self, contract, **kw):  # noqa: N802 - matches ib_async
        self.timeouts.append(kw["timeout"])
        bars, took = self.script.pop(0)
        self.now += took
        return bars


def _window(ib):
    pacer = ibkr.Pacer(max_per_window=100, window_s=600, min_gap_s=0)
    return ibkr.fetch_window(ib, "SPY", "20250115 23:59:00", "10 D", bar="1m", use_rth=False,
                             pacer=pacer, clock=lambda: ib.now)


def test_a_timed_out_window_is_asked_again_and_a_quick_empty_one_ends_the_history():
    timeout = ibkr.REQUEST_TIMEOUT_S
    slow_then_ok = _ScriptedIB([([], timeout), (["bar"], 30.0)])
    assert _window(slow_then_ok) == (["bar"], "ok") and slow_then_ok.timeouts == [timeout] * 2
    quick_empty = _ScriptedIB([([], 0.4)])
    assert _window(quick_empty) == ([], "empty") and len(quick_empty.timeouts) == 1
    never = _ScriptedIB([([], timeout)] * (ibkr.TIMEOUT_RETRIES + 1))
    assert _window(never) == ([], "timeout") and not never.script   # every try was used


def test_a_symbol_that_never_answers_is_left_out_not_stored_with_a_hole(monkeypatch):
    import sys
    import time
    from types import SimpleNamespace

    bar = {"date": pd.Timestamp("2024-01-10 15:00", tz="UTC"), "open": 1.0, "high": 1.0,
           "low": 1.0, "close": 1.0, "volume": 10.0}
    script = {"SLOW": [([], 0.06), ([], 0.06)],          # two windows' worth of silence
              "FAST": [([bar], 0.0), ([], 0.0)]}         # one window of bars, then the start

    class IB:
        def connect(self, *args, **kwargs):
            pass

        def isConnected(self):  # noqa: N802 - matches ib_async
            return True

        def qualifyContracts(self, *contracts):  # noqa: N802 - matches ib_async
            return contracts

        def sleep(self, s=0.0):
            pass

        def disconnect(self):
            pass

        def reqHistoricalData(self, contract, **kw):  # noqa: N802 - matches ib_async
            bars, took = script[contract.symbol].pop(0)
            time.sleep(took)                                # a real (short) wait: a timeout
            return bars

    monkeypatch.setitem(sys.modules, "ib_async", SimpleNamespace(
        IB=IB, Stock=lambda symbol, exchange, currency: SimpleNamespace(symbol=symbol),
        util=SimpleNamespace(df=lambda raw: pd.DataFrame(raw))))
    monkeypatch.setattr(ibkr, "REQUEST_TIMEOUT_S", 0.05)
    monkeypatch.setattr(ibkr, "TIMEOUT_RETRIES", 1)
    seen = []
    out = ibkr.fetch_batch(["SLOW", "FAST"], "1m", date(2024, 1, 1), date(2024, 1, 15),
                           settings=_settings(), on_symbol=lambda s, bars: seen.append(s))
    assert list(out) == ["FAST"] and len(out["FAST"]) == 1 and seen == ["FAST"]


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
