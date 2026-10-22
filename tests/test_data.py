"""Panel invariants and configuration loading (no database)."""

import pytest
import yaml

from trading_simple.config import Settings, load_settings


def test_panel_freq_and_symbols(panel):
    assert panel.freq == "1D"
    assert panel.symbols == ["AAA", "BBB"]
    assert panel.rth is None                     # regular-session load has no mask
    assert (panel.open_ <= panel.high).all().all()
    assert (panel.low <= panel.close * 1.0001).all().all()


def test_settings_defaults_and_universe_lookup():
    s = Settings(universe={"starter": ["AAPL", "SPY"]})
    assert s.db.sqlalchemy_url.startswith("postgresql+psycopg://")
    assert s.ibkr.extended_hours is False        # ETH is opt-in by default
    assert s.backtest.engine == "auto"
    assert s.universe_symbols("starter") == ["AAPL", "SPY"]
    with pytest.raises(KeyError, match="starter"):   # message lists known universes
        s.universe_symbols("nope")


def test_load_settings_merges_universes_file(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(yaml.safe_dump({"universe": {"starter": ["AAPL"]}}))
    universes_file = tmp_path / "universes.yaml"
    universes_file.write_text(yaml.safe_dump({"universe": {"Generated": ["NVDA", "MSFT"]}}))
    monkeypatch.setattr("trading_simple.config.UNIVERSES_FILE", universes_file)
    s = load_settings(settings_file)
    assert set(s.universe) == {"starter", "Generated"}


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("TRADING_IBKR__PORT", "4001")
    monkeypatch.setenv("TRADING_DB__PASSWORD", "secret")
    s = Settings()
    assert s.ibkr.port == 4001
    assert "secret" in s.db.sqlalchemy_url


# --- derived intraday bars ----------------------------------------------------------
def _minute_panel(days: int = 2, n: int = 13):
    """``days`` sessions of ``n`` one-minute bars (09:30 New York); closes ramp 1, 2, 3, ..."""
    import numpy as np
    import pandas as pd

    from trading_simple.data import Panel

    tz = "America/New_York"
    idx = pd.DatetimeIndex([pd.Timestamp(f"2026-06-0{d} 09:30", tz=tz) + pd.Timedelta(minutes=m)
                            for d in range(1, days + 1) for m in range(n)]).tz_convert("UTC")
    close = pd.DataFrame({"AAA": np.arange(1, days * n + 1, dtype=float)}, index=idx)
    return Panel(close=close, open_=close - 0.5, high=close + 1.0, low=close - 1.0,
                 volume=close * 0.0 + 100.0, bar="1m", tz=tz)


def test_resample_panel_builds_session_aligned_candles():
    from trading_simple.data import resample_panel

    p5 = resample_panel(_minute_panel(), 5)
    assert p5.bar == "5m" and p5.freq == "5min"
    assert len(p5.close) == 6                                    # 5 + 5 + 3 bars, twice
    local = p5.close.index.tz_convert("America/New_York")
    assert [t.strftime("%H:%M") for t in local[:3]] == ["09:30", "09:35", "09:40"]
    aaa = p5.close["AAA"]
    assert aaa.iloc[0] == 5.0 and p5.open_["AAA"].iloc[0] == 0.5    # last close, first open
    assert p5.high["AAA"].iloc[0] == 6.0 and p5.low["AAA"].iloc[0] == 0.0
    assert p5.volume["AAA"].iloc[0] == 500.0 and p5.volume["AAA"].iloc[2] == 300.0
    assert aaa.iloc[2] == 13.0                                   # the partial last candle


def test_resample_never_merges_across_sessions():
    from trading_simple.data import resample_panel

    p5 = resample_panel(_minute_panel(), 5)
    local = p5.close.index.tz_convert("America/New_York")
    assert local[3].strftime("%m-%d %H:%M") == "06-02 09:30"      # day 2 restarts its buckets
    assert p5.volume["AAA"].iloc[3] == 500.0                      # not 3 leftover + 2 new bars
    with pytest.raises(ValueError, match="1m panel"):
        resample_panel(p5, 15)


def test_panel_freq_parses_any_minute_bar():
    from trading_simple.data import bar_freq

    assert bar_freq("15m") == "15min" and bar_freq("2m") == "2min" and bar_freq("1d") == "1D"
    with pytest.raises(KeyError):
        bar_freq("weekly")
