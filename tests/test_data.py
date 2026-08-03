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
