"""Settings helpers that don't need a DB or a settings.yaml on disk."""

from trading_simple.config import Settings


def test_all_symbols_is_alphabetical_union_across_universes():
    settings = Settings(universe={"A": ["MSFT", "AAPL"], "B": ["AAPL", "NVDA"]})
    assert settings.all_symbols() == ["AAPL", "MSFT", "NVDA"]


def test_all_symbols_empty_when_no_universes():
    assert Settings(universe={}).all_symbols() == []
