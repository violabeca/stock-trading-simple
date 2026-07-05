"""Universe screening, sector caps, YAML persistence, sibling imports (no network)."""

import pandas as pd
import pytest
import yaml

from trading_simple.universe import (
    _to_float,
    import_universes,
    liquid_leaders,
    screen_candidates,
    write_universe,
)


def _listing() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["AAPL", "TINY", "CHEAP", "WARR", "BRKAB", "MSFT"],
        "name": ["Apple Inc", "Tiny Co", "Cheap Co", "Foo Warrant", "Long Symbol Co", "Microsoft"],
        "price": [200.0, 50.0, 2.0, 30.0, 100.0, 400.0],
        "market_cap": [3e12, 5e8, 5e10, 3e9, 3e9, 3e12],
        "volume": [1e7] * 6,
        "sector": ["Tech", "Tech", "Tech", "Tech", "Industrials", "Tech"],
        "industry": [None] * 6,
        "exchange": ["NASDAQ"] * 6,
    })


def test_screen_candidates_filters():
    out = screen_candidates(_listing(), min_market_cap=1e9, min_price=5.0)
    syms = set(out["symbol"])
    assert "TINY" not in syms      # below cap floor
    assert "CHEAP" not in syms     # below price floor
    assert "WARR" not in syms      # name keyword: warrant
    assert {"AAPL", "MSFT", "BRKAB"} <= syms
    assert out["market_cap"].is_monotonic_decreasing


def test_liquid_leaders_sector_cap():
    screened = pd.DataFrame({
        "symbol": [f"S{i}" for i in range(8)],
        "market_cap": range(800, 0, -100),
        "sector": ["Tech"] * 6 + ["Energy", "Health"],
    })
    picks = liquid_leaders(screened, n=4, sector_cap=0.5)  # max 2 per sector
    assert picks == ["S0", "S1", "S6", "S7"]               # Tech capped at 2


def test_write_universe_merges(tmp_path):
    path = tmp_path / "universes.yaml"
    write_universe("First", ["B", "A"], path=path)
    write_universe("Second", ["C"], path=path)
    loaded = yaml.safe_load(path.read_text())
    assert loaded["universe"] == {"First": ["A", "B"], "Second": ["C"]}  # sorted + merged


def test_import_universes_defaults_to_new_only(tmp_path):
    src, dst = tmp_path / "src.yaml", tmp_path / "dst.yaml"
    src.write_text(yaml.safe_dump({"universe": {"HighOctane": ["B", "A"], "Steady": ["C"]}}))
    write_universe("HighOctane", ["OLD"], path=dst)
    written = import_universes(src, path=dst)
    assert written == {"Steady": 1}                            # existing one untouched
    loaded = yaml.safe_load(dst.read_text())["universe"]
    assert loaded["HighOctane"] == ["OLD"] and loaded["Steady"] == ["C"]


def test_import_universes_overwrite_and_named(tmp_path):
    src, dst = tmp_path / "src.yaml", tmp_path / "dst.yaml"
    src.write_text(yaml.safe_dump({"universe": {"HighOctane": ["B", "A"], "Steady": ["C"]}}))
    write_universe("HighOctane", ["OLD"], path=dst)
    written = import_universes(src, names=["HighOctane"], overwrite=True, path=dst)
    assert written == {"HighOctane": 2}
    loaded = yaml.safe_load(dst.read_text())["universe"]
    assert loaded["HighOctane"] == ["A", "B"]                  # replaced (and sorted)
    assert "Steady" not in loaded                              # names= limits the import


def test_import_universes_unknown_name_lists_available(tmp_path):
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump({"universe": {"X": ["A"]}}))
    with pytest.raises(KeyError, match="Available"):
        import_universes(src, names=["Nope"], path=tmp_path / "dst.yaml")


def test_to_float_parses_screener_strings():
    assert _to_float("$205.19") == 205.19
    assert _to_float("4,965,598,000") == 4_965_598_000.0
    assert pd.isna(_to_float("N/A")) and pd.isna(_to_float(""))
