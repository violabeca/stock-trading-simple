"""CLI command composition and stdout parsing (pure parts of jobs.py)."""

from trading_simple.jobs import altdata_cmd, backfill_cmd, parse_rows_written, parse_symbol_bars


def test_altdata_cmd_composition_and_flags():
    base = " ".join(altdata_cmd("fundamentals", since="2020-01-01", universe="LiquidLeaders"))
    assert "-m trading_simple.cli altdata --dataset fundamentals --since 2020-01-01" in base
    assert "--universe LiquidLeaders" in base
    assert "--skip-covered" not in base and "--refresh" not in base  # off by default
    # symbols override universe; skip/refresh flags appear only when set
    full = " ".join(altdata_cmd("all", since="2020-01-01", universe="ALL",
                                symbols=["AAPL", "MSFT"], skip_covered=True, refresh=True))
    assert "--symbols AAPL,MSFT" in full and "--universe" not in full
    assert "--skip-covered" in full and "--refresh" in full


def test_backfill_cmd_composition():
    cmd = backfill_cmd(["AAPL", "NESN"], bar="1m", since="2026-01-01",
                       until="2026-02-01", provider="ibkr", extended_hours=True)
    joined = " ".join(cmd)
    assert "-m trading_simple.cli backfill" in joined
    assert "--symbols AAPL,NESN --bar 1m --since 2026-01-01" in joined
    assert "--until 2026-02-01 --provider ibkr --extended-hours" in joined


def test_backfill_cmd_regular_only_flag():
    assert "--regular-only" in backfill_cmd(["A"], bar="1m", since="2026-01-01",
                                            extended_hours=False)
    assert "--extended-hours" not in " ".join(
        backfill_cmd(["A"], bar="1d", since="2026-01-01"))


def test_parse_rows_written():
    out = "prices AAPL 1d: 273 bars\nprices NVDA 1d: 273 bars\nprices: 546 rows written\n"
    assert parse_rows_written(out) == 546
    assert parse_rows_written("nothing useful") == 0


def test_parse_symbol_bars():
    assert parse_symbol_bars("prices AAPL 1m: 2759 bars\n") == ("AAPL", 2759)
    assert parse_symbol_bars("prices: 546 rows written") is None
    assert parse_symbol_bars("random noise") is None
