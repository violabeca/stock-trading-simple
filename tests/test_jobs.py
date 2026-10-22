"""CLI command composition and stdout parsing (pure parts of jobs.py)."""

import pandas as pd

from trading_simple.jobs import (
    altdata_cmd,
    backfill_cmd,
    covered_symbols,
    mark_symbol,
    merge_tracker,
    parse_rows_written,
    parse_symbol_bars,
    read_tracker,
    seed_tracker,
    write_tracker,
)


def test_altdata_cmd_composition_and_flags():
    base = " ".join(altdata_cmd("fundamentals", since="2020-01-01", universe="LiquidLeaders"))
    assert "-m trading_simple.cli altdata --dataset fundamentals --since 2020-01-01" in base
    assert "--universe LiquidLeaders" in base
    assert "--refresh" not in base and "--stale-days" not in base  # off by default
    # symbols override universe; refresh/stale flags appear only when set
    full = " ".join(altdata_cmd("all", since="2020-01-01", universe="ALL",
                                symbols=["AAPL", "MSFT"], refresh=True, stale_days=0))
    assert "--symbols AAPL,MSFT" in full and "--universe" not in full
    assert "--refresh" in full and "--stale-days 0" in full


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


def test_seed_tracker_is_alphabetical_and_deduped_and_pending():
    seed = seed_tracker(["MSFT", "AAPL", "AAPL"], bar="1m", since="2026-03-28", until="2026-09-28")
    assert list(seed["symbol"]) == ["AAPL", "MSFT"]
    assert (seed["status"] == "pending").all()
    assert (seed["bars"] == 0).all()


def test_merge_tracker_carries_over_matching_completed_rows():
    seed = seed_tracker(["AAPL", "MSFT", "NVDA"], bar="1m", since="2026-03-28", until="2026-09-28")
    existing = pd.DataFrame([
        {"symbol": "AAPL", "bar": "1m", "since": "2026-03-28", "until": "2026-09-28",
         "status": "done", "bars": 1000, "updated_at": "2026-09-01T00:00:00+00:00"},
        {"symbol": "MSFT", "bar": "1m", "since": "2026-01-01",  # stale window (since moved)
         "until": "2026-09-28", "status": "done", "bars": 2000,
         "updated_at": "2026-09-01T00:00:00+00:00"},
    ])
    merged = merge_tracker(existing, seed).set_index("symbol")
    assert merged.loc["AAPL", "status"] == "done"  # target unchanged -> carried over
    assert merged.loc["AAPL", "bars"] == 1000
    assert merged.loc["MSFT", "status"] == "pending"  # target window moved -> reset
    assert merged.loc["NVDA", "status"] == "pending"  # new symbol -> pending


def test_merge_tracker_empty_existing_returns_seed():
    seed = seed_tracker(["AAPL"], bar="1m", since="2026-03-28", until="2026-09-28")
    merged = merge_tracker(pd.DataFrame(), seed)
    pd.testing.assert_frame_equal(merged.reset_index(drop=True), seed.reset_index(drop=True))


def test_mark_symbol_sets_done_or_empty():
    tracker = seed_tracker(["AAPL", "MSFT"], bar="1m", since="2026-03-28", until="2026-09-28")
    tracker = mark_symbol(tracker, "AAPL", 1500)
    tracker = mark_symbol(tracker, "MSFT", 0)
    row = tracker.set_index("symbol")
    assert row.loc["AAPL", "status"] == "done" and row.loc["AAPL", "bars"] == 1500
    assert row.loc["MSFT", "status"] == "empty" and row.loc["MSFT", "bars"] == 0
    assert row.loc["AAPL", "updated_at"] != ""


def test_tracker_csv_roundtrip(tmp_path):
    path = tmp_path / "tracker.csv"
    seed = seed_tracker(["AAPL"], bar="1m", since="2026-03-28", until="2026-09-28")
    assert read_tracker(path).empty  # no file yet
    write_tracker(seed, path)
    pd.testing.assert_frame_equal(read_tracker(path), seed)


def test_covered_symbols_spans_window_with_tolerance():
    cov = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "NVDA"],
        "bars": [50000, 40000, 100],
        "first_ts": pd.to_datetime(["2025-06-18", "2026-04-01", "2026-03-01"], utc=True),
        "last_ts": pd.to_datetime(["2026-09-27", "2026-09-27", "2026-09-27"], utc=True),
    })
    covered = covered_symbols(cov, since="2026-03-28", until="2026-09-28")
    assert covered == {"AAPL": 50000, "NVDA": 100}  # MSFT: first_ts is after since -> a real gap


def test_covered_symbols_empty_frame():
    assert covered_symbols(pd.DataFrame(), since="2026-03-28", until="2026-09-28") == {}
