"""Order-book sync rules of collectors/binance_depth.py (pure parts, no network or DB)."""

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "binance_depth", Path(__file__).resolve().parents[1] / "collectors" / "binance_depth.py")
bd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bd)

SNAP = {"lastUpdateId": 100, "bids": [["10.0", "1"], ["9.0", "2"]], "asks": [["11.0", "3"]]}


def ev(U, u, pu, b=(), a=()):
    """A futures depthUpdate with string levels, as Binance sends it."""
    return {"e": "depthUpdate", "U": U, "u": u, "pu": pu,
            "b": [[str(p), str(q)] for p, q in b], "a": [[str(p), str(q)] for p, q in a]}


def test_rebuild_drops_stale_events_bridges_and_chains_on_pu():
    events = [
        ev(80, 90, 70, b=[(10.0, 99)]),  # u < lastUpdateId: already in the snapshot, dropped
        ev(95, 105, 90, b=[(10.0, 0)], a=[(12.0, 4)]),  # bridges 100; qty 0 removes the level
        ev(130, 140, 105, b=[(9.5, 5)]),  # U is NOT 105 + 1 on futures: pu chains it
    ]
    book = bd.rebuild(SNAP, events)
    assert book["bids"] == {9.0: 2.0, 9.5: 5.0}
    assert book["asks"] == {11.0: 3.0, 12.0: 4.0}
    assert book["u"] == 140 and (book["lo"], book["hi"]) == (9.0, 11.0)


def test_rebuild_waits_when_the_snapshot_is_ahead_of_the_stream():
    assert bd.rebuild(SNAP, [ev(80, 90, 70)]) is None


def test_rebuild_refuses_gaps():
    with pytest.raises(bd.Desync, match="predates"):  # events start after the snapshot's id
        bd.rebuild(SNAP, [ev(120, 130, 110)])
    with pytest.raises(bd.Desync, match="gap"):  # second event does not chain on pu
        bd.rebuild(SNAP, [ev(95, 105, 90), ev(130, 140, 111)])


def test_snapshot_rows_top_levels_inside_the_snapshot_range_and_skips_dead_books():
    book = bd.rebuild(SNAP, [ev(95, 105, 90, b=[(8.0, 7), (10.5, 1)], a=[(13.0, 1)])])
    book["t"] = 50.0
    ts = datetime(2026, 10, 2, tzinfo=UTC)
    books = {"BTCUSDT": book, "ETHUSDT": None, "SOLUSDT": {**book, "t": 0.0}}
    rows, skipped = bd.snapshot_rows(books, levels=2, ts=ts, now=55.0)
    # 8.0 and 13.0 lie past the snapshot's range (a touched level, not the full book): clipped
    assert [(r[2], r[3]) for r in rows] == [("bid", 10.5), ("bid", 10.0), ("ask", 11.0)]
    assert all(r[0] == ts and r[1] == "BTCUSDT" for r in rows)
    assert skipped == ["ETHUSDT", "SOLUSDT"]  # out of sync; stale for > STALE_AFTER
