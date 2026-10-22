"""The pure missing-only planner (cli._coverage_plan, cli._split_at) — no DB, no network."""

from datetime import date

from trading_simple.cli import (
    COMPLETE_SENTINEL,
    FetchSpec,
    _coverage_plan,
    _group_by_window,
    _split_at,
)

_SYMS = ["AAPL", "MSFT", "NVDA", "NEW"]
_TODAY = date(2026, 7, 1)
_FROM = date(2020, 1, 1)
# AAPL deep and refreshed today; MSFT deep but last refreshed 2026-06-01; NVDA only from
# 2023 and refreshed today; NEW absent from the ledger.
_COV = {
    "AAPL": (date(2019, 1, 1), _TODAY),
    "MSFT": (date(2019, 1, 1), date(2026, 6, 1)),
    "NVDA": (date(2023, 1, 1), _TODAY),
}


def _plan(kind, syms=_SYMS, **kw):
    return _coverage_plan(syms, _COV, _FROM, _TODAY, kind=kind, **kw)


def test_range_fetches_only_the_missing_pieces():
    specs = _plan("range", overlap=2)
    # AAPL is complete: nothing. MSFT: only its tail, re-read from covered_to - overlap.
    # NVDA: only its history, bounded by covered_from. NEW: the whole window.
    assert specs == [
        FetchSpec("MSFT", date(2026, 5, 30), None, date(2026, 5, 30), _TODAY),
        FetchSpec("NVDA", _FROM, date(2023, 1, 1), _FROM, date(2023, 1, 1)),
        FetchSpec("NEW", _FROM, None, _FROM, _TODAY),
    ]


def test_range_history_and_tail_are_two_pieces():
    cov = {"X": (date(2023, 1, 1), date(2026, 6, 1))}
    specs = _coverage_plan(["X"], cov, _FROM, _TODAY, kind="range", overlap=0)
    assert specs == [FetchSpec("X", _FROM, date(2023, 1, 1), _FROM, date(2023, 1, 1)),
                     FetchSpec("X", date(2026, 6, 1), None, date(2026, 6, 1), _TODAY)]


def test_stale_days_lets_a_recent_tail_wait():
    # MSFT's tail is 30 days old: missing at stale_days 29, not at 30.
    assert [sp.symbol for sp in _plan("range", syms=["MSFT"], stale_days=29)] == ["MSFT"]
    assert _plan("range", syms=["MSFT"], stale_days=30) == []


def test_overlap_never_reaches_before_the_wanted_history():
    cov = {"X": (_FROM, date(2020, 1, 10))}
    [sp] = _coverage_plan(["X"], cov, _FROM, _TODAY, kind="range", overlap=31)
    assert sp.since == _FROM


def test_refresh_ignores_the_ledger():
    specs = _plan("range", refresh=True)
    assert [sp.symbol for sp in specs] == _SYMS
    assert all(sp.since == _FROM and sp.until is None for sp in specs)


def test_whole_refetches_everything_once_and_records_the_sentinel():
    cov = {**_COV, "AAPL": (COMPLETE_SENTINEL, _TODAY), "MSFT": (COMPLETE_SENTINEL,
                                                                 date(2026, 6, 1))}
    specs = _coverage_plan(_SYMS, cov, _FROM, _TODAY, kind="whole", overlap=30)
    # AAPL complete and fresh: nothing; MSFT stale, NVDA shallow (pre-sentinel ledger), NEW
    # new: one full fetch each, recorded complete to inception.
    assert [sp.symbol for sp in specs] == ["MSFT", "NVDA", "NEW"]
    assert all(sp == FetchSpec(sp.symbol, _FROM, None, COMPLETE_SENTINEL, _TODAY)
               for sp in specs)


def test_monolithic_fetches_once_from_the_earliest_missing_day():
    specs = _plan("monolithic", overlap=30)
    by = {sp.symbol: sp for sp in specs}
    assert "AAPL" not in by
    assert by["MSFT"] == FetchSpec("MSFT", date(2026, 5, 2), None, date(2026, 5, 2), _TODAY)
    assert by["NVDA"] == FetchSpec("NVDA", _FROM, None, _FROM, _TODAY)   # history: one re-fetch
    assert by["NEW"] == FetchSpec("NEW", _FROM, None, _FROM, _TODAY)


def test_split_at_routes_history_to_bulk_and_the_rest_to_the_scraper():
    q_end = date(2026, 3, 31)
    tail = FetchSpec("A", date(2026, 3, 1), None, date(2026, 3, 1), _TODAY)
    history = FetchSpec("B", _FROM, date(2023, 1, 1), _FROM, date(2023, 1, 1))
    recent = FetchSpec("C", date(2026, 5, 1), None, date(2026, 5, 1), _TODAY)
    upto, after = _split_at([tail, history, recent], q_end)
    assert upto == [FetchSpec("A", date(2026, 3, 1), q_end, date(2026, 3, 1), q_end), history]
    assert after == [FetchSpec("A", date(2026, 4, 1), None, date(2026, 4, 1), _TODAY), recent]


def test_group_by_window_collapses_same_and_splits_distinct():
    a = FetchSpec("A", _FROM, None, _FROM, _TODAY)
    b = FetchSpec("B", _FROM, None, _FROM, _TODAY)          # same window
    c = FetchSpec("C", date(2019, 1, 1), _FROM, date(2019, 1, 1), _FROM)
    groups = _group_by_window([a, b, c])
    assert list(groups) == [(_FROM, None), (date(2019, 1, 1), _FROM)]
    assert [sp.symbol for sp in groups[(_FROM, None)]] == ["A", "B"]
    assert [sp.symbol for sp in groups[(date(2019, 1, 1), _FROM)]] == ["C"]
