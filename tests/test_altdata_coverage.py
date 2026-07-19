"""The pure backfill-skip / interval planner (cli._coverage_*) — no DB, no network."""

from datetime import date

from trading_simple.cli import (
    COMPLETE_SENTINEL,
    FetchSpec,
    _coverage_plan,
    _coverage_targets,
    _group_by_window,
)

_SYMS = ["AAPL", "MSFT", "NVDA", "NEW"]
# AAPL/MSFT backfilled deep (2019); NVDA only shallow (2023); NEW absent from the ledger.
_COV = {
    "AAPL": (date(2019, 1, 1), date(2026, 1, 1)),
    "MSFT": (date(2019, 1, 1), date(2026, 1, 1)),
    "NVDA": (date(2023, 1, 1), date(2026, 1, 1)),
}
_TODAY = date(2026, 7, 1)


def test_skip_off_fetches_everything():
    to_fetch, skipped = _coverage_targets(_SYMS, _COV, date(2020, 1, 1), skip_covered=False)
    assert to_fetch == _SYMS and skipped == []


def test_skip_covered_keeps_only_uncovered_and_shallow():
    # Requesting depth 2020: AAPL/MSFT (covered_from 2019) are deep enough -> skipped;
    # NVDA (covered_from 2023 > 2020) and NEW (no ledger row) still need fetching.
    to_fetch, skipped = _coverage_targets(_SYMS, _COV, date(2020, 1, 1), skip_covered=True)
    assert to_fetch == ["NVDA", "NEW"]
    assert skipped == ["AAPL", "MSFT"]


def test_adding_one_symbol_fetches_only_the_new_one():
    # The headline use case: everything already backfilled to 2020 except a newly added name.
    cov = {s: (date(2019, 1, 1), date(2026, 1, 1)) for s in ("AAPL", "MSFT", "NVDA")}
    to_fetch, skipped = _coverage_targets(["AAPL", "MSFT", "NVDA", "NEW"], cov,
                                          date(2020, 1, 1), skip_covered=True)
    assert to_fetch == ["NEW"] and len(skipped) == 3


def test_order_is_preserved():
    to_fetch, _ = _coverage_targets(["NEW", "AAPL", "NVDA"], _COV, date(2020, 1, 1),
                                    skip_covered=True)
    assert to_fetch == ["NEW", "NVDA"]  # input order kept


# --- interval-aware planner ------------------------------------------------
def test_plan_skip_off_fetches_full_window_all():
    specs, skipped = _coverage_plan(["NVDA", "NEW"], _COV, date(2020, 1, 1), _TODAY,
                                    kind="range", skip_covered=False)
    assert skipped == []
    assert all(sp.since == date(2020, 1, 1) and sp.until is None and sp.rec_to == _TODAY
               for sp in specs)


def test_plan_range_partial_fetches_gap_only_and_keeps_tail():
    specs, skipped = _coverage_plan(_SYMS, _COV, date(2020, 1, 1), _TODAY,
                                    kind="range", skip_covered=True)
    assert skipped == ["AAPL", "MSFT"]                 # covered_from 2019 <= 2020
    by = {sp.symbol: sp for sp in specs}
    # NVDA partial: fetch only [2020, 2023); recorded tail stays the pre-existing covered_to.
    assert by["NVDA"] == FetchSpec("NVDA", date(2020, 1, 1), date(2023, 1, 1),
                                   date(2020, 1, 1), date(2026, 1, 1))
    # NEW: full open window to today.
    assert by["NEW"] == FetchSpec("NEW", date(2020, 1, 1), None, date(2020, 1, 1), _TODAY)


def test_plan_range_skips_when_request_shallower_than_coverage():
    # want_from 2024 is later than NVDA's covered_from 2023 -> already deep enough.
    specs, skipped = _coverage_plan(["NVDA"], _COV, date(2024, 1, 1), _TODAY,
                                    kind="range", skip_covered=True)
    assert skipped == ["NVDA"] and specs == []


def test_plan_whole_skips_covered_and_records_sentinel():
    specs, skipped = _coverage_plan(_SYMS, _COV, date(2020, 1, 1), _TODAY,
                                    kind="whole", skip_covered=True)
    assert skipped == ["AAPL", "MSFT"]
    by = {sp.symbol: sp for sp in specs}
    # A whole-history fetch is never gap-bounded, and records the complete-history sentinel.
    assert by["NEW"] == FetchSpec("NEW", date(2020, 1, 1), None, COMPLETE_SENTINEL, _TODAY)
    assert by["NVDA"].until is None and by["NVDA"].rec_from == COMPLETE_SENTINEL


def test_plan_whole_self_heals_shallow_ledger_when_deepened():
    # want_from earlier than every covered_from -> re-fetch, rewriting covered_from to the sentinel.
    specs, skipped = _coverage_plan(["AAPL", "NVDA"], _COV, date(2015, 1, 1), _TODAY,
                                    kind="whole", skip_covered=True)
    assert skipped == []
    assert all(sp.rec_from == COMPLETE_SENTINEL and sp.until is None for sp in specs)


def test_plan_monolithic_refetches_whole_frame_to_deepen():
    specs, skipped = _coverage_plan(["NVDA", "NEW", "AAPL"], _COV, date(2020, 1, 1), _TODAY,
                                    kind="monolithic", skip_covered=True)
    assert skipped == ["AAPL"]                          # covered_from 2019 <= 2020
    by = {sp.symbol: sp for sp in specs}
    # Monolithic (yfinance) has no server-side bound: one re-fetch, recorded to today.
    assert by["NVDA"] == FetchSpec("NVDA", date(2020, 1, 1), None, date(2020, 1, 1), _TODAY)
    assert by["NEW"] == FetchSpec("NEW", date(2020, 1, 1), None, date(2020, 1, 1), _TODAY)


def test_group_by_window_collapses_same_and_splits_distinct():
    a = FetchSpec("A", date(2020, 1, 1), None, date(2020, 1, 1), _TODAY)
    b = FetchSpec("B", date(2020, 1, 1), None, date(2020, 1, 1), _TODAY)          # same window
    c = FetchSpec("C", date(2019, 1, 1), date(2020, 1, 1), date(2019, 1, 1), date(2026, 1, 1))
    groups = _group_by_window([a, b, c])
    assert list(groups) == [(date(2020, 1, 1), None), (date(2019, 1, 1), date(2020, 1, 1))]
    assert [sp.symbol for sp in groups[(date(2020, 1, 1), None)]] == ["A", "B"]
    assert [sp.symbol for sp in groups[(date(2019, 1, 1), date(2020, 1, 1))]] == ["C"]
