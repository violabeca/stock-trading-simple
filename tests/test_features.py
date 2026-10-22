"""Feature-layer rules: fundamentals screening and per-day pick, and the max age of levels.

Pure functions on synthetic frames — no database, no network.
"""

from datetime import date

import numpy as np
import pandas as pd

from trading_simple import features as F


def _rows(*rows, form="10-Q"):
    """Long fundamentals rows, shaped like :func:`features._fundamental_sql` output."""
    frame = pd.DataFrame(rows, columns=["symbol", "d", "period_end", "accn", "value"])
    return frame.assign(form=form)


def _utc(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def test_newest_period_screens_before_the_day_pick_and_never_regresses():
    rows = _rows(
        ("AAA", date(2024, 2, 1), date(2023, 12, 31), "a1", 1.00e8),
        ("AAA", date(2024, 5, 1), date(2024, 3, 31), "q1", 1.01e8),
        ("AAA", date(2024, 5, 1), date(2024, 4, 20), "q1", 0.0),        # same-day zero
        ("AAA", date(2024, 6, 1), date(2023, 12, 31), "k1a", 9.9e7),    # stale re-report
        ("AAA", date(2024, 8, 1), date(2024, 6, 30), "q2", 1.9e14),     # thousand-fold unit slip
        ("AAA", date(2024, 11, 1), date(2024, 9, 30), "q3", 1.02e8),
    )
    out = F._newest_period(F._screen_shares(rows))
    assert list(zip(out["d"], out["value"], strict=True)) == [
        (date(2024, 2, 1), 1.00e8),
        (date(2024, 5, 1), 1.01e8),     # the zero cannot hide the real count filed that day
        (date(2024, 11, 1), 1.02e8),    # the stale re-report and the 1.9e14 slip are gone
    ]


def test_newest_period_breaks_same_day_ties_by_accession():
    rows = _rows(("AAA", date(2024, 5, 1), date(2024, 3, 31), "0002", 7.0),
                 ("AAA", date(2024, 5, 1), date(2024, 3, 31), "0001", 5.0))
    assert F._newest_period(rows)["value"].tolist() == [7.0]
    assert F._newest_period(rows.iloc[::-1])["value"].tolist() == [7.0]   # input order irrelevant


def test_newest_period_keeps_negative_levels():
    """Negative equity is real (heavy buybacks); only share counts are screened."""
    rows = _rows(("AAA", date(2024, 2, 1), date(2023, 12, 31), "a1", -3.5e9))
    assert F._newest_period(rows)["value"].tolist() == [-3.5e9]


def test_screen_shares_ignores_counts_furnished_on_form_6k():
    """RY's 2019 6-Ks tagged its share count in thousands: inside the band, so only the form
    can catch it."""
    rows = pd.concat([
        _rows(("RY", date(2018, 11, 28), date(2018, 10, 31), "a18", 1.439e9), form="40-F"),
        _rows(("RY", date(2019, 2, 22), date(2019, 1, 31), "k19", 1.437074e6), form="6-K"),
        _rows(("RY", date(2019, 12, 4), date(2019, 10, 31), "a19", 1.431e9), form="40-F"),
    ])
    assert F._newest_period(F._screen_shares(rows))["value"].tolist() == [1.439e9, 1.431e9]


def _level(*obs):
    """A long ``(symbol, d, value)`` frame for one symbol."""
    return pd.DataFrame({"symbol": "AAA", "d": [pd.Timestamp(d).date() for d, _ in obs],
                         "value": [v for _, v in obs]})


def test_to_grid_max_age_masks_levels_older_than_the_limit():
    index = pd.date_range("2020-01-02", "2020-01-20", freq="D", tz="UTC")
    col = F._to_grid(_level(("2020-01-01", 5.0)), index, fill="ffill", max_age_days=10)["AAA"]
    assert col[_utc("2020-01-02")] == 5.0          # observed 01-01, visible the next bar
    assert col[_utc("2020-01-12")] == 5.0          # as of 01-11 it is 10 days old: carried
    assert np.isnan(col[_utc("2020-01-13")])       # as of 01-12 it is 11 days old: masked
    assert col[_utc("2020-01-13"):].isna().all()


def test_to_grid_max_age_refreshes_on_a_new_observation():
    index = pd.date_range("2020-01-02", "2020-02-15", freq="D", tz="UTC")
    col = F._to_grid(_level(("2020-01-01", 5.0), ("2020-01-31", 6.0)), index,
                     fill="ffill", max_age_days=20)["AAA"]
    assert np.isnan(col[_utc("2020-01-25")])       # 23 days after the first filing
    assert col[_utc("2020-02-01")] == 6.0          # a new filing makes the level live again


def test_to_grid_without_max_age_is_unchanged():
    index = pd.date_range("2020-01-02", "2020-03-31", freq="B", tz="UTC")
    long = pd.concat([_level(("2019-11-15", 2.0), ("2020-02-03", 3.0)),
                      _level(("2020-01-10", 9.0)).assign(symbol="BBB")])
    base = F._to_grid(long, index, fill="ffill")
    pd.testing.assert_frame_equal(F._to_grid(long, index, fill="ffill", max_age_days=None), base)
    pd.testing.assert_frame_equal(F._to_grid(long, index, fill="ffill", max_age_days=10**6), base)


def test_fund_levels_identical_for_train_only_and_full_load():
    """A train-only load must match the full load on every train bar.

    Train and holdout panels are loaded separately. A rule that consulted a later filing
    would leak holdout-era information into training data through exactly this seam.
    """
    rows = _rows(
        ("AAA", date(2019, 11, 1), date(2019, 9, 30), "q319", 1.0e8),
        ("AAA", date(2020, 2, 1), date(2019, 12, 31), "a19", 1.9e14),    # unit slip inside train
        ("AAA", date(2020, 5, 1), date(2020, 3, 31), "q120", 1.1e8),
        ("AAA", date(2020, 8, 1), date(2020, 6, 30), "q220", 1.2e8),     # after the train cutoff
        ("AAA", date(2020, 9, 1), date(2020, 3, 31), "q120a", 1.05e8),   # late amendment
        ("BBB", date(2018, 1, 5), date(2017, 12, 31), "b17", 5.0e7),     # goes stale in train
    )
    train_index = pd.date_range("2020-01-01", "2020-06-30", freq="B", tz="UTC")
    full_index = pd.date_range("2020-01-01", "2020-12-31", freq="B", tz="UTC")

    def grid(index):
        known = rows[rows["d"] <= index[-1].date()]          # what an as_of-bounded query returns
        return F._to_grid(F._newest_period(F._screen_shares(known)), index, fill="ffill",
                          max_age_days=F._FUND_MAX_AGE_DAYS)

    train, full = grid(train_index), grid(full_index)
    pd.testing.assert_frame_equal(train, full.loc[train_index, train.columns], check_exact=True)
    assert train["BBB"].isna().iloc[-1] and train["BBB"].notna().iloc[0]   # the cap did act


def test_only_fundamentals_are_screened_and_only_levels_age_capped():
    """Fundamentals are screened; they and short interest expire; the event flows are read
    exactly as before."""
    assert sorted(F._FUNDAMENTALS) == ["fund_assets", "fund_equity", "fund_gp_ttm",
                                       "fund_ni_ttm", "fund_shares"]
    assert {"short_interest": F._SI_MAX_AGE_DAYS,
            **dict.fromkeys(F._FUNDAMENTALS, F._FUND_MAX_AGE_DAYS),
            **dict.fromkeys(F._OPTION_FEATURES, F._OPT_MAX_AGE_DAYS)} == F._MAX_AGE_DAYS


def test_short_interest_filed_under_a_reused_ticker_is_dropped():
    """BNY's rows until 2026-05 are a municipal fund's: they must never rank BNY Mellon."""
    rows = pd.DataFrame({
        "symbol": ["BNY", "BNY", "AAA", "META", "META"],
        "d": [date(2025, 1, 24), date(2026, 6, 9), date(2025, 1, 24), date(2021, 8, 10),
              date(2022, 6, 24)],
        "settlement_date": [date(2025, 1, 15), date(2026, 5, 29), date(2025, 1, 15),
                            date(2021, 7, 30), date(2022, 6, 15)],
        "value": [1.0, 3.06, 2.0, 0.4, 1.8]})
    kept = F._screen_short_interest(rows)
    assert list(kept.columns) == ["symbol", "d", "value"]
    assert kept["value"].tolist() == [3.06, 2.0, 1.8]      # the fund and the ETF rows are gone


def test_a_renamed_company_keeps_its_own_rows_under_the_old_ticker():
    """Facebook's rows were filed as FB until the rename: they ARE META's history."""
    rows = pd.DataFrame({
        "symbol": ["FB", "META", "FB"],
        "d": [date(2021, 8, 10), date(2021, 8, 10), date(2022, 7, 12)],
        "settlement_date": [date(2021, 7, 30), date(2021, 7, 30), date(2022, 6, 30)],
        "value": [1.2, 0.4, 9.9]})
    kept = F._screen_short_interest(rows)
    assert kept["symbol"].tolist() == ["META", "FB"]    # FB inside its window becomes META;
    assert kept["value"].tolist() == [1.2, 9.9]         # the ETF's META row is gone


def test_short_interest_that_stops_arriving_is_masked_not_carried():
    """2026-07: settlements kept arriving for AAA but stopped for BBB (never ingested). BBB's
    last days-to-cover must expire, not rank for months as if it were current."""
    settles = pd.date_range("2026-01-01", "2026-09-15", freq="SMS")   # the 1st and 15th
    last_seen = pd.Timestamp("2026-07-01")                            # BBB's last settlement
    long = pd.concat([_level(*[(d, 1.0) for d in settles]),
                      _level(*[(d, 2.0) for d in settles if d <= last_seen]).assign(symbol="BBB")])
    index = pd.date_range("2026-02-02", "2026-09-22", freq="B", tz="UTC")
    grid = F._to_grid(long, index, fill="ffill", max_age_days=F._SI_MAX_AGE_DAYS)
    assert grid["AAA"].notna().all()                       # refreshed every ~15 days: never masked
    age = (index.tz_localize(None) - pd.Timedelta(days=1) - last_seen).days   # as-of D-1
    assert grid["BBB"][age <= F._SI_MAX_AGE_DAYS].notna().all()
    assert grid["BBB"][age > F._SI_MAX_AGE_DAYS].isna().all()
    assert grid["BBB"].isna().any()                        # ... and it did expire in the window
