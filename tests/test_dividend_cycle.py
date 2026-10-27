"""dividend_cycle: point-in-time ex-date prediction, yield, event windows — synthetic, no DB."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "dividend_cycle_research",
    Path(__file__).resolve().parents[1] / "strategies/dividend_cycle/research.py")
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)

CAL = pd.bdate_range("2019-01-01", "2023-12-29")
# a quarterly payer: every 91 days (same weekday), $1 on a $50 stock, paying past the calendar
EX = pd.DatetimeIndex([pd.Timestamp("2019-03-15") + pd.Timedelta(days=91 * k) for k in range(22)])


def _div(skip=None, amount=1.0):
    d = pd.Series(amount, index=EX)
    return d.drop(EX[skip]) if skip is not None else d


def _store(div, close):
    return {"A": {"div": div, "close": close}}


def test_predict_is_point_in_time_and_marks_a_skipped_dividend():
    full = rs.predict(_div(skip=10))
    assert full["anchor"].iloc[0] == EX[3]                       # needs 4 ex-dates of history
    assert (full["pred"] - full["anchor"]).dt.days.eq(91).all()
    skipped = full.loc[full["anchor"] == EX[9]].iloc[0]          # EX[10] never came
    assert pd.isna(skipped["ex"]) and np.isnan(skipped["amount"])
    # the anchor after the gap needs 3 clean quarterly gaps again: the 182-day gap breaks it
    assert EX[10] not in set(full["anchor"]) and EX[11] not in set(full["anchor"])
    # truncating the future changes no earlier row's anchor or prediction
    cut = rs.predict(_div(skip=10).loc[:EX[8]])
    pd.testing.assert_frame_equal(
        cut[["anchor", "pred"]].reset_index(drop=True),
        full.loc[full["anchor"] <= EX[8], ["anchor", "pred"]].reset_index(drop=True))


def test_irregular_payers_get_no_events():
    annual = pd.Series(1.0, index=pd.DatetimeIndex(
        [f"{y}-06-01" for y in range(2015, 2023)]))
    assert rs.predict(annual).empty


def test_ttm_yield_counts_only_dividends_already_gone_ex():
    close = pd.Series(50.0, index=CAL)
    div = pd.Series(np.arange(1.0, len(EX) + 1), index=EX)          # EX[k] pays k + 1
    y = rs.ttm_yield(div, close, CAL)
    day = CAL.get_indexer([EX[5]])[0]
    assert np.isclose(y[day - 1], (2 + 3 + 4 + 5) / 50)             # EX[1..4]; EX[5] not yet ex
    assert np.isclose(y[day], (3 + 4 + 5 + 6) / 50)                 # EX[2..5], never 5 at once


def _synthetic(runup=0.0, skip=None):
    """Flat $50 stock whose adjusted open gains ``runup`` over the 15 sessions to each ex-date."""
    ret = np.zeros(len(CAL))
    for x in EX:
        t = CAL.searchsorted(x)
        if t < len(CAL):
            ret[t - 14:t + 1] = (1 + runup) ** (1 / 15) - 1
    px = pd.Series(50.0 * np.cumprod(1 + ret), index=CAL)
    return pd.DataFrame({"A": px}), _store(_div(skip=skip), pd.Series(50.0, index=CAL))


def test_rule_recovers_a_planted_runup_and_keeps_missed_dividends():
    opens, store = _synthetic(runup=0.03, skip=10)
    ev, Y = rs.events(opens, store)
    mem = np.ones(opens.shape, bool)
    idx = {"h": np.ones(len(opens))}
    d = rs.evaluate(ev, opens.to_numpy(), Y, mem, idx, n=15, h=10, min_yield=0.03)
    assert len(d) > 5 and d["hit"].sum() == len(d) - 1            # one dividend never came
    hit = d[d["hit"]]
    assert np.allclose(hit["L"], 0.03)                            # entered at the run-up's start
    assert np.allclose(hit["S"], 0.0)                             # flat afterwards
    miss = d[~d["hit"]].iloc[0]
    assert np.isnan(miss["S"])                                    # no short without a dividend
    assert np.isclose(miss["L"], 0.03)                            # but the long still ran


def test_low_yield_and_universe_filters():
    opens, store = _synthetic()
    ev, Y = rs.events(opens, store)
    mem = np.ones(opens.shape, bool)
    assert rs.chosen(ev, Y, mem, min_yield=0.03).any()
    assert not rs.chosen(ev, Y, mem, min_yield=0.20).any()        # the stock yields 8%
    assert not rs.chosen(ev, Y, np.zeros(opens.shape, bool), min_yield=0.03).any()


def test_month_t_clusters_by_month():
    x = pd.Series([1.0, 1.0, 1.0, 1.0, 5.0, 5.0])
    month = pd.Series(pd.PeriodIndex(["2020-01"] * 4 + ["2020-02"] * 2, freq="M"))
    mean, t, n = rs.month_t(x, month)
    assert n == 2 and np.isnan(t) and np.isclose(mean, x.mean())  # <3 months: no t
