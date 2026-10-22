"""overnight_open: the night table (``research.py``) on hand-made bars, no database.

No signal of session d reads anything from d's open on (its first hour, its close, its own
after-hours, later days, headlines dated d, filings after 09:30); after-hours prints belong to the
NEXT session's night; filings and headlines land in the right night; half-days are left out; the
walk-forward never sees its own month; and the statistics do what they say.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies

_spec = importlib.util.spec_from_file_location(
    "test_overnight_open_research", strategies.STRATEGIES_DIR / "overnight_open" / "research.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

SYMS = [f"S{i:02d}" for i in range(22)]          # >= 20 names, so fh_ret_x is defined
DAYS = pd.bdate_range("2025-10-01", "2025-12-12")  # holds the 2025-11-28 half-day


def bars(seed: int = 0):
    """Random-walk daily bars and night parts for :data:`SYMS` on :data:`DAYS`."""
    rng = np.random.default_rng(seed)
    daily, parts = [], []
    for s in SYMS:
        close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, len(DAYS))))
        prev = np.r_[close[0], close[:-1]]
        for i, day in enumerate(DAYS):
            c0 = prev[i]
            pm = c0 * np.exp(rng.normal(0, 0.01))
            o = pm * np.exp(rng.normal(0, 0.002))
            fh = o * np.exp(rng.normal(0, 0.01))
            hi = max(o, fh, close[i]) * 1.01
            lo = min(o, fh, close[i]) * 0.99
            daily.append((s, day, o, hi, lo, close[i], float(rng.integers(1e6, 2e6))))
            ah = close[i] * np.exp(rng.normal(0, 0.005))
            vol = lambda: float(rng.integers(1e3, 1e4))  # noqa: E731
            parts += [(s, day, "early", c0, pm, max(c0, pm), min(c0, pm), vol(), 30),
                      (s, day, "late", pm, pm, pm, pm, vol(), 5),
                      (s, day, "pre", pm, pm, pm * 1.002, pm * 0.998, vol(), 60),
                      (s, day, "first", o, fh, max(o, fh) * 1.003, min(o, fh) * 0.997,
                       vol() * 100, 60),
                      (s, day, "post", close[i], ah, max(close[i], ah), min(close[i], ah),
                       vol(), 40)]
    daily = pd.DataFrame(daily, columns=["symbol", "day", "open", "high", "low", "close",
                                         "volume"])
    parts = pd.DataFrame(parts, columns=["symbol", "day", "part", "open", "close", "high",
                                         "low", "volume", "prints"])
    return daily, parts


def ny(stamp: str) -> pd.Timestamp:
    """A New York time."""
    return pd.Timestamp(stamp).tz_localize(R.NY)


def table(daily, parts, filings=None, news=None):
    """The night table indexed by (day, symbol)."""
    return R.night_table(parts, daily, filings, news).set_index(["day", "symbol"])


SIGNALS = [*R.FEATURES, "news_prev"]
OUTCOMES = ["fh_ret", "fh_range", "fh_range_rel", "rest_ret"]


def test_no_signal_reads_the_session_or_anything_after_it():
    daily, parts = bars()
    d = DAYS[30]
    filings = pd.DataFrame({"symbol": ["S00", "S00"], "ts": [ny("2025-11-03 17:00"),
                                                             ny("2025-11-03 08:00")],
                            "form": "8-K", "items": "8.01", "url": ""})
    news = (pd.DataFrame({"symbol": ["S00"], "date": [DAYS[25]], "n": [2]}),
            pd.DataFrame({"symbol": ["S00"], "a": [DAYS[0]], "b": [DAYS[-1]]}))
    base = table(daily, parts, filings, news)

    later_daily = daily["day"] > d
    changed = daily.copy()
    on_d = changed["day"] == d
    changed.loc[on_d, ["high", "low", "close", "volume"]] *= [1.1, 0.9, 1.05, 3.0]
    changed.loc[later_daily, ["open", "high", "low", "close"]] *= 1.3
    moved = parts.copy()
    after_open = (moved["day"] > d) | ((moved["day"] == d) & moved["part"].isin(["first",
                                                                                 "post"]))
    moved.loc[after_open, ["open", "close", "high", "low"]] *= 1.05   # within the scale screen
    moved.loc[after_open, "volume"] *= 5
    late_filings = pd.concat([filings, pd.DataFrame(
        {"symbol": ["S00", "S00"], "ts": [ny(f"{d.date()} 10:00"), ny(f"{d.date()} 16:30")],
         "form": "8-K", "items": "2.02", "url": ""})], ignore_index=True)
    late_news = (pd.concat([news[0], pd.DataFrame({"symbol": ["S00"], "date": [d], "n": [9]})],
                           ignore_index=True), news[1])
    after = table(changed, moved, late_filings, late_news)

    pd.testing.assert_frame_equal(base.loc[d, SIGNALS], after.loc[d, SIGNALS])
    assert not np.allclose(base.loc[d, OUTCOMES], after.loc[d, OUTCOMES])
    assert after.loc[(d, "S00"), "news_same"] == 9 and base.loc[(d, "S00"), "news_same"] == 0


def test_after_hours_belong_to_the_next_session():
    daily, parts = bars()
    p, d = DAYS[30], DAYS[31]
    close_p = daily.loc[(daily["symbol"] == "S03") & (daily["day"] == p), "close"].item()
    sel = (parts["symbol"] == "S03") & (parts["day"] == p) & (parts["part"] == "post")
    parts.loc[sel, "close"] = close_p * 1.02
    t = table(daily, parts)
    row = t.loc[(d, "S03")]
    assert row["ah_ret_atr"] * row["atr_pct"] == pytest.approx(0.02)
    pm = parts.loc[(parts["symbol"] == "S03") & (parts["day"] == d) & (parts["part"] == "early"),
                   "open"].item()
    assert row["on_ret_atr"] * row["atr_pct"] == pytest.approx(pm / (close_p * 1.02) - 1)


def test_filings_land_in_the_right_night():
    daily, parts = bars()
    filings = pd.DataFrame({
        "symbol": ["S00", "S00", "S00", "S00", "S01"],
        "ts": [ny("2025-11-04 17:00"),    # Tue evening -> Wed's night
               ny("2025-11-05 11:00"),    # Wed session -> Thu's previous-session column
               ny("2025-11-08 10:00"),    # Saturday -> Monday's night
               ny("2025-11-10 07:00"),    # Monday pre-market -> Monday's night (results)
               pd.NaT],                   # S01 filed no 8-K: not a filer
        "form": "8-K", "items": ["8.01", "5.02", "7.01", "2.02,9.01", None], "url": ""})
    t = table(daily, parts, filings)
    wed, thu, mon = pd.Timestamp("2025-11-05"), pd.Timestamp("2025-11-06"), pd.Timestamp(
        "2025-11-10")
    assert t.loc[(wed, "S00"), ["sec_night", "sec_prev"]].tolist() == [1, 0]
    assert t.loc[(thu, "S00"), ["sec_night", "sec_prev"]].tolist() == [0, 1]
    assert t.loc[(mon, "S00"), ["sec_night", "earnings_night"]].tolist() == [2, 1]
    assert t.loc[(pd.Timestamp("2025-11-12"), "S00"), "sec_night"] == 0
    assert t.xs("S01", level="symbol")["sec_night"].isna().all()


def test_headlines_dated_on_the_session_are_hindsight_only():
    daily, parts = bars()
    fri, sat, mon, tue = (pd.Timestamp(x) for x in ("2025-11-07", "2025-11-08", "2025-11-10",
                                                    "2025-11-11"))
    counts = pd.DataFrame({"symbol": "S02", "date": [fri, sat, mon], "n": [1, 2, 4]})
    spans = pd.DataFrame({"symbol": ["S02"], "a": [DAYS[0]], "b": [DAYS[-1]]})
    t = table(daily, parts, news=(counts, spans))
    assert t.loc[(mon, "S02"), ["news_prev", "news_same"]].tolist() == [3, 4]
    assert t.loc[(tue, "S02"), ["news_prev", "news_same"]].tolist() == [4, 0]
    assert t.xs("S05", level="symbol")["news_prev"].isna().all()       # no news file


def test_half_days_and_the_next_session_are_left_out():
    daily, parts = bars()
    days = set(R.night_table(parts, daily)["day"])
    assert pd.Timestamp("2025-11-28") not in days and pd.Timestamp("2025-12-01") not in days
    assert pd.Timestamp("2025-12-02") in days


def test_first_hour_outcomes_and_market_demeaning():
    daily, parts = bars()
    t = R.night_table(parts, daily)
    row = t.iloc[100]
    o = daily.loc[(daily["symbol"] == row["symbol"]) & (daily["day"] == row["day"]), "open"].item()
    fh = parts.loc[(parts["symbol"] == row["symbol"]) & (parts["day"] == row["day"])
                   & (parts["part"] == "first")].iloc[0]
    assert row["fh_ret"] == pytest.approx(fh["close"] / o - 1)
    assert row["fh_range"] == pytest.approx((max(fh["high"], o) - min(fh["low"], o)) / o)
    assert t.groupby("day")["fh_ret_x"].mean().abs().max() < 1e-12


def test_walk_forward_never_sees_its_own_month():
    rng = np.random.default_rng(1)
    n = 3000
    t = pd.DataFrame({"day": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 150, n),
                                                                           unit="D"),
                      "x": rng.normal(size=n)})
    t["month"] = t["day"].dt.to_period("M").astype(str)
    t["y"] = (t["x"] + rng.normal(size=n) > 0).astype(float)
    p = R.walk_probs(t, ["x"], "y", first_month="2025-03")
    flipped = t.assign(y=np.where(t["month"] >= "2025-04", 1 - t["y"], t["y"]))
    q = R.walk_probs(flipped, ["x"], "y", first_month="2025-03")
    march = t["month"] == "2025-03"
    assert p[march].notna().all() and np.allclose(p[march], q[march])
    assert p[t["month"] < "2025-03"].isna().all()


def test_statistics():
    rng = np.random.default_rng(2)
    days = np.repeat(np.arange(50), 20)
    v = rng.normal(size=1000)
    g = rng.random(1000) < 0.3
    r = R.cluster_diff(v + 2 * g, g, days)
    assert r["diff"] == pytest.approx((v + 2 * g)[g].mean() - (v + 2 * g)[~g].mean())
    assert r["t"] > 10
    y = (v > 0).astype(float)
    from sklearn.metrics import roc_auc_score
    assert R.auc(v, y) == pytest.approx(roc_auc_score(y, v)) == pytest.approx(1.0)
    t = pd.DataFrame({"day": days, "pm_ret_atr": rng.choice([-1.0, 1.0], 1000)})
    t["fh_ret_x"] = 0.001 * np.sign(t["pm_ret_atr"])
    c = R.continuation(t)
    assert c["bp"] == pytest.approx(10.0) and c["share"] == 1.0


def test_unshift_undoes_edgars_doubled_offset():
    # Micron: index page 16:02:22 (summer) and 16:03:26 (winter); the JSON is 4 h / 5 h later
    js = pd.Series(pd.to_datetime(["2026-10-01T00:02:22Z", "2025-12-18T02:03:26Z"], utc=True))
    true = R.unshift(js).dt.tz_convert(R.NY).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    assert true == ["2026-09-30 16:02:22", "2025-12-17 16:03:26"]
