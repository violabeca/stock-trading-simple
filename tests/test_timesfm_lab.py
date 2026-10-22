"""timesfm_lab: the clock grid (half-days, missing minutes, coarser bars stamped at their close),
the forecast origins (never across a night intraday, a full context, the budget) and the score
(a "no change" forecast saves 0 %, a perfect one 100 %; the luck band is centred). Synthetic data
only: no DB and no model (the TimesFM weights are never loaded here).
"""

import importlib.util

import numpy as np
import pandas as pd

from trading_simple import strategies
from trading_simple.data import Panel

_spec = importlib.util.spec_from_file_location(
    "test_timesfm_lab", strategies.STRATEGIES_DIR / "darts_forecast" / "timesfm_lab.py")
TF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TF)
HALF = "2025-11-28"  # in strategy.HALF_DAYS


def _panel(seed=1):
    """Two stocks, 1-minute bars, Nov 20 - Dec 31 2025: every stored session 390 minutes long
    (as the DB stores half-days too), three minutes of the first day missing."""
    days = pd.bdate_range("2025-11-20", "2025-12-31")
    days = days[~days.isin(pd.to_datetime(["2025-11-27", "2025-12-25"]))]
    idx = pd.DatetimeIndex(np.concatenate([
        (d + pd.Timedelta(hours=9, minutes=30) + pd.to_timedelta(np.arange(390), "min"))
        for d in days])).tz_localize("America/New_York").tz_convert("UTC").delete([5, 6, 7])
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 5e-4, (len(idx), 2)), 0)),
                         index=idx, columns=["AAA", "BBB"])
    vol = pd.DataFrame(rng.integers(1, 5000, (len(idx), 2)).astype(float), index=idx,
                       columns=["AAA", "BBB"])
    return Panel(close=close, open_=close, high=close, low=close, volume=vol, bar="1m"), days


def _fake(B, org, kind):
    """Quantile paths: the median flat at the origin's close, or exactly the actual path."""
    t, c = org["t"].to_numpy(), org["col"].to_numpy()
    H = 60 if B["minutes"] == 1 else 195
    p0 = B["close"][t, c][:, None]
    act = B["close"][t[:, None] + np.arange(1, H + 1), c[:, None]]
    med = np.repeat(p0, H, 1) if kind == "flat" else act
    return np.stack([med * 0.995, med, med * 1.005], -1).astype(np.float32)


def test_grid_cuts_half_days_and_fills_missing_minutes():
    panel, days = _panel()
    B = TF.bars(panel, 1)
    assert len(B["time"]) == (len(days) - 2) * 390 + 2 * 210  # Nov 28 and Dec 24 end at 13:00
    assert B["time"][0] == pd.Timestamp("2025-11-20 09:31") and B["time"][389].hour == 16
    on_half = B["time"][B["time"].normalize() == pd.Timestamp(HALF)]
    assert on_half[-1] == pd.Timestamp(f"{HALF} 13:00")
    assert np.isfinite(B["close"]).all() and B["filled"]["AAA"] > 0
    assert B["volume"][5:8].sum() == 0  # a missing minute: the previous close, no volume


def test_ten_minute_bars_are_the_minutes_inside_them():
    panel, _ = _panel()
    B1, B10 = TF.bars(panel, 1), TF.bars(panel, 10)
    first = B10["day"] == 0
    assert first.sum() == 39 and B10["time"][0] == pd.Timestamp("2025-11-20 09:40")
    assert np.allclose(B10["close"][first][:, 0], B1["close"][9:390:10, 0])
    assert np.isclose(B10["volume"][1, 1], B1["volume"][10:20, 1].sum())
    on_half = B10["time"][B10["time"].normalize() == pd.Timestamp(HALF)]
    assert len(on_half) == 21 and on_half[-1].hour == 13


def test_intraday_origins_never_cross_a_night():
    panel, _ = _panel()
    B = TF.bars(panel, 1)
    org = TF.origins(B, context=1170, horizon=60, clocks=("10:00", "12:00", "15:00"))
    t = org["t"].to_numpy()
    assert (B["day"][t + 60] == B["day"][t]).all() and (t + 1 >= 1170).all()
    assert set(org["time"].dt.strftime("%H:%M")) == {"10:00", "12:00", "15:00"}
    half = org[org["time"].dt.strftime("%Y-%m-%d") == HALF]["time"].dt.strftime("%H:%M")
    assert "15:00" not in set(half) and "12:00" in set(half)  # 12:00 + 60 min ends at 13:00
    assert len(TF.origins(B, context=1170, horizon=60, clocks=("10:00",), limit=7)) == 7


def test_multi_day_origins_sit_at_each_close():
    panel, _ = _panel()
    B = TF.bars(panel, 10)
    org = TF.origins(B, context=780, horizon=195)
    t = org["t"].to_numpy()
    assert (B["day"][t + 1] == B["day"][t] + 1).all()  # the next bar opens the next session
    assert (t + 195 < len(B["time"])).all() and (t + 1 >= 780).all()


def test_score_no_change_saves_nothing_and_perfect_saves_all():
    panel, _ = _panel()
    B = TF.bars(panel, 1)
    org = TF.origins(B, context=1170, horizon=60, clocks=("10:00", "12:00", "15:00"))
    flat = TF.score(B, org, _fake(B, org, "flat"))
    assert abs(flat["skill"]["value"]) < 1e-4
    assert flat["pred_up"] == 0.0 and np.isnan(flat["hit"]["value"])  # a flat median: no call
    best = TF.score(B, org, _fake(B, org, "perfect"))
    assert best["skill"]["value"] > 0.999 and best["hit"]["value"] == 1.0
    assert best["inside"].min() == 1.0 and best["skill"]["value"] > best["skill"]["hi"]
    assert abs(best["corr"]["lo"] + best["corr"]["hi"]) < 1e-9  # mirror pairs: centred band
    assert list(TF.summary({"a": best, "b": flat}).index) == ["a", "b"]


def test_vol_bands_read_nothing_after_the_origin():
    panel, _ = _panel()
    B = TF.bars(panel, 1)
    org = TF.origins(B, context=1170, horizon=60, clocks=("10:00", "12:00", "15:00"))
    cut = int(org["t"].iat[len(org) // 2])
    late = dict(B, close=B["close"].copy())
    rng = np.random.default_rng(7)
    late["close"][cut + 1:] *= np.exp(rng.normal(0, 3e-3, late["close"][cut + 1:].shape))
    a = TF.vol_bands(B, org, 60, level=60, days=5)
    b = TF.vol_bands(late, org, 60, level=60, days=5)
    early = (org["t"] <= cut).to_numpy() & a["ok"]
    for k in ("clock profile", TF.REFERENCE):
        assert np.array_equal(a[k][early], b[k][early])
        assert not np.allclose(a[k][~early & a["ok"]], b[k][~early & a["ok"]])


def test_size_compare_prefers_the_right_size():
    """A random walk with a known volatility: the cheap band is honest, and a band of the true
    size beats the same band three times too wide or too narrow."""
    panel, _ = _panel()
    B = TF.bars(panel, 1)
    org = TF.origins(B, context=1170, horizon=60, clocks=("10:00", "12:00", "15:00"))
    bands = TF.vol_bands(B, org, 60, level=60, days=5)
    p0 = B["close"][org["t"], org["col"]][:, None, None]
    true = 5e-4 * np.sqrt(np.arange(1, 61))[None, :, None] * TF.Z[None, None, :]
    Q = {name: (p0 * np.exp(s * true)).astype(np.float32)
         for name, s in zip(TF.SETUPS, (1.0, 3.0), strict=True)}
    cmp = TF.size_compare(B, org, Q, bands, block=1)
    tab = cmp["table"]
    assert 0.75 < float(tab.loc[TF.REFERENCE, "inside 10-90%"].rstrip("%")) / 100 < 0.85
    assert tab.loc[TF.PRIMARY, "band loss (bp)"] < tab.loc[f"TimesFM band, centred "
                                                           f"({TF.SETUPS[1]})", "band loss (bp)"]
    narrow = {TF.SETUPS[0]: (p0 * np.exp(true / 3)).astype(np.float32),
              TF.SETUPS[1]: Q[TF.SETUPS[1]]}
    assert (TF.size_compare(B, org, narrow, bands)["table"].loc[TF.PRIMARY, "band loss (bp)"]
            > tab.loc[TF.PRIMARY, "band loss (bp)"])
    assert cmp["n"] + cmp["dropped"] == len(org) and not cmp["deciles"].empty
