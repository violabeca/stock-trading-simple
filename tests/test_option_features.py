"""providers.marketdata (option-chain summaries from local MarketData files) and the opt_*
features: implied volatilities, the reduction per name-day, the incremental cache, and the
point-in-time grid."""
import pickle

import numpy as np
import pandas as pd
import pytest

from trading_simple import features
from trading_simple.providers import marketdata as md

NY = "America/New_York"


def _vol(m):
    """A smile: puts below the money dearer (+0.5 vol points per 1% below), calls flat."""
    return 0.30 + 0.5 * np.clip(1 - m, 0, None)


def chain_rows(day, spot=100.0, symbol="TEST", volume=10, oi=100):
    """One session's chain: expiries 3, 31 and 90 days out, strikes 80..120, both sides,
    every contract priced by Black-Scholes at the smile's vol (so call - put IV = 0), plus
    one contract on an ADJUSTED root that must be ignored."""
    day = pd.Timestamp(day)
    rows = []
    for dte in (3, 31, 90):
        exp = day + pd.Timedelta(days=dte)
        epoch = int(pd.Timestamp(exp.date()).tz_localize(NY).replace(hour=16).timestamp())
        for k in np.arange(80.0, 121.0, 5.0):
            for side in ("call", "put"):
                mid = float(md.bs_price(spot, k, dte / 365, _vol(k / spot), side == "call"))
                occ = f"{symbol}{exp:%y%m%d}{side[0].upper()}{int(k * 1000):08d}"
                rows.append({"date": f"{day:%Y-%m-%d}", "optionSymbol": occ, "underlying": symbol,
                             "expiration": epoch, "side": side, "strike": k,
                             "bid": mid * 0.99, "ask": mid * 1.01, "mid": mid,
                             "openInterest": oi, "volume": volume, "underlyingPrice": spot})
    rows.append({**rows[0], "optionSymbol": f"{symbol}1{rows[0]['optionSymbol'][len(symbol):]}",
                 "volume": 99_999})
    return pd.DataFrame(rows)


def test_implied_vol_inverts_black_scholes():
    sig = np.array([0.15, 0.3, 0.8])
    for call in (True, False):
        price = md.bs_price(100.0, np.array([90.0, 100.0, 120.0]), 0.25, sig, call)
        assert md.implied_vol(price, 100.0, np.array([90.0, 100.0, 120.0]), 0.25, call) == \
            pytest.approx(sig, abs=1e-6)
    assert np.isnan(md.implied_vol(0.0, 100.0, 120.0, 0.1, True))   # below any volatility


def test_summarize_reads_atm_iv_skew_cpiv_and_flows_from_the_30_day_expiry():
    out = md.summarize(pd.concat([chain_rows("2026-03-02"), chain_rows("2026-03-03", 102.0)]),
                       "TEST")
    assert list(out["d"]) == list(pd.to_datetime(["2026-03-02", "2026-03-03"]))
    first = out.iloc[0]
    assert first["atm_iv"] == pytest.approx(0.30, abs=1e-4)
    assert first["skew"] == pytest.approx(_vol(0.95) - 0.30, abs=1e-4)      # the 95 put vs ATM
    assert first["put_skew"] == pytest.approx(_vol(0.95) - 0.30, abs=1e-4)
    assert first["cpiv"] == pytest.approx(0.0, abs=1e-4)                     # parity holds
    # the adjusted-root contract (99,999 contracts) is left out
    assert first["call_volume"] == 3 * 9 * 10 and first["put_oi"] == 3 * 9 * 100
    assert out.iloc[1]["spot"] == 102.0


def test_cache_redoes_only_the_files_that_changed(tmp_path, monkeypatch):
    root, cache = tmp_path / "chains", tmp_path / "options"
    (root / "TEST").mkdir(parents=True)
    chain_rows("2026-03-02").to_csv(root / "TEST" / "2026-03-01_2026-03-31.csv.gz", index=False)
    chain_rows("2026-04-01").to_csv(root / "TEST" / "2026-04-01_2026-04-30.csv.gz", index=False)
    table = md.build_cache(root=root, cache=cache, workers=1)
    assert len(table) == 2 and table["d"].is_monotonic_increasing
    calls = []
    real = md._summarize_file
    monkeypatch.setattr(md, "_summarize_file", lambda f, s: calls.append(f.name) or real(f, s))
    april = pd.concat([chain_rows("2026-04-01"), chain_rows("2026-04-02")])
    april.to_csv(root / "TEST" / "2026-04-01_2026-04-30.csv.gz", index=False)
    table = md.build_cache(root=root, cache=cache, workers=1)
    assert calls == ["2026-04-01_2026-04-30.csv.gz"]                 # March reused
    assert len(table) == 3
    assert pickle.loads((cache / "summary.pkl").read_bytes()).equals(table)


def test_opt_features_are_served_from_the_next_bar_and_expire(monkeypatch):
    """A chain quoted on the 2nd is visible from the 3rd (as-of D-1), never on the 2nd; a
    name whose chains stop is masked after 5 days instead of carried."""
    table = pd.DataFrame({"symbol": ["A", "A", "B"],
                          "d": pd.to_datetime(["2026-03-02", "2026-03-04", "2026-03-02"]),
                          **{c: np.nan for c in md.COLUMNS}})
    table["skew"] = [0.01, 0.03, 0.05]
    monkeypatch.setattr(md, "load_summaries", lambda cache=None: table)
    idx = pd.date_range("2026-03-02", "2026-03-12", freq="B", tz="UTC")
    out = features.load_features(None, idx, ["A", "B", "C"], ["opt_skew"])["opt_skew"]
    assert np.isnan(out.loc["2026-03-02", "A"])
    assert out.loc["2026-03-03", "A"] == 0.01 and out.loc["2026-03-04", "A"] == 0.01
    assert out.loc["2026-03-05", "A"] == 0.03
    assert out.loc["2026-03-06", "B"] == 0.05                          # 4 days old: carried
    assert np.isnan(out.loc["2026-03-09", "B"])                        # 7 days old: masked
    assert out["C"].isna().all()
