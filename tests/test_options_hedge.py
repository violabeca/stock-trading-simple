"""options_hedge research helpers: the ATM straddle a chain day yields, the four screens (causal,
first day of an episode only), news and earnings timing, the straddle's return at the ask, split
and earnings flags, and the matched-random control (it finds a planted edge, it does not find one
in noise, and it matches by date). Synthetic data only (no DB, no network)."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies

_SPEC = importlib.util.spec_from_file_location(
    "test_options_hedge_rs", strategies.STRATEGIES_DIR / "options_hedge" / "research.py")
rs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rs)

NYC = "America/New_York"


def _epoch(day):
    """MarketData stamps expiries at 16:00 New York."""
    return int(pd.Timestamp(f"{day} 16:00", tz=NYC).timestamp())


def chain_rows(date, S, expiries, strikes, root="ABC", bid=1.0, ask=1.2, vol=10):
    """Every call and put of every expiry × strike, all quoted ``bid``/``ask``."""
    rows = []
    for exp in expiries:
        for k in strikes:
            for side in ("call", "put"):
                rows.append(dict(date=date, optionSymbol=f"{root}{pd.Timestamp(exp):%y%m%d}"
                                 f"{side[0].upper()}{int(k * 1000):08d}", expiration=_epoch(exp),
                                 side=side, strike=k, bid=bid, ask=ask, underlyingPrice=S,
                                 volume=vol, openInterest=100))
    return pd.DataFrame(rows)


def test_summary_picks_the_expiry_nearest_the_target_and_the_strike_nearest_the_price():
    raw = chain_rows("2026-01-05", 101.0, ["2026-01-12", "2026-02-02", "2026-02-09", "2026-03-06"],
                     [95.0, 100.0, 105.0])
    out = rs.chain_day_summary(raw, "ABC").iloc[0]
    assert out["30d_exp"] == pd.Timestamp("2026-02-02")      # 28 days beats 35 (target 30)
    assert out["7d_exp"] == pd.Timestamp("2026-01-12")       # shortest 4-14 days out
    assert out["30d_K"] == 100.0 and out["7d_K"] == 100.0
    assert out["opt_vol"] == 4 * 3 * 2 * 10 and out["S"] == 101.0


def test_summary_tie_goes_to_the_shorter_expiry_and_a_one_sided_leg_is_skipped():
    raw = chain_rows("2026-01-05", 101.0, ["2026-01-30", "2026-02-09"], [95.0, 100.0, 105.0])
    one_sided = (raw["strike"] == 100.0) & (raw["side"] == "put")
    raw.loc[one_sided, "bid"] = 0.0
    out = rs.chain_day_summary(raw, "ABC").iloc[0]
    assert out["30d_exp"] == pd.Timestamp("2026-01-30")      # 25 and 35 days: the shorter
    assert out["30d_K"] == 105.0             # 100 has no put bid; 105 is nearer than 95


def test_summary_drops_adjusted_contracts_and_far_strikes():
    raw = pd.concat([chain_rows("2026-01-05", 100.0, ["2026-02-04"], [100.0], root="ABC1"),
                     chain_rows("2026-01-05", 100.0, ["2026-02-04"], [90.0, 110.0])])
    out = rs.chain_day_summary(raw, "ABC").iloc[0]
    assert np.isnan(out["30d_K"])            # ABC1 is another root; 90/110 > 5% away
    assert out["opt_vol"] == 2 * 2 * 10                        # only standard contracts counted


def test_liquid_variant_skips_a_wide_fresh_strike_for_the_liquid_one_beside_it():
    raw = pd.concat([chain_rows("2026-01-05", 101.2, ["2026-02-04"], [100.0], bid=2.0, ask=2.1),
                     chain_rows("2026-01-05", 101.2, ["2026-02-04"], [101.0], bid=1.0, ask=2.0)])
    out = rs.chain_day_summary(raw, "ABC").iloc[0]
    assert out["30d_K"] == 101.0                 # pre-registered: nearest two-sided strike
    assert out["30d_liq_K"] == 100.0             # liquid variant: nearest within the spread filter


def _frames(n=80, m=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-09-01", periods=n)
    cols = [f"S{i}" for i in range(m)]
    f = lambda a: pd.DataFrame(a, index=idx, columns=cols)  # noqa: E731
    return (f(rng.integers(40, 400, (n, m)).astype(float)), f(rng.lognormal(10, 0.6, (n, m))),
            f(100 * np.exp(np.cumsum(rng.normal(0, 0.03, (n, m)), 0))),
            f(rng.poisson(1.5, (n, m)).astype(float)))


def test_screens_never_read_a_later_session():
    full = rs.screens(*_frames())
    cut = rs.screens(*[x.iloc[:50] for x in _frames()])
    for k in rs.SCREENS:
        for a, b in zip(full[k], cut[k], strict=True):
            pd.testing.assert_frame_equal(a.iloc[:50], b)


def test_a_screen_counts_only_the_first_day_of_an_episode():
    ov, va, adj, news = _frames()
    ov[:] = 100.0
    ov.iloc[30:33, 0] = 1000.0                   # three unusual days in a row
    ov.iloc[40, 0] = 1000.0                      # and again after 7 quiet sessions
    pool, fire = rs.screens(ov, va, adj, news)["options_volume"]
    assert list(np.flatnonzero(fire["S0"])) == [30, 40]
    assert not pool.iloc[:rs.LOOKBACK].any().any()            # no full trailing window yet
    ov[:] = 10.0                                              # a median under 50 contracts
    assert not rs.screens(ov, va, adj, news)["options_volume"][0].any().any()


def test_news_counts_roll_weekends_to_monday_and_count_duplicates_once(tmp_path):
    sessions = pd.bdate_range("2026-01-05", "2026-01-16")
    day = lambda d: int(pd.Timestamp(d, tz=NYC).timestamp())  # noqa: E731 — the feed stamps midnight
    stamps = ["2026-01-06", "2026-01-06", "2026-01-10", "2026-01-11", "2026-01-12"]
    n = pd.DataFrame({"headline": ["a", "a", "b", "c", "d"],
                      "publicationDate": [day(d) for d in stamps]})
    (tmp_path / "ABC").mkdir()
    n.to_csv(tmp_path / "ABC" / "2026-01-06_2026-01-31.csv.gz", index=False)
    out = rs.news_counts(["ABC", "XYZ"], sessions, "2026-01", "2026-01", root=tmp_path)["ABC"]
    assert out[pd.Timestamp("2026-01-06")] == 1                # duplicate headline counted once
    assert out[pd.Timestamp("2026-01-12")] == 3                # Sat + Sun + Mon
    assert np.isnan(out[pd.Timestamp("2026-01-05")])           # before the file covers
    assert out[pd.Timestamp("2026-01-07")] == 0


def test_reaction_session_after_the_close_is_the_next_one():
    sessions = pd.bdate_range("2026-01-05", "2026-01-16")
    e = pd.DataFrame({"symbol": ["A", "A", "A"], "ts": pd.to_datetime(
        ["2026-01-06 16:05", "2026-01-08 07:00", "2026-01-10 09:00"]).tz_localize(NYC)})
    pos = rs.reaction_positions(e, sessions)["pos"].tolist()
    assert [sessions[p] for p in pos] == [pd.Timestamp("2026-01-07"), pd.Timestamp("2026-01-08"),
                                          pd.Timestamp("2026-01-12")]


HOLD = 22  # sessions from entry to expiry in the synthetic chain


def _summ():
    """Name ABC, daily rows; each day quotes a straddle at K 100 expiring HOLD sessions later."""
    days = pd.bdate_range("2026-01-05", periods=120)
    S = np.linspace(100, 110, len(days))
    exp = pd.Series(days).shift(-HOLD).to_numpy()
    d = pd.DataFrame({"symbol": "ABC", "date": days, "S": S, "opt_vol": 100.0, "opt_oi": 1.0,
                      "30d_exp": exp, "30d_K": np.where(pd.isna(exp), np.nan, 100.0),
                      "30d_cb": 2.0, "30d_ca": 2.1, "30d_pb": 1.9, "30d_pa": 2.0})
    return d, days


def test_straddle_return_at_the_ask_is_exact():
    summ, days = _summ()
    df = rs.straddles(summ, days, "30d")
    row = df[df["entry"] == days[2]].iloc[0]
    assert row["t"] == days[1] and row["exp_pos"] == 2 + HOLD   # screen day = the session before
    ST = summ["S"].iloc[2 + HOLD]
    assert row["ST"] == pytest.approx(ST) and row["payoff"] == pytest.approx(ST - 100)
    outlay = 100 * (2.1 + 2.0) + 2 * rs.FEE
    assert row["r_ask"] == pytest.approx((100 * (ST - 100) - rs.FEE) / outlay - 1)
    assert row["r_mid"] == pytest.approx((ST - 100) / 4.0 - 1)
    assert row["spread"] == pytest.approx(0.2 / 4.0)
    assert df["exp_pos"].max() < len(days)                     # no straddle past the data


def _flags(brk_pos=None, react_pos=None):
    summ, days = _summ()
    df = rs.straddles(summ, days, "30d")
    price = summ.pivot(index="date", columns="symbol", values="S")
    adj = price.copy()
    if brk_pos is not None:
        adj.iloc[brk_pos:] *= 2                # a 1:2 reverse split in the DB only
    r = react_pos if react_pos is not None else 10**6
    react = pd.DataFrame({"symbol": ["ABC"], "pos": [r]})
    out = rs.flag_rows(df, rs.breaks(price, adj), price.notna(), react)
    return out.set_index("t_pos")


def test_a_split_anywhere_from_t_minus_20_to_expiry_breaks_the_row():
    b = 60
    brk = rs.breaks(_summ()[0].pivot(index="date", columns="symbol", values="S"),
                    _summ()[0].pivot(index="date", columns="symbol", values="S").mul(
                        np.r_[np.ones(b), 2 * np.ones(60)], axis=0))
    assert list(np.flatnonzero(brk["ABC"])) == [b]
    f = _flags(brk_pos=b)["broken"]
    exp_after_t = HOLD + 1                                       # exp_pos = t_pos + 1 + HOLD
    assert f[b + rs.LOOKBACK] and not f[b + rs.LOOKBACK + 1]
    assert f[b - exp_after_t] and not f[b - exp_after_t - 1]
    assert not _flags()["broken"].any()


def test_earnings_window_is_t_minus_2_to_expiry_plus_1():
    r = 60
    e = _flags(react_pos=r)["earn"]
    assert e[r + 2] and not e[r + 3]
    last = r - (HOLD + 1) - 1                                    # exp_pos + 1 == r
    assert e[last] and not e[last - 1]


def _pool(n_days=40, names=60, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-01-05", periods=n_days)
    df = pd.DataFrame({"entry": np.repeat(days, names),
                       "symbol": np.tile([f"S{i}" for i in range(names)], n_days)})
    df["tercile"] = rng.integers(1, 4, len(df))
    df["payoff"] = rng.lognormal(0, 0.8, len(df))
    df["outlay"] = 100 * np.exp(0.32) + 2 * rs.FEE      # priced at the payoff's mean
    df["cost_mid"] = df["outlay"] / 100
    df["r_ask"] = rs.r_ask(df["payoff"], df["outlay"])
    df["r_mid"] = df["payoff"] / df["cost_mid"] - 1
    return df


def test_control_finds_a_planted_edge_and_not_noise():
    pool = _pool()
    rng = np.random.default_rng(5)
    ev = pool.iloc[rng.choice(len(pool), 300, replace=False)].copy()
    noise = rs.score_cell(pool, ev, n_draws=4000)
    assert noise["p"] > 0.01 and not noise["skill"]
    assert noise["power_1.30"] > 0.8                           # 300 events see +30%
    boosted = pool.copy()
    boosted.loc[ev.index, "r_ask"] = rs.r_ask(ev["payoff"], ev["outlay"], 1.5)
    hit = rs.score_cell(boosted, boosted.loc[ev.index], n_draws=4000)
    assert hit["skill"] and hit["p"] < rs.ALPHA


def test_control_matches_the_entry_date():
    pool = _pool()
    good = pool["entry"] == pool["entry"].iloc[0]
    pool.loc[good, "r_ask"] += 1.0                              # one day when every straddle paid
    ev = pool[good].iloc[:20]
    res = rs.score_cell(pool, ev, n_draws=2000)
    assert res["ctrl_r_ask"] == pytest.approx(pool.loc[good, "r_ask"].mean(), abs=0.15)
    assert not res["skill"]                                     # the day, not the screen


def test_robustness_drops_the_best_week_and_top_names():
    ev = pd.DataFrame({"entry": pd.to_datetime([
                           "2026-01-05", "2026-01-06", "2026-01-13", "2026-01-20"]),
                       "symbol": ["A", "A", "B", "C"], "r_ask": [3.0, 1.0, -0.5, -0.5]})
    week, names = rs.without_best(ev)
    assert week == pytest.approx(-0.5)
    assert np.isnan(names)                                      # every name is in the top 5
    lo, hi = rs.week_ci(ev, n=500)
    assert lo <= ev["r_ask"].mean() <= hi


def test_assemble_reads_the_screen_on_the_session_before_the_entry():
    summ, days = _summ()
    summ = summ.assign(**{f"7d_{c}": summ[f"30d_{c}"] for c in ("exp", "K", *rs.LEGS)})
    summ = pd.concat([summ, summ.assign(symbol="XYZ")], ignore_index=True)
    spike = days[40]
    summ.loc[(summ["symbol"] == "ABC") & (summ["date"] == spike), "opt_vol"] = 1000.0
    daily = summ[["symbol", "date"]].assign(close=100.0, adj_close=100.0, volume=1e6)
    news = pd.DataFrame(0.0, index=days, columns=["ABC", "XYZ"])
    before_open = pd.to_datetime([days[100]] * 2).tz_localize(NYC) + pd.Timedelta("8h")
    earn = pd.DataFrame({"symbol": ["ABC", "XYZ"], "ts": before_open})
    sessions = rs.calendar(summ, daily)
    assert sessions.equals(days)
    out = rs.assemble(summ, daily, news, earn, sessions)
    rows = out["rows"]["30d"]
    fired = rows[rows["options_volume_fire"]]
    got = list(zip(fired["symbol"], fired["t"], fired["entry"], strict=True))
    assert got == [("ABC", spike, days[41])]
    w = rs.window_rows(rows, out["react"], sessions, str(days[0].date()), str(days[-1].date()))
    assert w["elig"].any() and w["earn_set"].any()
    assert not (w["elig"] & w["earn_set"]).any()
    assert set(w.loc[w["elig"], "tercile"]) == {1, 2, 3}       # implied rises with S: three bins
    assert not w.loc[w["exp_pos"] >= 99, "elig"].any()   # reaction 100 <= expiry + 1


def test_name_clustered_p_sees_one_name_repeated_as_one_name():
    pool = _pool()
    lucky = pool[pool["symbol"] == "S0"].iloc[:30]               # one name, 30 events
    pool.loc[lucky.index, "r_ask"] += 2.0
    others = pool[pool["symbol"].isin([f"S{i}" for i in range(1, 10)])].groupby("symbol").head(1)
    pool.loc[others.index, "r_ask"] = -0.9                       # the other 9 names lost
    events = pool.loc[lucky.index.append(others.index)]
    assert rs.score_cell(pool, events, n_draws=2000)["p"] < rs.ALPHA   # independent picks: "skill"
    assert rs.name_cluster_p(pool, events) > 0.1               # but it is ONE name's luck
    assert np.isnan(rs.name_cluster_p(pool, pool.loc[lucky.index]))    # one name cannot be judged
    spread = pool[pool["symbol"].isin([f"S{i}" for i in range(30, 60)])].groupby("symbol").head(1)
    pool.loc[spread.index, "r_ask"] += 2.0
    assert rs.name_cluster_p(pool, pool.loc[spread.index]) < 0.01   # 30 names agree


def test_bootstrap_p_by_cluster():
    rng = np.random.default_rng(0)
    groups = pd.Series(np.repeat(np.arange(40), 5))
    assert rs.bootstrap_p(pd.Series(rng.normal(0.5, 1, 200)), groups) < 0.01
    assert rs.bootstrap_p(pd.Series(rng.normal(-0.5, 1, 200)), groups) > 0.99
    assert np.isnan(rs.bootstrap_p(pd.Series(np.ones(9)), pd.Series(np.arange(9))))


def test_event_straddle_opens_the_first_expiry_after_the_event_and_marks_it_later(tmp_path):
    entry = chain_rows("2026-01-05", 101.0, ["2026-01-09", "2026-01-16"], [100.0, 105.0],
                       bid=2.0, ask=2.1)
    wide = (entry["strike"] == 100.0) & (entry["expiration"] == _epoch("2026-01-16"))
    entry.loc[wide & (entry["side"] == "call"), "ask"] = 3.0     # 100 strike too wide on 01-16
    exit_ = chain_rows("2026-01-08", 104.0, ["2026-01-16"], [105.0], bid=1.0, ask=1.2)
    (tmp_path / "ABC").mkdir()
    pd.concat([entry, exit_]).to_csv(tmp_path / "ABC" / "2026-01-01_2026-01-31.csv.gz",
                                     index=False)
    plans = pd.DataFrame({"entry": [pd.Timestamp("2026-01-05")],
                          "exit": [pd.Timestamp("2026-01-08")],
                          "min_exp": [pd.Timestamp("2026-01-12")], "tag": ["x"]})
    out = rs.event_straddles("ABC", plans, root=tmp_path).iloc[0]
    assert out["exp"] == pd.Timestamp("2026-01-16")             # not the 01-09 expiry
    assert out["K"] == 105.0                                    # 100 fails the spread filter
    assert (out["cb0"], out["ca0"], out["cb1"], out["ca1"]) == (2.0, 2.1, 1.0, 1.2)
    assert out["S1"] == 104.0 and out["tag"] == "x"
