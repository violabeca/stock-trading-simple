"""informed_flow_ls: the long/short book, its borrow model, the squeeze stop, the sleeves and
their point-in-time behaviour, on synthetic panels (no DB)."""
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_simple import backtest as bt
from trading_simple import strategies
from trading_simple.data import Panel

S = strategies.load("informed_flow_ls")
# the mechanics tests run the book bare: no smoothing (it has its own test)
SMALL = {"neutral_groups": 0, "q_in": 0.8, "q_out": 0.7, "long_slots": 2, "short_slots": 2,
         "skew_smooth": 1, "score_smooth": 1}


def _panel(n=120, dtc=None, close=None, extra=None):
    """Ten names A..J, flat prices, 1M shares a day, 100M shares outstanding, so a name's
    short interest share = days-to-cover x 1% (FINRA's shares short = days-to-cover x 1M).
    Default days-to-cover: A=1 .. J=10."""
    names = list("ABCDEFGHIJ")
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    px = pd.DataFrame(50.0, index=idx, columns=names) if close is None else close
    dtc = dtc if dtc is not None else pd.DataFrame(
        np.tile(np.arange(1.0, 11.0), (n, 1)), index=idx, columns=names)
    feats = {"short_interest": dtc, "si_shares": dtc * 1e6,
             "fund_shares": pd.DataFrame(1e8, index=idx, columns=names), **(extra or {})}
    return Panel(close=px, open_=px.copy(), high=px.copy(), low=px.copy(),
                 volume=pd.DataFrame(1e6, index=idx, columns=names), bar="1d", features=feats)


def _run(panel, **over):
    return S.signals(panel, **{**S.params, **SMALL, **over})


def test_signals_accepts_every_config_param():
    assert set(S.params) <= set(inspect.signature(S.signals).parameters)
    assert S.valid(S.params)


def test_long_the_least_shorted_short_the_most_shorted_next_bar_dollar_neutral():
    out = _run(_panel())
    w = out["weights"]
    assert (w.iloc[0] == 0).all()                         # decided at bar 0's close, filled at 1
    last = w.iloc[-1]
    assert last[["A", "B"]].tolist() == pytest.approx([0.49 / 2] * 2)
    assert last[["I", "J"]].tolist() == pytest.approx([-0.49 / 2] * 2)
    assert last.sum() == pytest.approx(0.0) and last.abs().sum() == pytest.approx(0.98)


def _shorted(p, name, days):
    """``days`` days to cover = ``days``% of shares sold short, in both FINRA fields."""
    p.features["short_interest"][name] = days
    p.features["si_shares"][name] = days * 1e6


def test_crowded_names_are_never_shorted_and_borrow_rises_with_crowding():
    p = _panel()
    _shorted(p, "J", 40.0)                                # 40% of shares sold short
    out = _run(p)
    last = out["weights"].iloc[-1]
    assert last["J"] == 0 and last["I"] == pytest.approx(-0.245)   # H ranks 0.3: not in the zone
    b, tiers = out["borrow"].iloc[-1], [S.params[k] for k in ("borrow_gc", "borrow_mid",
                                                              "borrow_high")]
    assert tiers[0] < tiers[1] < tiers[2]
    assert b["A"] == tiers[0] and b["I"] == tiers[0]      # 9% of shares: general collateral
    _shorted(p, "I", 15.0)
    assert _run(p)["borrow"].iloc[-1]["I"] == tiers[1]    # 15%: the middle tier
    _shorted(p, "I", 25.0)
    assert _run(p)["borrow"].iloc[-1]["I"] == tiers[2]


def test_crowding_reads_finra_shares_short_and_ignores_a_later_split():
    """The stored volume is split-adjusted BACKWARD: before a 10:1 split it reads ten times the
    shares that traded. Days-to-cover x that volume made a lightly shorted name look crowded on
    knowledge of the future split; FINRA's own shares short / SEC's count cannot."""
    p = _panel()
    clean = S.signals.__globals__["compose"](p, **_compose_args())["si_share"].iloc[-1]
    p.volume.loc[: p.volume.index[60], "J"] *= 10              # J splits 10:1 on bar 61
    split = S.signals.__globals__["compose"](p, **_compose_args())
    assert split["si_share"].iloc[-1].equals(clean)            # unchanged by the adjustment
    assert split["si_share"].iloc[30]["J"] == pytest.approx(0.10)   # 10M / 100M, not 100%
    del p.features["si_shares"]                                # the old measure, for contrast
    with pytest.warns(UserWarning, match="reads future splits"):
        leaky = S.signals.__globals__["compose"](p, **_compose_args())["si_share"].iloc[30]["J"]
    assert leaky == pytest.approx(1.0)                         # 10 x 10M x 1 / 100M


def test_an_implausible_crowding_ratio_is_unknown_never_shorted_and_cheap():
    """FINRA's post-split shares short over a pre-split SEC count reads k times too high; above
    si_plausible_max the ratio is unknown: no new short, the cheapest tier, a held one covered."""
    p = _panel()
    p.features["si_shares"].loc[p.close.index[60]:, "J"] = 2e8   # 200% "short" from bar 60
    parts = S.signals.__globals__["compose"](p, **_compose_args())
    assert np.isnan(parts["si_share"]["J"].iloc[-1]) and not parts["shortable"]["J"].iloc[-1]
    assert parts["borrow"]["J"].iloc[-1] == S.params["borrow_gc"]
    w = _run(p)["weights"]["J"]
    assert w.iloc[59] < 0 and (w.iloc[62:] == 0).all()   # held short, then covered


def test_research_load_and_cache_always_attach_finra_shares_short(tmp_path, monkeypatch):
    rs = _research()
    calls = []
    monkeypatch.setattr(rs, "load_panel", lambda *a, **k: _panel())
    monkeypatch.setattr(rs.features, "attach_features", lambda *a, **k: None)
    monkeypatch.setattr(rs, "attach_si_shares",
                        lambda panel, engine: (calls.append(1),
                                               panel.features.__setitem__("si_shares", 1.0)))
    monkeypatch.setattr(rs, "borrow_fees", lambda index, columns: pd.DataFrame(
        np.nan, index=index, columns=columns))
    monkeypatch.setattr(rs, "STUDIES", tmp_path)
    settings = SimpleNamespace(exchanges=None)
    assert "si_shares" in rs.load(None, settings, S, "2024-12-31", ["A"]).features
    first = rs.cached_load(None, settings, S, "2024-12-31", ["A"])      # loads, then caches
    again = rs.cached_load(None, settings, S, "2024-12-31", ["A"])      # from the pickle
    assert "si_shares" in first.features and "si_shares" in again.features
    assert len(calls) == 3                    # load, the cache miss's load, the hit re-attaches


def _compose_args():
    import inspect as _inspect
    g = S.signals.__globals__
    keys = _inspect.signature(g["compose"]).parameters
    return {k: v for k, v in {**g["_DEFAULTS"], **S.params, **SMALL}.items() if k in keys}


def test_the_broker_borrow_fee_overrides_the_tiers_where_it_is_known():
    p = _panel()
    fee = pd.DataFrame(np.nan, index=p.close.index, columns=p.symbols)
    fee["I"] = 0.004                                      # IBKR's own rate for I only
    p.features["borrow_fee"] = fee
    b = _run(p)["borrow"].iloc[-1]
    assert b["I"] == 0.004 and b["J"] == S.params["borrow_gc"]


def test_the_long_and_the_short_side_can_hold_different_counts():
    last = _run(_panel(), long_slots=1, short_slots=3)["weights"].iloc[-1]
    assert (last > 0).sum() == 1 and last["A"] == pytest.approx(0.49)   # one long: half the gross
    assert (last < 0).sum() == 2                          # only I and J sit in the short zone
    assert last["J"] == pytest.approx(-0.49 / 3)          # each short a third of its side


def test_short_stop_buys_back_a_squeeze_and_waits_for_the_rank_to_leave():
    p = _panel()
    p.close.loc[p.close.index[60]:, "J"] = 80.0           # J squeezes +60% on bar 60
    w = _run(p, short_stop=0.5)["weights"]["J"]
    assert w.iloc[59] < 0 and w.iloc[60] < 0              # decided at 60's close ...
    assert (w.iloc[61:] == 0).all()                       # ... bought back at 61, never again
    assert _run(p)["weights"]["J"].iloc[-1] < 0           # the shipped 100% stop holds on


def test_rising_put_skew_ranks_a_name_toward_the_short_side_and_options_are_optional():
    p = _panel(dtc=pd.DataFrame(5.0, index=pd.date_range("2024-01-01", periods=120, freq="B",
                                                         tz="UTC"), columns=list("ABCDEFGHIJ")))
    skew = pd.DataFrame(0.02, index=p.close.index, columns=p.symbols)
    skew.loc[skew.index[50]:, "C"] = 0.10                  # C's puts get dear from bar 50
    skew.loc[skew.index[50]:, "D"] = -0.05                 # D's get cheap
    p.features["opt_skew"] = skew
    w = _run(p, w_si=0.0, skew_window=20)["weights"]
    assert w["C"].iloc[55] < 0 and w["D"].iloc[55] > 0
    assert (w.iloc[:51] == 0).all().all()                  # no skew change yet: nothing ranks
    del p.features["opt_skew"]
    assert (_run(p)["weights"] == 0).all().all()           # all tied: no long, no short


def test_member_mask_keeps_non_members_out_of_the_book():
    p = _panel()
    member = pd.DataFrame(1.0, index=p.close.index, columns=p.symbols)
    member.loc[: member.index[79], "A"] = 0.0              # A joins the universe on bar 80
    p.features["member"] = member
    w = _run(p, long_slots=3)["weights"]["A"]           # B and C fill two of three slots
    assert (w.iloc[:81] == 0).all() and w.iloc[-1] > 0


def test_weights_are_causal_truncating_the_future_changes_nothing():
    rng = np.random.default_rng(3)
    n = 150
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    close = pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, 10)), axis=0)),
                         index=idx, columns=list("ABCDEFGHIJ"))
    dtc = pd.DataFrame(rng.uniform(1, 12, (n, 10)), index=idx, columns=close.columns)
    full = _run(_panel(n, dtc, close))["weights"]
    cut = 100
    part = _run(_panel(cut, dtc.iloc[:cut], close.iloc[:cut]))["weights"]
    pd.testing.assert_frame_equal(full.iloc[:cut], part)


def test_long_side_alone_and_the_engine_charges_borrow_on_the_shorts():
    p = _panel()
    assert (_run(p, sides="long")["weights"] >= 0).all().all()
    out = _run(p)
    free = bt.run(p, {"weights": out["weights"]}, fees=0.0, slippage=0.0)
    paid = bt.run(p, {k: out[k] for k in ("weights", "borrow")}, fees=0.0, slippage=0.0)
    gross = paid.asset_value(group_by=False).abs().sum(axis=1)
    assert (gross <= paid.value() + 1e-6).all()
    assert paid.value().iloc[-1] < free.value().iloc[-1]   # flat prices: borrow is the only cost


def _research():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "informed_flow_ls_research", strategies.STRATEGIES_DIR / "informed_flow_ls" / "research.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pbo_is_low_for_a_real_edge_and_a_coin_flip_for_noise():
    rs = _research()
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-03", periods=800, freq="B", tz="UTC")
    noise = pd.DataFrame(rng.normal(0, 0.01, (800, 30)), index=idx)
    assert 0.25 <= rs.pbo(noise) <= 0.75                   # picking among noise: a coin flip
    edge = noise.copy()
    edge[0] += 0.003                                       # one trial truly better, every stretch
    assert rs.pbo(edge) <= 0.05


def test_robust_prefers_steady_alpha_over_one_lucky_stretch():
    rs = _research()
    idx = pd.date_range("2022-01-03", periods=400, freq="B", tz="UTC")
    bench = pd.Series(0.0, index=idx)
    steady = pd.Series(0.0004, index=idx)
    lucky = pd.Series(0.0, index=idx)
    lucky.iloc[:100] = 0.0016                              # the same total, all in one stretch
    assert rs.robust(steady, bench) > rs.robust(lucky, bench)


def test_member_mask_follows_each_formation_point_in_time():
    rs = _research()
    idx = pd.date_range("2024-01-01", "2024-12-31", freq="B", tz="UTC")
    history = {pd.Timestamp("2024-01-01"): ["A"], pd.Timestamp("2024-07-01"): ["B"]}
    m = rs.member_mask(idx, ["A", "B"], history)
    assert m.loc["2024-06-28", "A"] == 1 and m.loc["2024-06-28", "B"] == 0
    assert m.loc["2024-07-01", "A"] == 0 and m.loc["2024-07-01", "B"] == 1


def test_smoothing_cuts_the_band_crossings_of_a_noisy_score():
    """score_smooth: a composite that jitters around the entry band trades less when averaged."""
    rng = np.random.default_rng(1)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    dtc = pd.DataFrame(np.tile(np.arange(1.0, 11.0), (n, 1)) + rng.normal(0, 2.0, (n, 10)),
                       index=idx, columns=list("ABCDEFGHIJ")).clip(lower=0.5)
    raw = _run(_panel(n, dtc), dtc_window=1)["weights"]
    smooth = _run(_panel(n, dtc), dtc_window=1, score_smooth=5)["weights"]
    changes = lambda w: int((w.diff().abs() > 0).sum().sum())  # noqa: E731
    assert changes(smooth) < changes(raw)


def test_plateau_picks_the_setting_whose_neighbours_also_score():
    """A lone spike loses to a slightly lower trial surrounded by good ones."""
    rs = _research()
    space = {"a": {"type": "int", "low": 0, "high": 100}, "b": {"type": "float", "low": 0.0,
                                                                "high": 1.0}}
    done = pd.DataFrame({"params_a": [50, 52, 48, 51, 49, 100, 95, 90, 98, 92],
                         "params_b": [0.5, 0.5, 0.5, 0.52, 0.48, 1.0, 0.95, 0.9, 0.9, 1.0],
                         "value": [0.08, 0.07, 0.07, 0.07, 0.07, 0.20, 0.0, 0.0, 0.0, 0.0]})
    params, score = rs.plateau(done, space)
    assert params["a"] in (48, 49, 50, 51, 52) and isinstance(params["a"], int)
    assert score < 0.2


def test_long_share_splits_the_gross_between_the_sides():
    w = _run(_panel(), long_share=0.4)["weights"].iloc[-1]
    assert w.clip(lower=0).sum() == pytest.approx(0.98 * 0.4)
    assert w.clip(upper=0).sum() == pytest.approx(-0.98 * 0.6)


def test_borrow_fees_reads_the_saved_ibkr_snapshots(tmp_path, monkeypatch):
    rs = _research()
    (tmp_path / "usa-20261021.txt").write_text(
        "#BOF|2026.10.21|06:17:11\n#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|FIGI|\n"
        "AAA|USD|AAA INC|1|US0000000001|3.10|0.4500|1000000|BBG000000001|\n#EOF|1\n")
    monkeypatch.setattr(rs, "IBKR_DIR", tmp_path)
    idx = pd.date_range("2026-10-19", "2026-10-30", freq="D", tz="UTC")
    fees = rs.borrow_fees(idx, ["AAA", "BBB"])
    assert fees["BBB"].isna().all()                       # not in the file: the tiers stand
    a = fees["AAA"]
    assert a.loc[:"2026-10-20"].isna().all()              # before the snapshot: nothing known
    assert a.loc["2026-10-21":"2026-10-26"].tolist() == pytest.approx([0.0045] * 6)  # 5 days on
    assert a.loc["2026-10-27":].isna().all()              # too old: back to the tiers
