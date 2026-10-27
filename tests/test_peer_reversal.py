"""peer_reversal (STRATEGY_OVERVIEW §6 proposal 5): the peer-relative signal, the clusters, the
book's entry/exit mechanics, the stress-sized slots frozen while held, borrow tiers and causality
— synthetic panels, no DB."""
import importlib.util
import inspect

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies
from trading_simple.data import Panel

PR = strategies.load("peer_reversal")
_spec = importlib.util.spec_from_file_location("peer_reversal_strategy", PR.dir / "strategy.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

NAMES = [f"{g}{k}" for g in "ABC" for k in range(6)]          # three planted groups of six
SMALL = {"n_clusters": 3, "cluster_window": 60, "reform_every": 30, "vol_window": 10,
         "vol_median_window": 40, "slots": 2, "max_hold": 10}


def _panel(n=200, seed=0, idio=0.002, shocks=(), group_shocks=(), vol_jump_at=None,
           volume=None, features=None):
    """Three groups of six names: a market factor, one factor per group, small idiosyncratic
    noise; ``shocks`` = (name, bar, return) added on that bar, ``group_shocks`` = (group
    letter, bar, return) for every member."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="UTC")
    mkt = rng.normal(0.0, 0.01, n)
    if vol_jump_at is not None:
        mkt[vol_jump_at:] *= 4.0
    rets = (mkt[:, None] + np.repeat(rng.normal(0.0, 0.01, (n, 3)), 6, axis=1)
            + rng.normal(0.0, idio, (n, len(NAMES))))
    for name, bar, size in shocks:
        rets[bar, NAMES.index(name)] += size
    for group, bar, size in group_shocks:
        rets[bar, [i for i, c in enumerate(NAMES) if c[0] == group]] += size
    rets[0] = 0.0
    close = pd.DataFrame(50.0 * np.cumprod(1.0 + rets, axis=0), index=idx, columns=NAMES)
    vol = pd.DataFrame(1e6, index=idx, columns=NAMES) if volume is None else volume
    return Panel(close=close, open_=close.copy(), high=close * 1.01, low=close * 0.99,
                 volume=vol, bar="1d", features=features)


def _compose(panel, **over):
    keys = inspect.signature(S.compose).parameters
    params = {k: v for k, v in {**PR.params, **SMALL, **over}.items() if k in keys}
    return S.compose(panel, **params)


def test_every_config_param_is_a_signals_keyword_and_the_shipped_config_is_valid():
    assert set(PR.params) == set(inspect.signature(S.signals).parameters) - {"panel"}
    assert PR.valid(PR.params)
    assert PR.fill == "open" and PR.order_notional == 1000.0


def test_clusters_recover_the_planted_groups_from_past_data_only():
    labels = _compose(_panel())["labels"]
    assert (labels[:60] == -1).all()                       # no formation before bar 60
    for b in (60, 90, 120, 150, 180):
        row = labels[b]
        for g in "ABC":
            members = [NAMES.index(c) for c in NAMES if c[0] == g]
            assert len(set(row[members])) == 1             # one group per planted group
        assert len(set(row)) == 3


def test_signal_is_peer_relative_and_measured_one_day_before_the_decision():
    p = _panel(shocks=[("A0", 140, -0.15)], group_shocks=[("B", 140, -0.15)])
    c = _compose(p)                                        # clusters formed at 120 hold to 150
    sig, move = c["signal"], c["move"]
    assert abs(sig["A0"].iloc[140]) < 0.03                 # bar 140's own move is skipped
    assert sig["A0"].iloc[141] < -0.12                     # A0 fell against its peers
    assert move["B0"].iloc[141] < -0.12                    # B0 fell with its whole cluster ...
    assert abs(sig["B0"].iloc[141]) < 0.03                 # ... so it did not fall against it
    assert 0.0 < sig["A1"].iloc[141] < 0.06                # leave-one-out: A0 drags A1's peers
    assert c["rank"]["A0"].iloc[141] == pytest.approx(1 / 18)


def test_book_enters_the_extremes_exits_on_band_and_max_hold_and_fills_next_bar():
    idx = pd.date_range("2024-01-01", periods=16, freq="B", tz="UTC")
    rank = pd.DataFrame(0.55, index=idx, columns=list("abcdefghij"))
    rank.loc[idx[1], ["a", "b", "c"]] = [0.05, 0.04, 0.08]
    rank.loc[idx[2]:, "a"] = 0.3                     # held between zone and band ...
    rank.loc[idx[2], "b"] = 0.6                      # ... b crosses the band at bar 2
    rank.loc[idx[2], "c"] = 0.08                     # c gets b's slot at bar 2
    rank.loc[idx[3]:, "c"] = 0.45
    rank.loc[idx[1], ["i", "j"]] = [0.97, 0.95]      # i ranks higher but is not shortable
    rank.loc[idx[2]:, "j"] = 0.52
    rank.loc[idx[7]:, "e"] = 0.3
    rank.loc[idx[[7, 9]], "e"] = 0.05                # a fresh extreme restarts e's clock
    shortable = pd.DataFrame(True, index=idx, columns=rank.columns)
    shortable["i"] = False
    enter_ok = pd.DataFrame(True, index=idx, columns=rank.columns)
    can_open = pd.Series(True, index=idx)
    longs, shorts, exits = S._select(rank, shortable, enter_ok, can_open, q_in=0.1, q_out=0.5,
                                     slots=2, max_hold=3)
    assert longs["a"].tolist()[:6] == [False, True, True, True, False, False]   # max_hold
    assert longs["b"].tolist()[:3] == [False, True, False]                       # band
    assert longs["c"].tolist()[:7] == [False, False, True, True, True, False, False]
    assert longs["e"].tolist()[6:14] == [False, True, True, True, True, True, False, False]
    assert not shorts["i"].any()                                                # not shortable
    assert shorts["j"].tolist()[:6] == [False, True, True, True, False, False]  # max_hold
    assert {(s, c, cause) for s, c, _, _, cause in exits} >= {
        (1, "a", "max_hold"), (1, "b", "band"), (1, "e", "max_hold")}
    w = S._weights(longs, shorts, pd.Series(0.1, index=idx))
    assert w["a"].tolist()[:6] == [0.0, 0.0, 0.1, 0.1, 0.1, 0.0]                 # next bar
    assert w["j"].iloc[2] == pytest.approx(-0.1)


def test_nothing_opens_when_closed_and_the_screen_blocks_entries_only():
    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    rank = pd.DataFrame(0.55, index=idx, columns=list("abcdefghij"))
    rank["d"] = 0.05
    shortable = pd.DataFrame(True, index=idx, columns=rank.columns)
    enter_ok = pd.DataFrame(True, index=idx, columns=rank.columns)
    can_open = pd.Series([False, False, True, True, True, True], index=idx)
    longs, _, _ = S._select(rank, shortable, enter_ok, can_open, q_in=0.1, q_out=0.5,
                            slots=2, max_hold=10)
    assert longs["d"].tolist() == [False, False, True, True, True, True]
    enter_ok.loc[idx[2], "d"] = False
    longs, _, _ = S._select(rank, shortable, enter_ok, can_open, q_in=0.1, q_out=0.5,
                            slots=2, max_hold=10)
    assert longs["d"].tolist() == [False, False, False, True, True, True]


def test_slot_size_is_set_at_entry_and_frozen_while_held():
    idx = pd.date_range("2024-01-01", periods=8, freq="B", tz="UTC")
    longs = pd.DataFrame(False, index=idx, columns=["a", "b"])
    shorts = longs.copy()
    longs.loc[idx[1]:idx[5], "a"] = True                  # opened while calm
    longs.loc[idx[4]:idx[6], "b"] = True                  # opened after the switch
    size = pd.Series([0.01] * 3 + [0.02] * 5, index=idx)
    w = S._weights(longs, shorts, size)
    assert w["a"].tolist() == [0, 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0]
    assert w["b"].tolist() == [0, 0, 0, 0, 0, 0.02, 0.02, 0.02]


def test_stress_switch_reads_the_market_volatility_against_its_trailing_median():
    c = _compose(_panel(n=220, vol_jump_at=150))
    assert c["stress"].iloc[152:170].all()
    assert c["slot_size"].iloc[160] == pytest.approx(0.98 / 2 / 2)
    calm = ~c["stress"].iloc[60:150]
    assert calm.any() and c["slot_size"].iloc[60:150][calm].eq(0.49 / 2 / 2).all()


def test_book_is_dollar_neutral_in_equal_slots_and_never_above_gross():
    p = _panel(n=260, vol_jump_at=180)
    w = PR.signals(p, **{**PR.params, **SMALL})["weights"]
    sizes = set(np.round(w.abs().to_numpy()[w.abs().to_numpy() > 0], 12))
    assert sizes <= {round(0.98 / 4, 12), round(0.49 / 4, 12)}
    assert w.abs().sum(axis=1).max() <= 0.98 + 1e-12
    assert (w > 0).sum(axis=1).max() <= 2 and (w < 0).sum(axis=1).max() <= 2
    assert (w.iloc[:61] == 0).all().all()                  # nothing before the first clusters
    held = w.ne(0) & w.eq(w.shift())                       # a held weight never changes
    changes = w.ne(w.shift()) & w.ne(0) & w.shift().ne(0)
    assert held.any().any() and not (changes & (np.sign(w) == np.sign(w.shift()))).any().any()


def test_signals_on_a_truncated_panel_reproduce_the_full_panel_up_to_the_cut():
    full = _panel(n=240, vol_jump_at=170, shocks=[("A0", 150, -0.15), ("C3", 200, 0.12)])
    cut = 185
    trunc = Panel(close=full.close.iloc[:cut], open_=full.open_.iloc[:cut],
                  high=full.high.iloc[:cut], low=full.low.iloc[:cut],
                  volume=full.volume.iloc[:cut], bar="1d")
    a = PR.signals(full, **{**PR.params, **SMALL})["weights"]
    b = PR.signals(trunc, **{**PR.params, **SMALL})["weights"]
    pd.testing.assert_frame_equal(a.iloc[:cut], b)
    assert a.iloc[:cut].abs().sum().sum() > 0


def test_borrow_tiers_and_the_crowded_short_screen():
    p = _panel(n=120)
    idx = p.close.index
    dtc = pd.DataFrame(np.nan, index=idx, columns=NAMES)
    dtc[["A0", "A1", "A2", "A3"]] = [5.0, 15.0, 25.0, 35.0]        # x 1e6 / 1e8 = 5% ... 35%
    shares = pd.DataFrame(1e8, index=idx, columns=NAMES)
    p = Panel(close=p.close, open_=p.open_, high=p.high, low=p.low, volume=p.volume, bar="1d",
              features={"short_interest": dtc, "fund_shares": shares})
    c = _compose(p)
    last = c["borrow"].iloc[-1]
    assert last[["A0", "A1", "A2", "A3", "B0"]].tolist() == [0.005, 0.03, 0.1, 0.1, 0.005]
    ok = c["shortable"].iloc[-1]
    assert ok[["A0", "A2", "B0"]].all() and not ok["A3"]
    assert (PR.signals(p, **{**PR.params, **SMALL})["weights"]["A3"] >= 0).all()


def test_sector_peers_read_the_sector_codes_and_need_them():
    p = _panel(shocks=[("A0", 150, -0.15)])
    codes = pd.DataFrame(np.tile(np.repeat([0.0, 1.0, 2.0], 6), (len(p.close), 1)),
                         index=p.close.index, columns=NAMES)
    q = Panel(close=p.close, open_=p.open_, high=p.high, low=p.low, volume=p.volume, bar="1d",
              features={"sector": codes})
    c = _compose(q, peers="sector")
    assert (c["labels"] == codes.to_numpy().astype(int)).all()
    assert c["signal"]["A0"].iloc[151] < -0.12 and c["signal"]["A0"].iloc[20] == pytest.approx(
        c["move"]["A0"].iloc[20] - c["move"][["A1", "A2", "A3", "A4", "A5"]].iloc[20].mean())
    with pytest.raises(ValueError, match="sector"):
        _compose(p, peers="sector")
    m = _compose(p, peers="market")
    others = m["move"].iloc[151].drop("A0").mean()
    assert m["signal"]["A0"].iloc[151] == pytest.approx(m["move"]["A0"].iloc[151] - others)


def test_volume_cap_blocks_entries_on_a_high_volume_move():
    vol = pd.DataFrame(1e6, index=pd.date_range("2023-01-02", periods=200, freq="B", tz="UTC"),
                       columns=NAMES)
    vol.iloc[150, NAMES.index("A0")] = 5e6
    p = _panel(shocks=[("A0", 150, -0.15)], volume=vol)
    c = _compose(p, volume_cap=3.0)
    assert not c["enter_ok"]["A0"].iloc[151:156].any() and c["enter_ok"]["A0"].iloc[156]
    assert c["enter_ok"].drop(columns="A0").iloc[151:156].all().all()
    free = {**PR.params, **SMALL, "slots": 18}
    plain = PR.signals(p, **free)["weights"]["A0"]
    capped = PR.signals(p, **{**free, "volume_cap": 3.0})["weights"]["A0"]
    assert plain.iloc[151] == 0 and plain.iloc[152] > 0              # decided at 151, filled 152
    assert (capped.iloc[151:157] == 0).all()


def test_valid_rejects_what_the_book_cannot_express():
    assert PR.valid({})
    for bad in ({"q_in": 0.6}, {"q_out": 0.05}, {"q_out": 0.95}, {"peers": "industry"},
                {"scale": "rank"}, {"scale_window": 2},
                {"gross_stress": 1.2}, {"gross_calm": -0.1}, {"slots": 0},
                {"si_mid": 0.3, "si_high": 0.2}, {"formation": 0}, {"n_clusters": 1}):
        assert not PR.valid(bad), bad


def test_vol_scaling_ranks_a_move_unusual_for_its_name_as_more_extreme():
    rng = np.random.default_rng(5)
    n = 200
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="UTC")
    rets = (np.repeat(rng.normal(0.0, 0.01, (n, 3)), 6, axis=1)
            + rng.normal(0.0, 0.002, (n, len(NAMES))))
    rets[:, NAMES.index("A0")] += rng.normal(0.0, 0.03, n)          # A0: a noisy name
    rets[140, NAMES.index("A0")] -= 0.06                             # ~1 of its own sigmas
    rets[140, NAMES.index("A1")] -= 0.04                             # ~20 of A1's
    rets[0] = 0.0
    close = pd.DataFrame(50.0 * np.cumprod(1.0 + rets, axis=0), index=idx, columns=NAMES)
    p = Panel(close=close, open_=close.copy(), high=close * 1.01, low=close * 0.99,
              volume=pd.DataFrame(1e6, index=idx, columns=NAMES), bar="1d")
    raw, scaled = _compose(p), _compose(p, scale="vol")
    assert raw["signal"]["A0"].iloc[141] < raw["signal"]["A1"].iloc[141]
    assert scaled["signal"]["A1"].iloc[141] < scaled["signal"]["A0"].iloc[141]
    assert scaled["rank"]["A1"].iloc[141] == pytest.approx(1 / 18)
    daily = raw["move"] * 0 + p.close.pct_change() - pd.DataFrame(
        S._peer_mean(p.close.pct_change().to_numpy(), raw["labels"], 2), index=idx,
        columns=NAMES)
    sigma = daily.shift(1).rolling(63, min_periods=31).std()
    assert scaled["signal"]["A1"].iloc[141] == pytest.approx(
        raw["signal"]["A1"].iloc[141] / sigma["A1"].iloc[141])
