"""Engine helpers (pure) and full vectorbt runs (skipped if not installed)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_simple import backtest as bt
from trading_simple.config import CostSettings, Settings
from trading_simple.data import Panel


# --- pure helpers ------------------------------------------------------------
def test_sma_strict_warmup(panel):
    out = bt.sma(panel.close, 5)
    assert out.iloc[:4].isna().all().all() and not np.isnan(out.iloc[4, 0])


def test_next_bar_shifts_forward(panel):
    raw = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
    raw.iloc[10] = True
    shifted = bt.next_bar(raw)
    assert not shifted.iloc[10].any() and shifted.iloc[11].all()


def test_exit_after_and_validation(panel):
    entries = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
    entries.iloc[10] = True
    assert bt.exit_after(entries, 3).iloc[13].all()
    with pytest.raises(ValueError, match="horizon"):
        bt.exit_after(entries, 0)


def test_atr_positive_after_warmup(panel):
    out = bt.atr(panel, 14)
    assert out.iloc[:13].isna().all().all()
    assert (out.iloc[20:] > 0).all().all()


def test_volume_profile_time_of_day_baseline():
    # Two bars per day at 09:31 and 12:30 local; 12:30 always half the volume.
    days = pd.date_range("2026-03-02", periods=15, freq="B", tz="America/New_York")
    idx = pd.DatetimeIndex([d + off for d in days
                            for off in (pd.Timedelta("9h31min"), pd.Timedelta("12h30min"))])
    vol = pd.DataFrame({"AAA": [1000.0, 500.0] * 15}, index=idx.tz_convert("UTC"))
    base = bt.volume_profile(vol, "America/New_York", days=5, min_obs=3)
    tail = base.iloc[-2:]["AAA"]
    assert tail.iloc[0] == 1000.0 and tail.iloc[1] == 500.0  # per time-of-day, not blended
    assert base.iloc[:6].isna().all().all()                  # warm-up + current-day exclusion


def test_volume_profile_smoothing_widens_the_slot():
    # Day 1 prints [0, 3000, 0] over three minutes; smooth=1 averages ±1 minute
    # within the session, so day 2's baseline sees [1500, 1000, 1500].
    tz = "America/New_York"
    idx = pd.DatetimeIndex([pd.Timestamp(f"2026-06-0{d} 09:30", tz=tz) + pd.Timedelta(minutes=m)
                            for d in (1, 2) for m in range(3)]).tz_convert("UTC")
    vol = pd.DataFrame({"AAA": [0.0, 3000.0, 0.0, 500.0, 500.0, 500.0]}, index=idx)
    base = bt.volume_profile(vol, tz, days=1, min_obs=1, smooth=1)
    assert base["AAA"].iloc[3:].tolist() == [1500.0, 1000.0, 1500.0]
    assert base["AAA"].iloc[:3].isna().all()              # day 1 has no prior session


def test_session_mask_and_end_block_session_edges():
    tz = "America/New_York"
    idx = pd.DatetimeIndex([pd.Timestamp(f"2026-06-0{d} 09:30", tz=tz) + pd.Timedelta(minutes=m)
                            for d in (1, 2) for m in range(60)]).tz_convert("UTC")
    ok = bt.session_mask(idx, tz, skip_open=10, skip_close=10)
    day1 = ok.iloc[:60]
    assert not day1.iloc[:10].any() and not day1.iloc[50:].any()   # edges blocked
    assert day1.iloc[10:50].all()                                  # core tradable
    end = bt.session_end(idx, tz)
    assert end.iloc[59] and end.iloc[119] and end.sum() == 2       # one final bar per day


def test_leg_costs_arithmetic():
    fees, slip = bt.leg_costs(CostSettings())  # 500 shares @ $20 => $2.50 commission
    assert fees == pytest.approx(2.5 / 10_000 + 1.0 / 1e4)   # commission + half spread
    assert slip == pytest.approx(1.0 / 1e4)


def test_grid_from_config_product_and_valid():
    grid = bt.grid_from_config({"a": [1, 2], "b": [10, 20]}, valid=lambda p: p["a"] < 2)
    assert grid == [{"a": 1, "b": 10}, {"a": 1, "b": 20}]


def test_best_combo_ignores_nan():
    stats = pd.DataFrame({"fast": [1, 2], "excess_cagr": [np.nan, 0.5], "n_trades": [0, 3]})
    assert bt.best_combo(stats)["fast"] == 2                    # defaults to excess_cagr
    with pytest.raises(ValueError, match="NaN"):
        bt.best_combo(stats.assign(excess_cagr=np.nan))


def test_best_combo_strips_every_metric_column():
    """Metric columns must never leak into the returned parameter dict."""
    stats = pd.DataFrame({"fast": [1], "sharpe": [0.5], "total_return": [0.2], "cagr": [0.1],
                          "excess_cagr": [0.1], "info_ratio": [0.3], "n_trades": [7],
                          "trades_per_year": [3.5], "years": [2.0], "max_drawdown": [-0.1],
                          "win_rate": [0.5], "pct_profitable": [0.6]})
    assert bt.best_combo(stats) == {"fast": 1}


def test_cagr_is_window_length_invariant():
    """The point of the per-year clock: a cumulative return rewards a long window.

    The same 10%-a-year strategy compounds to +10% over one year and +46% over four. A
    cumulative excess return would rank the four-year run as four times better than the
    one-year run — which is exactly how a walk-forward fold of unequal length gets picked
    for being long rather than for being good.
    """
    assert bt._cagr(0.10, 1.0) == pytest.approx(0.10)
    assert bt._cagr(1.10 ** 4 - 1, 4.0) == pytest.approx(0.10)   # same rate, 4x the window
    assert bt._cagr(1.10 ** 4 - 1, 4.0) != pytest.approx(1.10 ** 4 - 1)   # ...unlike the total
    assert np.isnan(bt._cagr(-1.0, 2.0))                         # a wipeout has no rate


# --- full engine runs (need vectorbt) -----------------------------------------
vbt = pytest.importorskip("vectorbt")


def _settings() -> Settings:
    return Settings()


def _one_spike_signals(panel):
    """One long entry on AAA at bar 41 (spike at 40, next-bar execution)."""
    entries = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
    entries.iloc[41, 0] = True
    return {"entries": entries, "exits": bt.exit_after(entries, 5)}


def test_run_executes_engineered_trade(panel):
    pf = bt.run(panel, _one_spike_signals(panel), fees=0.0, slippage=0.0)
    trades = bt.trade_records(pf)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["symbol"] == "AAA"
    assert t["entry_ts"] == panel.close.index[41]
    assert t["entry_price"] == pytest.approx(panel.open_.iloc[41, 0])  # next-bar open fill
    assert t["pnl"] > 0                                                # AAA trends up


def test_run_with_trailing_stop_caps_loss(panel):
    # Short the up-trending AAA with a tight 1% trailing stop: the stop must
    # exit quickly and keep the loss small.
    short_entries = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
    short_entries.iloc[5, 0] = True
    pf = bt.run(panel, {"short_entries": short_entries, "sl_stop": 0.01, "sl_trail": True},
                fees=0.0, slippage=0.0)
    trades = bt.trade_records(pf)
    t = trades[trades["symbol"] == "AAA"].iloc[0]
    assert t["exit_ts"] < panel.close.index[-1]          # stopped out, not end-of-test
    assert t["return"] > -0.05                           # loss capped near the stop


def test_portfolio_stats_and_equity(panel):
    pf = bt.run(panel, _one_spike_signals(panel), fees=0.0, slippage=0.0, init_cash=10_000)
    stats = bt.portfolio_stats(pf)
    assert set(stats) == {"total_return", "cagr", "sharpe", "max_drawdown", "win_rate",
                          "pct_profitable", "n_trades", "trades_per_year", "years"}
    assert stats["n_trades"] == 1
    curve = bt.equity_curve(pf, init_cash=10_000)
    assert curve["value"].iloc[0] == pytest.approx(20_000)  # 2 symbols x 10k


def test_symbol_stats_per_symbol_breakdown(panel):
    pf = bt.run(panel, _one_spike_signals(panel), fees=0.0, slippage=0.0, init_cash=10_000)
    per = bt.symbol_stats(pf)
    assert list(per.index) == ["AAA", "BBB"]                # sorted by total_return desc
    assert per.loc["AAA", "n_trades"] == 1 and per.loc["BBB", "n_trades"] == 0
    assert per.loc["AAA", "total_return"] > 0               # the engineered winner


def test_sweep_one_row_per_combo(panel):
    from trading_simple import strategies

    strat = strategies.load("volume_momentum")
    grid = [{"fast": 5, "slow": 20, "vol_window": 10, "threshold": 3.0, "horizon": h}
            for h in (3, 5, 8)]
    stats = bt.sweep(strat, panel, grid, settings=_settings(), chunk_size=2)
    assert len(stats) == 3
    assert stats["n_trades"].tolist() == [2, 2, 2]  # AAA long + BBB short per combo


def test_sweep_with_stops_matches_single_runs(panel):
    from trading_simple import strategies

    strat = strategies.load("donchian_breakout")
    grid = [{"entry_window": 10, "atr_window": 7, "atr_mult": m} for m in (1.5, 3.0)]
    stats = bt.sweep(strat, panel, grid, settings=_settings(), chunk_size=2)
    single = bt.portfolio_stats(
        bt.run_strategy(strat, panel, grid[0], settings=_settings()))
    assert stats.iloc[0]["n_trades"] == single["n_trades"]
    assert stats.iloc[0]["total_return"] == pytest.approx(single["total_return"])


# --- benchmark-relative metrics ------------------------------------------------
def test_benchmark_is_buy_and_hold_and_beats_a_flat_strategy(panel):
    """Zero excess_cagr and zero info_ratio means "exactly the benchmark".

    The point of these metrics: a long-only strategy in a rising market posts a
    positive Sharpe just by being invested, so only a benchmark-RELATIVE number can
    separate skill from exposure. Running the benchmark against itself must therefore
    score exactly zero on both.
    """
    weights = pd.DataFrame(1.0 / len(panel.symbols), index=panel.close.index,
                           columns=panel.close.columns)     # own everything, forever
    pf = bt.run(panel, {"weights": weights}, fees=0.0, slippage=0.0)
    stats = bt.portfolio_stats(pf)
    active = bt.active_stats(pf, panel, fees=0.0, slippage=0.0,
                             total_return=stats["total_return"])
    assert abs(active["excess_cagr"]) < 1e-12
    assert abs(active["info_ratio"]) < 1e-9

    # A strategy that never trades earns nothing, so it trails a rising benchmark.
    bench_return, _ = bt.benchmark(panel, fees=0.0, slippage=0.0)
    if bench_return > 0:
        flat = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
        pf0 = bt.run(panel, {"entries": flat, "exits": flat}, fees=0.0, slippage=0.0)
        s0 = bt.portfolio_stats(pf0)
        a0 = bt.active_stats(pf0, panel, fees=0.0, slippage=0.0,
                             total_return=s0["total_return"])
        assert a0["excess_cagr"] < 0


def test_benchmark_matches_the_mean_of_the_per_symbol_returns(panel):
    """The benchmark moved to portfolio mode; its NUMBER must not move with it.

    Buying 1/N of one shared pot in every name and never trading is *identically* the
    mean of the per-symbol buy-and-hold returns — which is what `portfolio_stats` reports
    in per-symbol mode. Pinning that proves the rebuilt path is sane, and proves the six
    per-symbol strategies' excess_cagr / info_ratio did not silently shift underneath
    them when the benchmark changed construction.

    The identity is EXACT only without costs, and the reason is worth knowing: in a
    shared account the orders fill **sequentially**, so the first name's fees shrink the
    book before the second name's target percent is computed off it. The result is a
    slight tilt toward whichever name fills first — real, but bounded by the total fee
    (a fraction of a basis point at this repo's cost levels), and now paid by the
    strategy and the benchmark alike, which is the point.
    """
    def per_symbol_mean(fees, slippage):
        entries = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
        entries.iloc[0] = True
        exits = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
        pf = bt.run(panel, {"entries": entries, "exits": exits}, fees=fees, slippage=slippage)
        return bt.portfolio_stats(pf)["total_return"]

    free, _ = bt.benchmark(panel, fees=0.0, slippage=0.0)
    assert free == pytest.approx(per_symbol_mean(0.0, 0.0), rel=1e-9)      # exact, cost-free

    costed, _ = bt.benchmark(panel, fees=0.001, slippage=0.001)            # 10x real costs
    assert costed == pytest.approx(per_symbol_mean(0.001, 0.001), abs=1e-3)


# --- portfolio mode: one shared cash account ------------------------------------
def _rotate(panel, first: str, second: str, at: int, slots: int = 1) -> pd.DataFrame:
    """Target weights that hold `first`, then rotate the whole book into `second`."""
    weights = pd.DataFrame(0.0, index=panel.close.index, columns=panel.close.columns)
    weights.iloc[1:at, weights.columns.get_loc(first)] = 1.0 / slots
    weights.iloc[at:, weights.columns.get_loc(second)] = 1.0 / slots
    return weights


def test_portfolio_mode_shares_one_cash_account(panel):
    """The whole book is init_cash — NOT init_cash per symbol.

    This is the defect the mode exists to remove. With a cash account per symbol, a
    100-name universe starts with $1,000,000 and the money attached to a name the
    strategy is flat in can never be deployed anywhere else, so a strategy holding a
    third of the names is structurally a third invested against a fully-invested
    benchmark and cannot beat it however well it picks.
    """
    weights = _rotate(panel, "AAA", "BBB", at=20)
    pf = bt.run(panel, {"weights": weights}, fees=0.0, slippage=0.0, init_cash=10_000)
    assert bt.is_grouped(pf)
    curve = bt.equity_curve(pf, init_cash=10_000)
    assert curve["value"].iloc[0] == pytest.approx(10_000)   # one book, not 2 x 10k
    assert curve["return"].iloc[0] == pytest.approx(0.0)     # ...and the base is pre-fee


def test_portfolio_stats_are_not_nan_under_grouping(panel):
    """Regression guard: grouped metrics used to fail SILENTLY, not loudly.

    Under `group_by=True` vectorbt returns scalars where the per-symbol path returns a
    Series. `_metric_frame`'s `safe()` swallows the shape mismatch in a bare `except`, so
    every metric came back NaN — with `n_trades` reading zero — and would have quietly
    poisoned `evaluate`, `sweep`, `best_combo` and every Optuna trial with no error
    anywhere. Nothing about that is visible without this test.
    """
    weights = _rotate(panel, "AAA", "BBB", at=20)
    pf = bt.run(panel, {"weights": weights}, fees=0.0, slippage=0.0, init_cash=10_000)
    stats = bt.portfolio_stats(pf)
    assert stats["n_trades"] > 0
    for name in ("total_return", "cagr", "sharpe", "max_drawdown", "win_rate"):
        assert np.isfinite(stats[name]), f"{name} is NaN under grouping"
    assert bt.symbol_stats(pf)["total_return"].notna().any()   # per-symbol view survives


def test_rotation_sells_to_fund_a_buy(panel):
    """The capital constraint itself: with one slot, buying BBB requires selling AAA.

    Both legs must land on the SAME bar. vectorbt only manages that with
    `call_seq="auto"`, which orders the sell ahead of the buy; without it the buy is
    partially rejected for want of cash that has not arrived yet, and the shortfall sits
    idle — reintroducing, quietly, the exact drag this mode removes.
    """
    weights = _rotate(panel, "AAA", "BBB", at=20)
    pf = bt.run(panel, {"weights": weights}, fees=0.0, slippage=0.0, init_cash=10_000)
    orders = pf.orders.records_readable
    rotation = orders[orders["Timestamp"] == panel.close.index[20]]
    assert set(rotation["Column"]) == {"AAA", "BBB"}
    assert set(rotation["Side"]) == {"Sell", "Buy"}
    assert float(pf.cash().iloc[-1]) == pytest.approx(0.0, abs=1e-6)   # nothing left idle


def test_no_orders_on_non_rotation_bars(panel):
    """Weights drift between rotations; the book is not re-trimmed every bar.

    Target-percent sizing acts on every bar it is given a number for, so a dense weight
    frame silently rebalances daily — hundreds of tiny orders whose fees are charged
    straight against the benchmark. `rotation_targets` blanks the unchanged bars, which
    is what keeps trading event-driven rather than on a hidden daily calendar.
    """
    weights = _rotate(panel, "AAA", "BBB", at=20)
    pf = bt.run(panel, {"weights": weights}, fees=0.0, slippage=0.0, init_cash=10_000)
    stamps = pf.orders.records_readable["Timestamp"]
    assert set(stamps) == {panel.close.index[1], panel.close.index[20]}   # entry + rotation

    dense = bt.rotation_targets(weights)
    assert dense.notna().any(axis=1).sum() == 3      # bar 0 (flat), bar 1 (buy), bar 20


def test_held_name_is_not_trimmed_when_a_peer_rotates():
    """A held position must not be re-trimmed just because another name rotates.

    `rotation_targets` masks per NAME, not per row: AAA is held at a constant slot
    across the whole run while BBB rotates out to CCC at bar 20. With a whole-row mask
    the entire book would be re-emitted on bar 20 and target-percent would trim AAA back
    to its slot of the (drifted) book value — a real partial sell of a position the
    strategy only meant to hold. Per-cell masking leaves AAA's cell NaN there, so AAA
    trades exactly once (its entry) and is never trimmed. Prices drift so that, were the
    trim happening, it would place a nonzero AAA order.
    """
    idx = pd.date_range("2024-01-02", periods=30, freq="B", tz="UTC")
    close = pd.DataFrame({"AAA": np.linspace(100.0, 160.0, 30),   # drifts up strongly...
                          "BBB": np.linspace(100.0, 90.0, 30),
                          "CCC": np.linspace(100.0, 130.0, 30)}, index=idx)
    p = Panel(close=close, open_=close * 0.999, high=close * 1.01, low=close * 0.99,
              volume=pd.DataFrame(1e6, index=idx, columns=close.columns), bar="1d")
    w = pd.DataFrame(0.0, index=idx, columns=close.columns)
    w.iloc[1:, w.columns.get_loc("AAA")] = 1.0 / 3      # AAA held the whole run
    w.iloc[1:20, w.columns.get_loc("BBB")] = 1.0 / 3    # BBB rotates out to CCC at bar 20
    w.iloc[20:, w.columns.get_loc("CCC")] = 1.0 / 3

    masked = bt.rotation_targets(w)
    assert not np.isfinite(masked["AAA"].iloc[20])       # AAA gets NO target on the rotation bar
    assert masked["BBB"].iloc[20] == 0.0 and masked["CCC"].iloc[20] == 1.0 / 3

    pf = bt.run(p, {"weights": w}, fees=0.0, slippage=0.0, init_cash=10_000)
    orders = pf.orders.records_readable
    aaa = orders[orders["Column"] == "AAA"]
    assert list(aaa["Timestamp"]) == [idx[1]]            # bought once, never trimmed thereafter
    rot = orders[orders["Timestamp"] == idx[20]]
    assert set(rot["Column"]) == {"BBB", "CCC"}          # only the rotating pair trades


def test_sweep_portfolio_matches_evaluate(panel):
    """A portfolio-mode sweep must group by COMBO — each combo its own shared book.

    Column-stacking puts every combo in one vectorbt call, so a plain `group_by=True`
    would pool them all into a single giant cash account and let combo 0's positions fund
    combo 1's. Grouping on the combo level keeps them separate; agreeing with `evaluate`
    on the same parameters is the proof.
    """
    from trading_simple import strategies

    strat = strategies.load("informed_flow")

    def weights_only(p, *, slot: float, **_):
        w = pd.DataFrame(0.0, index=p.close.index, columns=p.close.columns)
        w.iloc[5:, 0] = slot                       # hold AAA at `slot` of the book
        return {"weights": w}

    strat = replace(strat, signals=weights_only, params={"slot": 0.5})
    grid = [{"slot": 0.5}, {"slot": 1.0}]
    stats = bt.sweep(strat, panel, grid, settings=_settings(), chunk_size=2)
    for row, combo in zip(stats.to_dict("records"), grid, strict=True):
        single = bt.evaluate(strat, panel, combo, settings=_settings())
        assert row["total_return"] == pytest.approx(single["total_return"], rel=1e-9)
        assert row["excess_cagr"] == pytest.approx(single["excess_cagr"], rel=1e-6)
    # A bigger slot in the same shared book must move the number — proof the combos are
    # not silently sharing one pot.
    assert stats["total_return"].iloc[0] != pytest.approx(stats["total_return"].iloc[1])


def test_benchmark_cache_is_keyed_by_panel_not_identity(panel):
    """A different panel must not silently reuse another panel's cached benchmark.

    The cache is keyed by the panel's index/shape/symbols rather than by ``id()``,
    precisely because walk-forward slicing creates and discards panels — a recycled
    id would otherwise hand back a benchmark computed over a different window.
    """
    n = len(panel.close) // 2
    half = replace(panel, close=panel.close.iloc[:n], open_=panel.open_.iloc[:n],
                   high=panel.high.iloc[:n], low=panel.low.iloc[:n],
                   volume=panel.volume.iloc[:n])
    _, curve_full = bt.benchmark(panel, fees=0.0, slippage=0.0)
    _, curve_half = bt.benchmark(half, fees=0.0, slippage=0.0)
    assert len(curve_full) == len(panel.close)
    assert len(curve_half) == n < len(curve_full)      # not the stale cached curve


# --- cash-shared signal mode: one wallet, fractional entries -------------------
def _cash_frac_signals(panel, frac=0.25):
    """Both symbols enter at bar 5 against one shared wallet."""
    entries = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
    entries.iloc[5] = True
    return {"entries": entries, "exits": bt.exit_after(entries, 30), "cash_frac": frac}


def test_cash_frac_is_one_wallet_never_overdeployed(panel):
    pf = bt.run(panel, _cash_frac_signals(panel), fees=0.0, slippage=0.0, init_cash=10_000)
    assert bt.is_grouped(pf)
    # init_cash is the WHOLE book, not per symbol...
    assert bt.equity_curve(pf, init_cash=10_000)["value"].iloc[0] == pytest.approx(10_000)
    # ...and total invested can never exceed the wallet's equity.
    invested = pf.asset_value(group_by=False).sum(axis=1)
    assert (invested <= pf.value() + 1e-9).all()
    # Same-bar entries split the free cash: 25% of 10k, then 25% of the remainder.
    trades = bt.trade_records(pf)
    notionals = sorted(trades["size"] * trades["entry_price"])
    assert notionals == [pytest.approx(1_875.0), pytest.approx(2_500.0)]


def test_cash_frac_stops_still_fire_when_grouped(panel):
    # Long the down-trending BBB inside the shared book with a tight 2% stop.
    entries = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
    entries.iloc[5, 1] = True
    pf = bt.run(panel, {"entries": entries, "sl_stop": 0.02, "cash_frac": 0.25},
                fees=0.0, slippage=0.0)
    t = bt.trade_records(pf).iloc[0]
    assert t["exit_ts"] < panel.close.index[-1]          # stopped out, not end-of-test
    assert t["return"] > -0.06                           # loss capped near the stop


def test_cash_frac_short_entry_consumes_the_wallet(panel):
    # Shorting from flat is well-defined under percent sizing: the entry deploys a
    # fraction of the wallet's free short capacity and counts against the book.
    short_entries = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
    short_entries.iloc[5, 1] = True                      # short the down-trending BBB
    pf = bt.run(panel, {"short_entries": short_entries,
                        "short_exits": bt.exit_after(short_entries, 30),
                        "cash_frac": 0.25},
                fees=0.0, slippage=0.0, init_cash=10_000)
    assert bt.is_grouped(pf)
    trades = bt.trade_records(pf)
    assert len(trades) == 1 and trades.iloc[0]["direction"] == "Short"
    assert trades.iloc[0]["pnl"] > 0                     # BBB trends down; the short wins
    gross = pf.asset_value(group_by=False).abs().sum(axis=1)
    assert gross.max() > 0.0
    assert (gross <= pf.value() + 1e-9).all()            # exposure never exceeds the book


def test_cash_frac_opposite_entry_closes_not_reverses(panel):
    # Percent sizing cannot reverse a position (vectorbt raises); the wallet mode
    # therefore closes on an opposite entry (upon_opposite_entry="close") instead.
    idx, cols = panel.close.index, panel.close.columns
    entries = pd.DataFrame(False, index=idx, columns=cols)
    entries.iloc[5, 0] = True                            # long AAA at bar 5...
    short_entries = pd.DataFrame(False, index=idx, columns=cols)
    short_entries.iloc[15, 0] = True                     # ...opposite entry at bar 15
    pf = bt.run(panel, {"entries": entries, "short_entries": short_entries,
                        "cash_frac": 0.25}, fees=0.0, slippage=0.0)
    trades = bt.trade_records(pf)
    assert list(trades["direction"]) == ["Long"]         # closed, never flipped short
    assert trades.iloc[0]["exit_ts"] == idx[15]


def test_run_without_cash_frac_stays_per_symbol(panel):
    pf = bt.run(panel, _one_spike_signals(panel), fees=0.0, slippage=0.0, init_cash=10_000)
    assert not bt.is_grouped(pf)
    assert bt.equity_curve(pf, init_cash=10_000)["value"].iloc[0] == pytest.approx(20_000)


def test_cash_frac_conflicts_with_weights(panel):
    w = pd.DataFrame(0.5, index=panel.close.index, columns=panel.close.columns)
    with pytest.raises(ValueError, match="cash_frac"):
        bt.run(panel, {"weights": w, "cash_frac": 0.25}, fees=0.0, slippage=0.0)


def test_sweep_cash_frac_matches_evaluate(panel):
    """A cash-shared sweep must group per COMBO and agree with evaluate()."""
    from trading_simple import strategies

    strat = strategies.load("support_bounce")   # host shell; signals get replaced

    def sig(p, *, hold: int, **_):
        entries = pd.DataFrame(False, index=p.close.index, columns=p.close.columns)
        entries.iloc[5] = True
        return {"entries": entries, "exits": bt.exit_after(entries, hold),
                "cash_frac": 0.25}

    strat = replace(strat, signals=sig, params={"hold": 10})
    grid = [{"hold": 10}, {"hold": 30}]
    stats = bt.sweep(strat, panel, grid, settings=_settings(), chunk_size=2)
    for row, combo in zip(stats.to_dict("records"), grid, strict=True):
        single = bt.evaluate(strat, panel, combo, settings=_settings())
        assert row["total_return"] == pytest.approx(single["total_return"], rel=1e-9)
        assert row["n_trades"] == single["n_trades"]
