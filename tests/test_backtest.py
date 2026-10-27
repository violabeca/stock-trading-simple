"""Engine helpers (pure) and full vectorbt runs (skipped if not installed)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_panel
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


def test_panel_costs_prices_commission_at_the_panel_median():
    """Per-share commission must be priced at the panel, not a hard-coded $20.

    make_panel's median-of-medians is $100: 100 shares of a $10k order cost
    $0.50, so the $1 minimum binds — 1 bp/leg, not the $20-name's 2.5 bp.
    """
    fees, slip = bt.panel_costs(CostSettings(), make_panel())
    assert fees == pytest.approx(1.0 / 10_000 + 1.0 / 1e4)   # min commission + half spread
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


def test_time_stop_is_hard_even_when_a_fresh_signal_lands_on_it(panel):
    """An exit and an entry signal on the same bar must EXIT, not cancel out.

    vectorbt's default conflict mode ("ignore") dropped both signals: a horizon
    exit landing on a bar that also carried a fresh entry signal was swallowed,
    so the position ran past its supposed hard time stop (and the entry
    evaporated too). Exit-wins makes exit_after a real stop; the same-bar
    re-entry is still dropped — from_signals cannot exit and re-enter in one bar.
    """
    entries = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
    entries.iloc[5, 0] = True
    entries.iloc[13, 0] = True                      # fresh signal ON the time-stop bar
    exits = bt.exit_after(entries, 8)               # schedules exits at bars 13 and 21
    pf = bt.run(panel, {"entries": entries, "exits": exits}, fees=0.0, slippage=0.0)
    t = bt.trade_records(pf)
    t = t[t["symbol"] == "AAA"]
    assert len(t) == 1                              # no phantom second position
    assert t.iloc[0]["entry_ts"] == panel.close.index[5]
    assert t.iloc[0]["exit_ts"] == panel.close.index[13]   # the hard stop, not bar 21


def test_stop_gap_through_fills_at_the_open_not_the_stop():
    """A bar that gaps through a stop must fill at its (worse) true open.

    vectorbt's defaults are optimistic here: without ``open=`` it substitutes
    close for open in stop-hit detection, and ``stoplimit`` fills AT the stop
    level even when the bar opened far beyond it. run() now passes the open and
    ``stopmarket``, so the overnight collapse below prints at 85, not at 98.
    """
    idx = pd.date_range("2024-01-02", periods=20, freq="B", tz="UTC")
    close = pd.DataFrame({"AAA": 100.0}, index=idx)
    close.iloc[10:] = 85.0                                   # collapse at bar 10
    p = Panel(close=close, open_=close.copy(), high=close + 0.5, low=close - 0.5,
              volume=pd.DataFrame(1_000.0, index=idx, columns=["AAA"]), bar="1d")
    entries = pd.DataFrame(False, index=idx, columns=["AAA"])
    entries.iloc[5] = True                                   # fill at open(5) = 100
    pf = bt.run(p, {"entries": entries, "sl_stop": 0.02},    # stop level = 98
                fees=0.0, slippage=0.0)
    t = bt.trade_records(pf).iloc[0]
    assert t["exit_ts"] == idx[10]
    assert t["exit_price"] == pytest.approx(85.0)            # the open, NOT the 98 stop


# --- pessimistic fills ---------------------------------------------------------
def test_pessimistic_fill_buys_at_the_high_and_sells_at_the_low(panel):
    """Every buy prints at its bar's high, every sell at its low — in both engine modes."""
    pf = bt.run(panel, _one_spike_signals(panel), fees=0.0, slippage=0.0, fill="pessimistic")
    t = bt.trade_records(pf).iloc[0]
    assert t["entry_price"] == pytest.approx(panel.high.iloc[41, 0])
    assert t["exit_price"] == pytest.approx(panel.low.iloc[46, 0])
    # The configured slippage still applies on top of the worst side.
    slipped = bt.trade_records(bt.run(panel, _one_spike_signals(panel), fees=0.0,
                                      slippage=0.001, fill="pessimistic")).iloc[0]
    assert slipped["entry_price"] > panel.high.iloc[41, 0]
    assert slipped["exit_price"] < panel.low.iloc[46, 0]

    w = pd.DataFrame(0.0, index=panel.close.index, columns=panel.close.columns)
    w.iloc[5:20, 0] = 0.5                          # portfolio mode: buy bar 5, sell bar 20
    t = bt.trade_records(bt.run(panel, {"weights": w}, fees=0.0, slippage=0.0,
                                fill="pessimistic")).iloc[0]
    assert t["entry_price"] == pytest.approx(panel.high.iloc[5, 0])
    assert t["exit_price"] == pytest.approx(panel.low.iloc[20, 0])
    with pytest.raises(ValueError, match="fill"):
        bt.run(panel, {"weights": w}, fees=0.0, slippage=0.0, fill="close")


def test_pessimistic_fill_prices_a_stop_leg_at_the_bar_low():
    """A stop exit is an order like any other: under pessimistic it prints at the bar's low."""
    idx = pd.date_range("2024-01-02", periods=20, freq="B", tz="UTC")
    close = pd.DataFrame({"AAA": 100.0}, index=idx)
    close.iloc[10:] = 85.0                                   # collapse at bar 10
    p = Panel(close=close, open_=close.copy(), high=close + 0.5, low=close - 0.5,
              volume=pd.DataFrame(1_000.0, index=idx, columns=["AAA"]), bar="1d")
    entries = pd.DataFrame(False, index=idx, columns=["AAA"])
    entries.iloc[5] = True
    pf = bt.run(p, {"entries": entries, "sl_stop": 0.02}, fees=0.0, slippage=0.0,
                fill="pessimistic")
    t = bt.trade_records(pf).iloc[0]
    assert t["entry_price"] == pytest.approx(100.5)          # high of the entry bar
    assert t["exit_ts"] == idx[10]
    assert t["exit_price"] == pytest.approx(84.5)            # low of the gap bar


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
    assert set(stats) == {"total_return", "cagr", "sharpe", "volatility", "max_drawdown",
                          "win_rate", "pct_profitable", "n_trades", "trades_per_year", "years"}
    assert stats["n_trades"] == 1
    curve = bt.equity_curve(pf, init_cash=10_000)
    assert curve["value"].iloc[0] == pytest.approx(20_000)  # 2 symbols x 10k


def test_volatility_is_the_books_own_annualized_risk(panel):
    """The annual risk rate: std of the book's per-bar returns on the bars-per-year clock."""
    idx = panel.close.index
    half = pd.DataFrame(0.5, index=idx, columns=panel.close.columns)
    pf = bt.run(panel, {"weights": half}, fees=0.0, slippage=0.0)
    expected = pf.returns().std() * np.sqrt(len(idx) / bt._years(idx))
    assert bt.portfolio_stats(pf)["volatility"] == pytest.approx(expected)
    cash = bt.run(panel, {"weights": half * 0.0}, fees=0.0, slippage=0.0)
    assert bt.portfolio_stats(cash)["volatility"] == pytest.approx(0.0)   # cash does not move


def _slow_starter(p, **_):
    """A book that needs 30 bars of history before it holds anything."""
    ready = p.close.rolling(30).mean().notna()
    return {"weights": bt.next_bar(ready).astype(float) / len(p.symbols)}


def test_score_from_warm_starts_the_signals_and_scores_only_the_window(panel):
    """An out-of-sample window must inherit its indicators' history, not restart them.

    Slicing the panel to the window restarts every rolling window at its first bar, so a
    30-bar indicator never warms up inside a 20-bar window and the book sits in cash for
    the whole test. `score_from` computes the signals on the whole panel and trades only the
    window: the book is invested from the window's first bar, and every metric (years,
    benchmark, excess) belongs to the window alone.
    """
    from trading_simple import strategies

    strat = replace(strategies.load("informed_flow"), signals=_slow_starter, params={})
    start = panel.close.index[40]
    warm = bt.run_strategy(strat, panel, settings=_settings(), score_from=start)
    assert warm.wrapper.index[0] == start
    assert warm.asset_value(group_by=False).iloc[0].sum() > 0          # invested on bar one

    window, _ = bt._score_window(panel, {}, start)
    cold = bt.run_strategy(strat, window, settings=_settings())         # the old, sliced way
    assert cold.asset_value(group_by=False).sum().sum() == 0            # never warms up

    stats = bt.evaluate(strat, panel, {}, settings=_settings(), score_from=start)
    assert stats["years"] == pytest.approx(bt._years(window.close.index))
    fees, slip = bt.panel_costs(_settings().costs, window)
    manual = bt.run(window, {"weights": _slow_starter(panel)["weights"].loc[start:]},
                    fees=fees, slippage=slip)
    by_hand = bt.active_stats(manual, window, fees=fees, slippage=slip,
                              total_return=bt.portfolio_stats(manual)["total_return"])
    assert stats["excess_cagr"] == pytest.approx(by_hand["excess_cagr"], rel=1e-9)
    with pytest.raises(ValueError, match="score_from"):
        bt.run_strategy(strat, panel, settings=_settings(), score_from=panel.close.index[-1])


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
    assert abs(active["tracking_error"]) < 1e-9      # the benchmark never strays from itself

    # A strategy that never trades earns nothing, so it trails a rising benchmark.
    # AAA-only panel: the benchmark is unambiguously positive, so this half can't
    # silently no-op the way it would on the mixed AAA-up/BBB-down fixture.
    rising = make_panel(symbols=("AAA",))
    bench_return, _ = bt.benchmark(rising, fees=0.0, slippage=0.0)
    assert bench_return > 0
    flat = pd.DataFrame(False, index=rising.close.index, columns=rising.close.columns)
    pf0 = bt.run(rising, {"entries": flat, "exits": flat}, fees=0.0, slippage=0.0)
    s0 = bt.portfolio_stats(pf0)
    a0 = bt.active_stats(pf0, rising, fees=0.0, slippage=0.0,
                         total_return=s0["total_return"])
    assert a0["excess_cagr"] < 0
    assert a0["tracking_error"] > 0                   # sitting in cash strays from the market


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


def test_benchmark_buys_a_late_listed_name_at_its_listing():
    """A name with leading-NaN prices must still enter the buy-and-hold book.

    LiquidLeaders is screened from names listed TODAY but backfilled years further:
    a 2022 IPO on a 2021 panel starts with NaN prices. The dense constant-1/N frame
    put that name's only order on bar 0 — a NaN-price order vectorbt silently skips —
    so the name was never bought and its 1/N sat in cash forever, understating the
    hurdle every strategy is judged against. The masked frame emits the order on the
    name's first valid bar instead.
    """
    p = make_panel()
    for f in ("close", "open_", "high", "low", "volume"):
        frame = getattr(p, f)
        frame.iloc[:20, frame.columns.get_loc("BBB")] = np.nan   # BBB lists 20 bars late

    weights = pd.DataFrame(1.0 / len(p.symbols), index=p.close.index,
                           columns=p.close.columns)
    pf = bt.run(p, {"weights": weights.where(p.close.notna())},   # benchmark's construction
                fees=0.0, slippage=0.0)
    held = pf.asset_value(group_by=False)["BBB"].fillna(0.0)
    assert (held.iloc[:20] == 0.0).all()      # nothing held before the listing...
    assert held.iloc[20] > 0.0                # ...bought ON the first valid bar
    assert held.iloc[-1] > 0.0                # ...and held to the end

    # And the public entry point agrees: the staggered-panel benchmark ends more
    # invested than a book where BBB's 1/N is simply never deployed (its curve must
    # show BBB's post-listing decline, not flat cash).
    total, curve = bt.benchmark(p, fees=0.0, slippage=0.0)
    assert curve.iloc[21:].ne(0.0).any()
    aaa_only = p.close["AAA"].iloc[-1] / p.close["AAA"].iloc[0] - 1.0
    assert total < aaa_only / 2               # BBB's decline is IN the number, not idled away


def test_diagnostic_sharpe_is_on_the_bars_per_year_clock():
    """The diagnostic Sharpe must share info_ratio's clock: actual bars per year.

    vectorbt's own ``sharpe_ratio`` annualizes with a 365-calendar-day year — √365
    on daily bars that number ~252 a year (~20% high) and ~2.3x high on 1m bars.
    Pin the factor at √(bars/years) so a regression back to the wall-clock year
    cannot slip through.
    """
    index = pd.date_range("2024-01-01", "2024-12-31", freq="B", tz="UTC")   # ~1 year
    rets = pd.Series([0.001, -0.0005] * (len(index) // 2) + [0.001] * (len(index) % 2),
                     index=index)
    factor = float(bt._bar_sharpe(rets, index) / (rets.mean() / rets.std()))
    assert factor == pytest.approx(np.sqrt(len(index)), rel=0.02)   # ~16.1 on ~261 bars
    assert factor < np.sqrt(300)                                    # rules out sqrt(365)

    # Intraday, same formula: factor^2 * years == number of bars, exactly.
    intraday = pd.date_range("2026-03-02 14:30", periods=390, freq="1min", tz="UTC") \
                 .append(pd.date_range("2026-03-03 14:30", periods=390, freq="1min", tz="UTC"))
    rets2 = pd.Series([0.001, -0.0005] * 390, index=intraday)
    factor2 = float(bt._bar_sharpe(rets2, intraday) / (rets2.mean() / rets2.std()))
    assert factor2 ** 2 * bt._years(intraday) == pytest.approx(len(intraday), rel=1e-9)

    # Frame input -> per-column Series (the _metric_frame path).
    frame = pd.DataFrame({"AAA": rets, "BBB": -rets})
    out = bt._bar_sharpe(frame, index)
    assert list(out.index) == ["AAA", "BBB"] and out["AAA"] == pytest.approx(-out["BBB"])


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


def test_whole_shares_round_down_and_never_buy_a_name_priced_above_its_slot(panel):
    """A real account buys whole shares: each slot's remainder waits in cash, and a name
    priced above its slot is never bought at all. The default stays fractional."""
    scale = pd.Series({"AAA": 1.0, "BBB": 10.0})               # BBB trades at ~$1,000
    p = replace(panel, **{f: getattr(panel, f) * scale for f in ("close", "open_", "high", "low")})
    weights = pd.DataFrame(0.0, index=p.close.index, columns=p.close.columns)
    weights.iloc[1:] = 0.5                                     # two $500 slots of $1,000
    whole = bt.run(p, {"weights": weights}, fees=0.0, slippage=0.0, init_cash=1_000,
                   whole_shares=True)
    held = whole.assets()
    assert (held == held.round()).all().all()                  # whole shares only
    assert held["AAA"].iloc[-1] == 4                           # $500 buys 4 shares at ~$100
    assert (held["BBB"] == 0).all()                            # $500 cannot buy one $1,000 share
    assert whole.cash().iloc[-1] > 500                         # BBB's slot and AAA's remainder
    frac = bt.run(p, {"weights": weights}, fees=0.0, slippage=0.0, init_cash=1_000)
    assert frac.assets()["BBB"].iloc[-1] == pytest.approx(0.5, rel=0.01)   # fractional: bought


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


def _runaway_panel():
    """Three equal slots; AAA runs away 10x while BBB/CCC stay ~flat.

    With a 1/3 slot each, AAA's book weight climbs past 2/3 (2x its slot) as it rips, while
    the two flat names only ever drift BELOW their slot — so exactly one name is a concentration
    risk, which is what the retrim band is for.
    """
    idx = pd.date_range("2024-01-02", periods=50, freq="B", tz="UTC")
    close = pd.DataFrame({"AAA": np.linspace(100.0, 1000.0, 50),   # runs away 10x
                          "BBB": np.linspace(100.0, 104.0, 50),    # ~flat
                          "CCC": np.linspace(100.0, 96.0, 50)}, index=idx)   # ~flat
    p = Panel(close=close, open_=close * 0.999, high=close * 1.01, low=close * 0.99,
              volume=pd.DataFrame(1e6, index=idx, columns=close.columns), bar="1d")
    w = pd.DataFrame(0.0, index=idx, columns=close.columns)
    w.iloc[1:, :] = 1.0 / 3       # hold all three at equal slots for the whole run
    return p, w


def test_retrim_caps_a_runaway_winner():
    """retrim_k trims a held name back to its slot once it doubles it, so a pop cannot dominate.

    AAA rips 10x and, left alone, would grow to ~5/6 of the book (see the off-path test). With
    `retrim_k=2.0` the engine sells it back to its 1/3 slot the bar after its weight first reaches
    2/3, capping single-name concentration near the band. The two flat peers never breach 2x their
    slot, so they are bought once and never trimmed — the trim is per name, not a whole-book
    rebalance.
    """
    p, w = _runaway_panel()
    pf = bt.run(p, {"weights": w, "retrim_k": 2.0}, fees=0.0, slippage=0.0, init_cash=10_000)
    orders = pf.orders.records_readable

    aaa = orders[orders["Column"] == "AAA"]
    assert (aaa["Side"] == "Sell").any() and len(aaa) >= 2   # entry + at least one trim
    for peer in ("BBB", "CCC"):                              # never breach -> bought once, no trim
        assert list(orders[orders["Column"] == peer]["Side"]) == ["Buy"]
    weight = pf.asset_value(group_by=False)["AAA"] / pf.value()
    assert weight.max() < 0.72                               # held near the 2/3 band, not ~5/6


def test_retrim_off_lets_the_winner_run():
    """The default (no retrim_k) is unchanged: a held name drifts uncapped, bought once.

    The exact behavior `test_held_name_is_not_trimmed_when_a_peer_rotates` pins, on the runaway
    panel — proof the cap is strictly opt-in and off by default (so the benchmark and every other
    strategy are untouched).
    """
    p, w = _runaway_panel()
    pf = bt.run(p, {"weights": w}, fees=0.0, slippage=0.0, init_cash=10_000)  # retrim_k absent
    aaa = pf.orders.records_readable
    aaa = aaa[aaa["Column"] == "AAA"]
    assert list(aaa["Side"]) == ["Buy"]                     # bought once, never trimmed
    weight = pf.asset_value(group_by=False)["AAA"] / pf.value()
    assert weight.max() > 0.80                              # runs away past the 2x band, uncapped


def test_sweep_portfolio_retrim_matches_evaluate():
    """A portfolio sweep must apply the retrim cap identically to `evaluate`, per combo.

    The cap lives in both call sites (`_run_portfolio` and the sweep branch); if they drift,
    a swept number silently disagrees with the single run it should equal. Runs the same
    runaway book capped (k=2) and uncapped (k=None) as two combos and checks each against
    `evaluate` — and that the two differ, proving the cap actually fired inside the sweep.
    """
    from trading_simple import strategies

    strat = strategies.load("informed_flow")
    p, w = _runaway_panel()

    def capped(_panel, *, k, **__):
        return {"weights": w, "retrim_k": k}

    strat = replace(strat, signals=capped, params={"k": 2.0})
    grid = [{"k": 2.0}, {"k": None}]
    stats = bt.sweep(strat, p, grid, settings=_settings(), chunk_size=2)
    for row, combo in zip(stats.to_dict("records"), grid, strict=True):
        single = bt.evaluate(strat, p, combo, settings=_settings())
        assert row["total_return"] == pytest.approx(single["total_return"], rel=1e-9)
    # Capping the runaway changes the realized book, so the two combos must not be equal —
    # proof the cap ran in the column-stacked sweep, not just in evaluate.
    assert stats["total_return"].iloc[0] != pytest.approx(stats["total_return"].iloc[1])


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


def test_pessimistic_sweep_matches_evaluate_and_charges_the_benchmark_too(panel):
    """`strategy.fill` must reach sweep, evaluate AND the benchmark hurdle identically."""
    from trading_simple import strategies

    def weights_only(p, *, slot: float, **_):
        w = pd.DataFrame(0.0, index=p.close.index, columns=p.close.columns)
        w.iloc[5:30, 0] = slot
        w.iloc[30:, 1] = slot                      # rotate AAA -> BBB: a sell and a buy
        return {"weights": w}

    strat = replace(strategies.load("informed_flow"), signals=weights_only,
                    params={"slot": 0.5}, fill="pessimistic")
    grid = [{"slot": 0.5}, {"slot": 1.0}]
    stats = bt.sweep(strat, panel, grid, settings=_settings(), chunk_size=2)
    for row, combo in zip(stats.to_dict("records"), grid, strict=True):
        single = bt.evaluate(strat, panel, combo, settings=_settings())
        assert row["total_return"] == pytest.approx(single["total_return"], rel=1e-9)
        assert row["excess_cagr"] == pytest.approx(single["excess_cagr"], rel=1e-6)
    opened = bt.sweep(replace(strat, fill="open"), panel, grid, settings=_settings())
    assert (stats["total_return"] < opened["total_return"]).all()   # worse fills, every combo

    worst, _ = bt.benchmark(panel, fees=0.0, slippage=0.0, fill="pessimistic")
    neutral, _ = bt.benchmark(panel, fees=0.0, slippage=0.0)
    assert worst < neutral                         # its one buy per name prints at the high


def test_pessimistic_signal_sweep_with_stops_matches_evaluate(panel):
    """The from_signals branch (stops + cash-shared wallet) under pessimistic fills."""
    from trading_simple import strategies

    strat = replace(strategies.load("donchian_breakout"), fill="pessimistic")
    grid = [{"entry_window": 10, "atr_window": 7, "atr_mult": m} for m in (1.5, 3.0)]
    stats = bt.sweep(strat, panel, grid, settings=_settings(), chunk_size=2)
    for row, combo in zip(stats.to_dict("records"), grid, strict=True):
        single = bt.evaluate(strat, panel, combo, settings=_settings())
        assert row["n_trades"] == single["n_trades"]
        assert row["total_return"] == pytest.approx(single["total_return"], rel=1e-9)


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


def test_cash_frac_short_proceeds_never_fund_a_long_and_a_stop_covers_whole():
    """THE regression (2026-09-27): short proceeds are locked for the cover (lock_cash).

    Short B for 95% of the wallet, then go long A for 95% of what is free, then B rallies
    through its 10% stop. Without lock_cash vectorbt counted B's $9,500 of proceeds as cash:
    A was bought for $18,525 (the $10k wallet ran 280% invested) and B's stop could buy back
    only 8.9 of its 95 shares — the other 86 stayed short past their stop. burst_scan found it
    on a real book (free cash negative on 11% of bars, covers split, entries skipped).
    """
    n = 8
    idx = pd.date_range("2026-06-01 13:30", periods=n, freq="1min", tz="UTC")
    flat = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    high = flat.copy()
    high.iloc[4, 1] = 115.0                               # bar 4: through B's stop at 110
    p = Panel(close=flat, open_=flat.copy(), high=high, low=flat.copy(),
              volume=pd.DataFrame(1_000.0, index=idx, columns=["A", "B"]), bar="1m")
    none = pd.DataFrame(False, index=idx, columns=["A", "B"])
    entries, short_entries = none.copy(), none.copy()
    short_entries.iloc[1, 1] = True                       # short B first: its sale brings in cash
    entries.iloc[2, 0] = True                             # then long A
    stop = pd.DataFrame(np.nan, index=idx, columns=["A", "B"])
    stop.iloc[1, 1] = 0.10
    pf = bt.run(p, {"entries": entries, "exits": none, "short_entries": short_entries,
                    "short_exits": none, "sl_stop": stop, "cash_frac": 0.95},
                fees=0.0, slippage=0.0, init_cash=10_000)
    trades = bt.trade_records(pf)
    b = trades[trades["symbol"] == "B"]
    assert len(b) == 1                                    # ONE cover, not split in pieces
    assert b.iloc[0]["size"] == pytest.approx(95.0)
    assert b.iloc[0]["exit_ts"] == idx[4] and b.iloc[0]["exit_price"] == pytest.approx(110.0)
    a = trades[trades["symbol"] == "A"].iloc[0]
    assert a["size"] * a["entry_price"] == pytest.approx(0.95 * 500.0)   # only the FREE cash
    gross = pf.asset_value(group_by=False).abs().sum(axis=1)
    assert (gross <= pf.value() + 1e-9).all()             # never above 100%, shorts included
    assert (pf.cash(free=True) >= -1e-9).all()


def _flat_daily(n=14, symbols=("A", "B")):
    idx = pd.date_range("2026-01-05", periods=n, freq="D", tz="UTC")
    flat = pd.DataFrame(100.0, index=idx, columns=list(symbols))
    return Panel(close=flat, open_=flat.copy(), high=flat.copy(), low=flat.copy(),
                 volume=pd.DataFrame(1_000.0, index=idx, columns=list(symbols)), bar="1d")


def test_portfolio_short_proceeds_never_fund_a_long():
    """Portfolio mode locks a short's sale proceeds too (pending proposal #31, approved
    2026-10-19). Short B at 95% of the book, then ask for 95% long A: only the FREE cash may
    buy A, so gross exposure stays at the book and free cash never goes negative."""
    p = _flat_daily(6)
    w = pd.DataFrame(0.0, index=p.close.index, columns=["A", "B"])
    w.iloc[1:, 1] = -0.95
    w.iloc[2:, 0] = 0.95
    pf = bt.run(p, {"weights": w}, fees=0.0, slippage=0.0, init_cash=10_000)
    gross = pf.asset_value(group_by=False).abs().sum(axis=1)
    assert (gross <= pf.value() + 1e-6).all()
    assert (pf.cash(free=True) >= -1e-6).all()
    assert pf.asset_value(group_by=False)["A"].iloc[-1] == pytest.approx(500.0, rel=1e-6)


def test_borrow_charges_a_short_its_yearly_rate_for_the_days_it_was_held():
    """36.5%/yr on a $5,000 short held 10 calendar days costs 0.1% a day: $50, charged on
    the buy-back; a long pays nothing, and the same short without ``borrow`` pays nothing."""
    p = _flat_daily(14)
    w = pd.DataFrame(0.0, index=p.close.index, columns=["A", "B"])
    w.iloc[1:11, 1] = -0.5                                # short B on bar 1, buy back on bar 11
    w.iloc[1:, 0] = 0.3                                   # a long, never charged
    free = bt.run(p, {"weights": w}, fees=0.0, slippage=0.0, init_cash=10_000)
    paid = bt.run(p, {"weights": w, "borrow": 0.365}, fees=0.0, slippage=0.0,
                  init_cash=10_000)
    assert free.value().iloc[-1] == pytest.approx(10_000.0)
    assert free.value().iloc[-1] - paid.value().iloc[-1] == pytest.approx(50.0, rel=1e-6)


def test_borrow_of_a_short_open_on_the_last_bar_is_still_charged():
    """No buy-back inside the window: the accrual (12 days at 0.1%/day on $5,000 = $60) is
    charged on the order that opened the short, so a window cut short skips nothing; a
    per-name rate frame is read bar by bar (NaN = free)."""
    p = _flat_daily(14)
    w = pd.DataFrame(0.0, index=p.close.index, columns=["A", "B"])
    w.iloc[1:, 1] = -0.5
    rate = pd.DataFrame({"A": np.nan, "B": 0.365}, index=p.close.index)
    paid = bt.run(p, {"weights": w, "borrow": rate}, fees=0.0, slippage=0.0, init_cash=10_000)
    assert 10_000.0 - paid.value().iloc[-1] == pytest.approx(60.0, rel=1e-6)


def test_borrow_needs_portfolio_mode():
    p = _flat_daily(5)
    none = pd.DataFrame(False, index=p.close.index, columns=["A", "B"])
    with pytest.raises(ValueError, match="portfolio mode only"):
        bt.run(p, {"entries": none, "exits": none, "borrow": 0.01}, fees=0.0, slippage=0.0)


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


# --- per-entry cash_frac frame: risk-per-trade sizing ---------------------------
def _cash_frac_frame(panel, a: float = 0.10, b: float = 0.30):
    """Both symbols enter at bar 5; AAA deploys ``a`` of the free cash, BBB ``b``."""
    idx, cols = panel.close.index, panel.close.columns
    entries = pd.DataFrame(False, index=idx, columns=cols)
    entries.iloc[5] = True
    frac = pd.DataFrame(0.25, index=idx, columns=cols)     # dense: the slot everywhere else
    frac.iloc[5] = [a, b]
    return {"entries": entries, "exits": bt.exit_after(entries, 30), "cash_frac": frac}


def test_cash_frac_frame_sizes_each_entry_on_its_own(panel):
    """A per-bar frame is read at the FILL bar: 10% of $10k, then 30% of the $9k left."""
    pf = bt.run(panel, _cash_frac_frame(panel), fees=0.0, slippage=0.0, init_cash=10_000)
    assert bt.is_grouped(pf)
    trades = bt.trade_records(pf)
    notionals = sorted(trades["size"] * trades["entry_price"])
    assert notionals == [pytest.approx(1_000.0), pytest.approx(2_700.0)]
    invested = pf.asset_value(group_by=False).sum(axis=1)
    assert (invested <= pf.value() + 1e-9).all()             # still one wallet, never over


def test_cash_frac_frame_with_nan_is_refused(panel):
    """A NaN size at a fill bar SKIPS the entry silently — so a sparse frame is an error."""
    sig = _cash_frac_frame(panel)
    sig["cash_frac"] = sig["cash_frac"].where(sig["entries"])    # NaN off the entry bars
    with pytest.raises(ValueError, match="NaN"):
        bt.run(panel, sig, fees=0.0, slippage=0.0)
    assert bt.cash_frac_size(0.25) == 0.25                       # scalars pass through


def test_sweep_cash_frac_frame_matches_evaluate(panel):
    """A per-entry frame stacks per combo in the sweep and agrees with evaluate()."""
    from trading_simple import strategies

    strat = strategies.load("support_bounce")   # host shell; signals get replaced

    def sig(p, *, a: float, **_):
        return _cash_frac_frame(p, a=a)

    strat = replace(strat, signals=sig, params={"a": 0.10})
    grid = [{"a": 0.10}, {"a": 0.50}]
    stats = bt.sweep(strat, panel, grid, settings=_settings(), chunk_size=2)
    for row, combo in zip(stats.to_dict("records"), grid, strict=True):
        single = bt.evaluate(strat, panel, combo, settings=_settings())
        assert row["total_return"] == pytest.approx(single["total_return"], rel=1e-9)
        assert row["n_trades"] == single["n_trades"]
    assert stats.iloc[0]["total_return"] != stats.iloc[1]["total_return"]   # the frame mattered


# --- alpha after beta -------------------------------------------------------------
def test_alpha_cagr_is_the_return_after_beta():
    """The baseline IS the benchmark (beta 1, alpha 0); half of it has beta ~0.5; flat is 0/0.

    On a one-symbol panel, so the benchmark actually moves: the two-symbol fixture's names
    trend symmetrically and its equal-weight index is flat to 1e-14, which makes beta moot.
    """
    from trading_simple import strategies

    panel = make_panel(symbols=("AAA",))
    baseline = strategies.load("baseline")
    base = bt.evaluate(baseline, panel, {}, settings=_settings())
    assert base["beta"] == pytest.approx(1.0, abs=1e-6)
    assert base["alpha_cagr"] == pytest.approx(0.0, abs=1e-6)
    assert base["alpha_cagr"] == pytest.approx(base["excess_cagr"], abs=1e-6)

    def half(p, **_):
        return {"weights": baseline.signals(p)["weights"] * 0.5}   # 50% cash, 50% the index

    stats = bt.evaluate(replace(baseline, signals=half), panel, {}, settings=_settings())
    assert 0.4 < stats["beta"] < 0.6
    assert stats["alpha_cagr"] == pytest.approx(stats["cagr"] - stats["beta"] * base["cagr"],
                                                abs=1e-4)
    assert stats["excess_cagr"] < stats["alpha_cagr"] < 0.0    # charged only for the beta it ran

    def flat(p, **_):
        entries = pd.DataFrame(False, index=p.close.index, columns=p.close.columns)
        return {"entries": entries, "exits": entries, "cash_frac": 0.25}

    pf = bt.run_strategy(replace(baseline, signals=flat), panel, settings=_settings())
    fees, slip = bt.panel_costs(_settings().costs, panel)
    flat_stats = bt.active_stats(pf, panel, fees=fees, slippage=slip, total_return=0.0)
    assert flat_stats["beta"] == 0.0 and flat_stats["alpha_cagr"] == 0.0


# --- engine proposals #46, #47, #49 (2026-10-24) ---------------------------------------

def _flat_panel(close: pd.DataFrame) -> Panel:
    return Panel(close=close, open_=close.copy(), high=close.copy(), low=close.copy(),
                 volume=pd.DataFrame(1e6, index=close.index, columns=close.columns), bar="1d")


def test_call_order_runs_sales_and_covers_before_new_shorts_and_buys():
    """#46: long cuts, short covers, new shorts, long buys, then names with no order."""
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    t = pd.DataFrame({"buy": [0.0, 0.3], "short": [0.0, -0.3], "cover": [-0.3, 0.0],
                      "sell": [0.3, 0.0], "none": [0.1, np.nan]}, index=idx)
    order = list(t.columns[bt._call_order(t)[1]])
    assert order == ["sell", "cover", "short", "buy", "none"]
    two = pd.concat([t, t], axis=1)                       # a sweep: two books side by side
    seq = bt._call_order(two, groups=2)[1]
    assert list(seq[:5]) == list(seq[5:]) == list(bt._call_order(t)[1])


def test_a_new_short_is_not_starved_by_the_cover_that_funds_it():
    """#46: a full book covers B and shorts C on the same bar; under call_seq="auto" the new
    short ran first and was part-filled (~20 of ~500 shares)."""
    idx = pd.date_range("2024-01-01", periods=4, freq="B", tz="UTC")
    close = pd.DataFrame({"A": [10.0] * 4, "B": [10, 9, 9, 9.0], "C": [10.0] * 4}, index=idx)
    w = pd.DataFrame({"A": [0.49] * 4, "B": [-0.49, 0.0, 0.0, 0.0],
                      "C": [0.0, -0.49, -0.49, -0.49]}, index=idx)
    pf = bt._run_portfolio(_flat_panel(close), w, price=close, fees=0.0, slippage=0.0,
                           init_cash=10_000)
    assert pf.assets()["C"].iloc[-1] == pytest.approx(-0.49 * pf.value().iloc[1] / 10, rel=0.02)


def test_a_refused_cover_keeps_working_until_cash_frees_it():
    """#47: B's short tripled, so its cover needs free cash that a 98%-gross book lacks; the
    order fills on the next bar, once A's sale frees cash, instead of being lost."""
    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    close = pd.DataFrame({"A": [10.0] * 6, "B": [10, 20, 30, 30, 30, 30.0]}, index=idx)
    w = pd.DataFrame({"A": [0.49, 0.49, 0.49, 0.0, 0.0, 0.0],
                      "B": [-0.49, -0.49, 0.0, 0.0, 0.0, 0.0]}, index=idx)
    kw = dict(price=close, fees=0.0, slippage=0.0, init_cash=10_000)
    once = bt._run_portfolio(_flat_panel(close), w, working=False, **kw)
    kept = bt._run_portfolio(_flat_panel(close), w, **kw)
    assert once.assets()["B"].iloc[2] < -400 and once.assets()["B"].iloc[-1] < -400   # lost
    assert kept.assets()["B"].iloc[3:].eq(0.0).all()                                  # filled
    assert kept.assets()["A"].iloc[3:].eq(0.0).all()


def test_an_order_that_never_fills_warns():
    """#49: with no sale ever freeing cash the cover cannot fill; the run says so."""
    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    close = pd.DataFrame({"A": [10.0] * 6, "B": [10, 20, 30, 30, 30, 30.0]}, index=idx)
    w = pd.DataFrame({"A": [0.49] * 6, "B": [-0.49, -0.49, 0.0, 0.0, 0.0, 0.0]}, index=idx)
    with pytest.warns(RuntimeWarning, match="stayed unfilled"):
        pf = bt._run_portfolio(_flat_panel(close), w, price=close, fees=0.0, slippage=0.0,
                               init_cash=10_000)
    assert pf.assets()["B"].iloc[-1] < 0


def test_a_book_without_failures_runs_once_and_matches_the_plain_runner():
    """Working orders change nothing when every order fills."""
    p = make_panel()
    w = pd.DataFrame(0.0, index=p.close.index, columns=p.close.columns)
    w.iloc[:, 0], w.iloc[30:, 1] = 0.4, 0.3
    kw = dict(price=p.open_, fees=0.001, slippage=0.0005, init_cash=10_000)
    a = bt._run_portfolio(p, w, working=False, **kw).value()
    b = bt._run_portfolio(p, w, **kw).value()
    assert np.allclose(a.to_numpy(), b.to_numpy())
