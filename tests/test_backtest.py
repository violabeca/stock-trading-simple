"""Engine helpers (pure) and full vectorbt runs (skipped if not installed)."""

import numpy as np
import pandas as pd
import pytest

from trading_simple import backtest as bt
from trading_simple.config import CostSettings, Settings


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
    stats = pd.DataFrame({"fast": [1, 2], "sharpe": [np.nan, 0.5], "n_trades": [0, 3]})
    assert bt.best_combo(stats)["fast"] == 2
    with pytest.raises(ValueError, match="NaN"):
        bt.best_combo(stats.assign(sharpe=np.nan))


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
    assert set(stats) == {"total_return", "sharpe", "max_drawdown", "win_rate",
                          "pct_profitable", "n_trades"}
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
