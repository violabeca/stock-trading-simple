"""Figure payload guards: the envelope must not hide extremes, and must stay small.

These figures are the reason an executed 1-minute notebook used to weigh ~80 MB.
The thinning that fixed it is only safe because it keeps every bucket's min and
max, so a spike, wick or drawdown low can never be reduced away — that invariant
is what the tests below pin down, alongside a hard byte ceiling per figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest

from trading_simple import backtest as bt
from trading_simple import plots
from trading_simple.data import Panel

BARS = 82_680  # the 1m train window in the strategy configs: ~212 sessions x 390 bars


def intraday(n: int = BARS, symbols: tuple[str, ...] = ("AAA",)) -> Panel:
    """A synthetic 1-minute panel of the size the real strategy configs load."""
    index = pd.date_range("2025-07-01 13:30", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    close = pd.DataFrame(
        {s: 100.0 + np.cumsum(rng.standard_normal(n)) * 0.05 for s in symbols}, index=index)
    volume = pd.DataFrame(
        {s: rng.uniform(1e3, 5e3, n) for s in symbols}, index=index)
    return Panel(close=close, open_=close * 0.999, high=close * 1.001,
                 low=close * 0.999, volume=volume, bar="1m")


def test_keep_returns_none_when_nothing_to_drop():
    """Short series are left alone — and MAX_POINTS=None disables thinning entirely."""
    assert plots._keep(pd.Series(np.arange(100.0))) is None
    long = pd.Series(np.arange(plots.MAX_POINTS * 2, dtype=float))
    assert plots._keep(long) is not None
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(plots, "MAX_POINTS", None)
        assert plots._keep(long) is None


def test_envelope_keeps_endpoints_and_fits_the_budget():
    """Every trace stays within the point budget, first and last bar included."""
    s = pd.Series(np.sin(np.linspace(0, 50, BARS)))
    keep = plots._keep(s)
    assert keep[0] == 0 and keep[-1] == BARS - 1
    assert len(keep) <= plots.MAX_POINTS


def test_a_planted_spike_survives_thinning():
    """The load-bearing claim: thinning can never hide an extreme.

    A one-bar spike and a one-bar crash are exactly what naive ``[::k]`` striding
    drops — and on a trading chart that is a hidden drawdown or a missed wick.
    """
    s = pd.Series(np.zeros(BARS))
    s.iloc[12_345] = 99.0    # a single-bar spike, between any plausible stride
    s.iloc[54_321] = -99.0   # and a single-bar crash
    thinned = plots._thin(s)
    assert len(thinned) < len(s)
    assert thinned.max() == 99.0
    assert thinned.min() == -99.0
    assert 12_345 in thinned.index and 54_321 in thinned.index


def test_thin_handles_all_nan_buckets():
    """The panel index is a symbol union, so a series genuinely carries gaps."""
    s = pd.Series(np.arange(BARS, dtype=float))
    s.iloc[1_000:40_000] = np.nan  # a hole far wider than one bucket
    thinned = plots._thin(s)
    assert len(thinned) <= plots.MAX_POINTS
    assert thinned.notna().any()


def test_thin_frame_puts_every_column_on_one_grid():
    """plot_equity's unified hover needs all four traces to share one x grid."""
    n = BARS
    frame = pd.DataFrame({"a": np.arange(n, dtype=float),
                          "b": np.sin(np.linspace(0, 90, n)),
                          "c": np.cos(np.linspace(0, 30, n))})
    thinned = plots._thin(frame)
    assert len(thinned) <= plots.MAX_POINTS
    assert thinned.index.is_monotonic_increasing


def test_epoch_packs_datetimes_and_passes_others_through():
    """A DatetimeIndex becomes float64 ms; anything else is untouched."""
    idx = pd.date_range("2025-07-01", periods=3, freq="1min", tz="UTC")
    ms = plots._epoch(idx)
    assert ms.dtype == np.float64
    assert ms[0] == pd.Timestamp("2025-07-01").value / 1e6
    other = pd.Index(["a", "b"])
    assert plots._epoch(other) is other


@pytest.mark.parametrize("ceiling_mb", [4.0])
def test_figures_stay_small_on_a_realistic_intraday_panel(ceiling_mb):
    """The regression this whole change exists to prevent: multi-megabyte figures.

    Before thinning, plot_symbol serialized to ~36 MB and plot_equity to ~13 MB on
    exactly this panel, which is what pushed an executed notebook past 80 MB.
    """
    panel = intraday()
    close, volume = panel.close["AAA"], panel.volume["AAA"]
    trades = pd.DataFrame(columns=["symbol", "entry_ts", "entry_price", "exit_ts",
                                   "exit_price", "pnl", "return", "direction"])
    curve = pd.DataFrame({"value": np.linspace(1e5, 1.2e5, len(close)),
                          "return": np.linspace(0, 0.2, len(close)),
                          "fees": np.linspace(0, 900, len(close)),
                          "realized_pnl": np.linspace(0, 2e4, len(close))},
                         index=close.index)

    equity_fig = plots.plot_equity(curve)
    symbol_fig = plots.plot_symbol(
        close, volume, trades, symbol="AAA", equity=close * 10,
        overlays={"price": {"upper": close * 1.02, "lower": close * 0.98},
                  "momentum": {"mom": close.pct_change().fillna(0.0)},
                  "volume": {"avg": volume.rolling(20).mean()},
                  "volume_marks": volume > volume.quantile(0.99)})

    for name, fig in (("plot_equity", equity_fig), ("plot_symbol", symbol_fig)):
        mb = len(pio.to_json(fig).encode()) / 1e6
        assert mb < ceiling_mb, f"{name} serialized to {mb:.1f} MB (ceiling {ceiling_mb} MB)"


def test_equity_traces_share_one_x_grid():
    """All four plot_equity traces must land on identical x values (unified hover)."""
    n = BARS
    idx = pd.date_range("2025-07-01", periods=n, freq="1min", tz="UTC")
    curve = pd.DataFrame({"value": np.linspace(1e5, 1.2e5, n),
                          "return": np.linspace(0, 0.2, n),
                          "fees": np.linspace(0, 900, n),
                          "realized_pnl": np.sin(np.linspace(0, 40, n)) * 1e4},
                         index=idx)
    fig = plots.plot_equity(curve)
    grids = {tuple(trace.x) for trace in fig.data}
    assert len(grids) == 1, "plot_equity traces drifted onto different x grids"
    assert fig.layout.xaxis.type == "date"


def test_daily_panel_is_untouched(panel):
    """Daily figures already fit the budget, so thinning must be a no-op for them."""
    res = bt.run(panel, {"entries": panel.close > 0, "exits": panel.close < 0},
                 fees=0.0, slippage=0.0)
    curve = bt.equity_curve(res)
    fig = plots.plot_equity(curve)
    assert len(fig.data[0].x) == len(curve)
