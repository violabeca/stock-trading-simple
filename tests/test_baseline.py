"""baseline strategy: the engine benchmark expressed through the strategy contract.

The identity is the whole point: if baseline ever drifts from
``backtest.benchmark()``, the visible strategy and the optimizer's hurdle stop
being the same thing and every comparison against it silently lies.
"""

import pytest

from tests.conftest import make_panel
from trading_simple import backtest as bt
from trading_simple import strategies
from trading_simple.config import Settings


@pytest.mark.parametrize("fees,slippage", [(0.0, 0.0), (0.001, 0.0005)])
def test_baseline_equity_matches_engine_benchmark(fees, slippage):
    panel = make_panel()
    strat = strategies.load("baseline")
    pf = bt.run(panel, strat.signals(panel), fees=fees, slippage=slippage)
    bench_return, bench_curve = bt.benchmark(panel, fees=fees, slippage=slippage)
    curve = bt.equity_curve(pf)["value"].pct_change().fillna(0.0)
    assert curve.to_numpy() == pytest.approx(bench_curve.to_numpy())
    assert float(bt.portfolio_stats(pf)["total_return"]) == pytest.approx(bench_return)


def test_baseline_scores_zero_active_stats():
    panel = make_panel()
    stats = bt.evaluate(strategies.load("baseline"), panel, {}, settings=Settings())
    assert stats["excess_cagr"] == pytest.approx(0.0, abs=1e-12)
    assert stats["info_ratio"] == pytest.approx(0.0, abs=1e-12)
