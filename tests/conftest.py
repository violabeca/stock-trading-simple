"""Shared synthetic fixtures: tiny deterministic panels, no DB, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_simple.data import Panel


def make_panel(n: int = 60, symbols: tuple[str, ...] = ("AAA", "BBB")) -> Panel:
    """A small deterministic daily panel with one engineered volume spike.

    AAA trends up, BBB trends down; both print a 10x volume spike at bar 40,
    so trend-gated volume strategies produce exactly one long (AAA) and one
    short (BBB) signal there.
    """
    index = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    close = pd.DataFrame({
        "AAA": np.linspace(100.0, 130.0, n),
        "BBB": np.linspace(100.0, 70.0, n),
    }, index=index)[list(symbols)]
    volume = pd.DataFrame(1_000.0, index=index, columns=close.columns)
    volume.iloc[40] = 10_000.0
    return Panel(
        close=close,
        open_=close * 0.999,
        high=close * 1.01,
        low=close * 0.99,
        volume=volume,
        bar="1d",
    )


@pytest.fixture
def panel() -> Panel:
    """The standard two-symbol synthetic panel."""
    return make_panel()
