"""xs_portable_alpha — a constant SPY core plus the SI + MOM market-neutral overlay, in ONE cash
account (proposals lab, cross-strategy round, 2026-10-23): the absolute-performance design.

A hedged long/short book cannot reach a bull market's CAGR: shorts lock cash, so the book's gross
stays at or below about 100%, and the lab's books run at 0.90. This account buys back part of that
CAGR with beta it owns outright, and keeps the overlay for alpha and for falling markets:

* **Core:** SPY at ``core`` of the book (0.45: an equal split of the 0.90 gross between beta and
  alpha, fixed a priori and never tuned). The engine buys it on each period's first bar and holds
  it: a constant target is never re-sent, so it is never re-trimmed (about two orders a year).
  SPY is the lab's ``--hedge`` column (``lab.with_spy``). It is never ranked and never reaches a
  sleeve.
* **Overlay:** ``xs_three_book.signals(panel, sleeves="si,mom", gross=gross - core)``: the
  short-interest book (``informed_flow_ls_p1``, lab label ``v6_si``) and residual momentum
  (``resid_momentum`` ``U0v2``) at ``(gross - core) / 2`` each (0.225), each component run
  unchanged on the panel without SPY, summed and netted per name. That module strips SPY from the
  sleeve inputs, applies its coverage switch and moves every target change to the name's next
  priced open. Its file and the six component files it reads are pinned by :data:`SHA256`; a
  changed file refuses to load (:func:`book`).
* **Iteration I1** (``core_vol: true``): the core becomes ``core × m_t`` (:func:`core_scale`), a
  month-end volatility rule on SPY itself. The freed gross sits in cash; the overlay is never
  re-traded because of it.

Total nominal gross is ``gross`` (0.90). Borrow is the overlay's (SPY at 0.5%, never used: the
core is long). ``retrim_k`` 2 trims a long that doubles against its slot, as in every component.
Every order fills at the next open; the lab's SPY column has open = close (``lab.with_spy``), so
SPY fills at the fill bar's close.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGIES = Path(__file__).resolve().parent.parent
#: The book module this account wraps: (its folder, the prefixed module name it is loaded under).
BASE = ("xs_three_book", "_xspa_tb")
#: SHA-256 of every file the account reads, pinned 2026-10-23 before any run: xs_three_book's
#: book module and the six component files it loads (they are xs_three_book's own pins too).
SHA256 = {
    "xs_three_book/strategy.py":
        "aa7dcba5073a70b8e3db52de21b7afca07b5e125513d9a054aa51af6e8bba84a",
    "informed_flow_ls_p1/strategy.py":
        "a05e84d4176ef60bcf68c20eca41c4a6cb113e78d17089f37a3f3583b88dee67",
    "informed_flow_ls_p1/config.yaml":
        "959dbf26a2b7716623348bd09983a567d10b8c1638482fdea5e3cf8d6f0cc7a1",
    "resid_momentum/strategy.py":
        "5cc54c69620cab771e1a268f2900ba0137a107220889794abb3ff6cc3bc52753",
    "resid_momentum/config.yaml":
        "57429793cc801b25ea4775b3d2a2e91cd0bc12584f92af332994154cbe8f17bf",
    "adaptive_hedge/strategy.py":
        "52684a2118b5f6779fe3a0ddbd39c230fde2f1ee1a16cfd27251c013941b936c",
    "adaptive_hedge/config.yaml":
        "69cc3aea8f129e010987470576e741e107001366a5a391abe386f79caefc44e9",
}
CORE = "SPY"          # the core: the lab's index column, never ranked
GC = 0.005            # yearly borrow where the overlay sets no rate (SPY: never short)
RETRIM_K = 2.0        # a long that doubles against its slot is trimmed back (engine cap)
VAR_WINDOW = 21       # I1: sessions of SPY's realized variance
MEDIAN_WINDOW = 252   # I1: sessions of the trailing median that variance is compared with


def file_hashes() -> dict[str, str]:
    """SHA-256 of the pinned files as they are on disk now (compare with :data:`SHA256`)."""
    return {k: hashlib.sha256((STRATEGIES / k).read_bytes()).hexdigest() for k in SHA256}


@functools.cache
def book():
    """xs_three_book's ``strategy.py``, loaded once by path under a prefixed module name.

    Raises:
        RuntimeError: If any pinned file differs from :data:`SHA256` (every label of this
            account would then be about another book).
    """
    changed = sorted(k for k, v in file_hashes().items() if v != SHA256[k])
    if changed:
        raise RuntimeError(f"xs_portable_alpha: pinned file(s) changed: {changed}; re-pin and "
                           "re-run every label")
    folder, alias = BASE
    spec = importlib.util.spec_from_file_location(alias, STRATEGIES / folder / "strategy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def signals(panel, *, core: float = 0.45, core_vol: bool = False, sleeves: str = "si,mom",
            gross: float = 0.90, **rest) -> dict:
    """Target weights (long > 0, short < 0), every name's borrow rate and the long cap.

    Args:
        panel: Daily panel with ``features`` ``short_interest`` and ``fund_shares`` and a SPY
            column (the lab's ``--hedge``: prices only, no volume, no features).
        core: SPY's target weight (0.45; 0 = the overlay alone at ``gross``).
        core_vol: Iteration I1: SPY's weight is ``core × m_t`` (:func:`core_scale`).
        sleeves: The overlay's books, an ``xs_three_book`` ``sleeves`` string.
        gross: Nominal gross of the whole account; the overlay runs at ``gross − core``.
        **rest: Passed to ``xs_three_book.signals`` (its defaults otherwise).

    Returns:
        ``{"weights", "borrow", "retrim_k"}`` for the engine's portfolio mode.

    Raises:
        ValueError: If the panel has no SPY column.
    """
    close = panel.close
    if CORE not in close.columns:
        raise ValueError(f"{CORE} is not a column of the panel (lab: --hedge)")
    overlay = book().signals(panel, sleeves=sleeves, gross=gross - core, **rest)
    w = overlay["weights"].reindex(index=close.index, columns=close.columns).fillna(0.0)
    w[CORE] = w[CORE] + (core * core_scale(close[CORE]) if core_vol else core)
    borrow = overlay["borrow"].reindex(index=close.index, columns=close.columns).fillna(GC)
    return {"weights": w, "borrow": borrow, "retrim_k": RETRIM_K}


_DEFAULTS = {k: v.default for k, v in inspect.signature(signals).parameters.items()
             if v.default is not inspect.Parameter.empty}


def valid(params: dict) -> bool:
    """Reject accounts this design cannot express: ``0 <= core < gross <= 1``, a boolean
    ``core_vol``, and an overlay ``xs_three_book.valid`` accepts at ``gross − core`` (missing
    keys read as the defaults, so a partial dict is judged too)."""
    p = {**_DEFAULTS, **params}
    if not (isinstance(p["core_vol"], bool) and 0.0 <= p["core"] < p["gross"] <= 1.0):
        return False
    rest = {k: v for k, v in p.items() if k not in _DEFAULTS}
    return bool(book().valid({**rest, "sleeves": p["sleeves"], "gross": p["gross"] - p["core"]}))


def core_scale(spy: pd.Series) -> pd.Series:
    """I1's core multiplier ``m_t`` on the panel's index, in force at each bar.

    On SPY's own rows: its close-to-close return's ``VAR_WINDOW``-session variance (ddof 1)
    against the median of that variance over the trailing ``MEDIAN_WINDOW`` sessions (the bar
    itself included; at least half the window): ``m = 1`` at or below the median, ``0.5`` at or
    below twice it, ``0`` above. Decided only at month-end closes (resid_momentum's
    ``_month_ends`` on SPY's priced rows), applied from the next bar and held to the next month
    end; 1 before the first decision.
    """
    rows = spy.notna()
    var = spy[rows].pct_change(fill_method=None).rolling(VAR_WINDOW).var()
    median = var.rolling(MEDIAN_WINDOW, min_periods=MEDIAN_WINDOW // 2).median()
    m = pd.Series(np.select([var <= median, var <= 2.0 * median], [1.0, 0.5], 0.0),
                  index=var.index).where(median.notna(), 1.0)
    ends = book().component("mom")._month_ends(spy.index, rows)
    return m.reindex(spy.index).where(ends).ffill().shift(1).fillna(1.0)
