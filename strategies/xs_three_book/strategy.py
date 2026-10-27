"""xs_three_book — three proven books in ONE cash account: short interest, residual momentum and
the adaptive hedge, equal thirds, netted (proposals lab, cross-strategy round, 2026-10-23).

The lab's three books with T1 selection skill trade side by side in one portfolio-mode account:

* **SI** — ``informed_flow_ls_p1`` (lab label ``v6_si``): long the names short sellers avoid,
  short the ones they crowd; 15 + 15 slots, nominal gross 0.98.
* **MOM** — ``resid_momentum`` (``U0v2``): residual 12-1 momentum, 20 + 20 slots, nominal gross
  0.90, its own volatility scale inside the sleeve.
* **AH** — ``adaptive_hedge`` (``TUNEDv2`` = its config plus :data:`AH_TUNED`): the SI long rule
  plus breakdown shorts held about 300 days; 30 slots of 0.98 / 30.

Each component's ``signals()`` runs UNCHANGED on the panel without SPY. The files are loaded by
path under prefixed module names and never edited; :data:`SHA256` pins them. Sleeve ``s`` is
scaled to its budget ``b_s`` over its own nominal gross ``g_s``. ``b_s`` is ``(gross −
hedge_max) / n`` each, 0.30 at the defaults; ``budgets`` sets it per sleeve. The sleeves are then
SUMMED per name, so opposite positions cancel (no position, no borrow) and same-side ones add.
Nothing is rebalanced or estimated: a name trades only when some sleeve changes its target, so the
account inherits each sleeve's own turnover. The nominal gross of 0.90 leaves the headroom that
``lock_cash`` needs.

Switches for the controls and variants:

* ``sleeves`` / ``budgets``: which books, and their budgets in ``sleeves`` order (CRISIS =
  ``mom,ah``; C_core = ``si,mom``; S_si / S_mom / S_ah = one book at the whole budget).
* ``ah_variant: u0``: AH at its config values, without the TUNEDv2 overrides.
* ``ah_part: shorts``: AH's short weights only, ``b_ah / max_shorts`` a slot (the crisis sleeve
  of iteration I2). AH's longs duplicate the SI long rule.
* ``ah_hedge: SPY``: AH's own index mode. Every AH short slot is held as a SPY short of the slot's
  size: the same breadth timing, without name selection.

**Coverage switch.** On the fill bar after a decision bar where fewer than ``min_coverage`` of
the names closing at or above $5 carry a days-to-cover, the SI and AH sleeves are set to 0 and
their budget sits in cash. A short-interest table that went thin cannot tilt the account.

**Risk layer (iteration I1; off at the defaults).**

* ``book_vol`` > 0: the summed overlay is scaled by ``min(1, book_vol / σ)``. σ is the trailing
  ``vol_window``-bar realized volatility of the UNSCALED overlay (Σ weight × close-to-close
  return, resid_momentum's ``book_scale``). The scale is rounded to ``vol_step``, floored at
  ``min_scale``, decided on month-end rows and applied from the next open.
* ``hedge_max`` > 0: the sleeves share ``gross − hedge_max``. On the first bar after each month
  end, the overlay's ex-ante beta β = Σ w_i × β_i is computed: w is the book about to be held,
  and β_i is the trailing ``beta_window``-bar beta to the core panel's equal-weight market, known
  at the decision. A SPY short ``h = −min(max(β − beta_cap, 0) / β_SPY, hedge_max)`` caps it,
  rounded UP in size to ``hedge_step``, so that an uncapped hedge always brings β to
  ``beta_cap`` or below. It is held to the next month end and is never a long hedge.

Every order fills at the next open. The sleeves' decisions are their own (``bt.next_bar``), the
overlay's at month-end closes. A target change due on a bar without an open waits for the name's
next priced open (adaptive_hedge's ``_on_priced_opens``).
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
#: Sleeve key -> (component folder, the prefixed module name its strategy.py is loaded under).
COMPONENTS = {"si": ("informed_flow_ls_p1", "_xs3_p1"),
              "mom": ("resid_momentum", "_xs3_rm"),
              "ah": ("adaptive_hedge", "_xs3_ah")}
#: SHA-256 of every component file the book reads, pinned 2026-10-23 before any run.
SHA256 = {
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
#: adaptive_hedge's final, TUNEDv2: its config.yaml (U0) plus these overrides.
AH_TUNED = {"max_shorts": 20, "adx_min": 40.0, "entry_window": 27, "atr_mult": 16.0,
            "atr_window": 48}
SLEEVES = ("si", "mom", "ah")
AH_VARIANTS, AH_PARTS = ("tuned", "u0"), ("all", "shorts")
HEDGE = "SPY"        # the index fund: never ranked, traded only by ah_hedge and the beta cap
GC = 0.005           # yearly borrow where no sleeve sets a rate
MIN_PRICE = 5.0      # the coverage switch counts names closing at or above this
TINY = 1e-12         # a summed weight this small is a netted name: exactly 0


def file_hashes() -> dict[str, str]:
    """SHA-256 of the component files as they are on disk now (compare with :data:`SHA256`)."""
    return {k: hashlib.sha256((STRATEGIES / k).read_bytes()).hexdigest() for k in SHA256}


@functools.cache
def component(key: str):
    """A component's ``strategy.py``, loaded once by path under its prefixed module name."""
    folder, alias = COMPONENTS[key]
    spec = importlib.util.spec_from_file_location(alias, STRATEGIES / folder / "strategy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.cache
def _config(key: str) -> tuple:
    """A component's shipped ``config.yaml`` values, read once through the repo's loader."""
    from trading_simple import strategies
    return tuple(strategies.load(COMPONENTS[key][0]).params.items())


def component_params(key: str, ah_variant: str = "tuned") -> dict:
    """A component's shipped ``config.yaml`` values (AH: plus :data:`AH_TUNED` when tuned)."""
    params = dict(_config(key))
    if key == "ah" and ah_variant == "tuned":
        params.update(AH_TUNED)
    return params


def signals(panel, *, sleeves: str = "si,mom,ah", budgets: str = "", gross: float = 0.90,
            ah_variant: str = "tuned", ah_part: str = "all", ah_hedge: str = "",
            book_vol: float = 0.0, vol_window: int = 126, vol_step: float = 0.25,
            min_scale: float = 0.25, beta_cap: float = 0.0, hedge_max: float = 0.0,
            beta_window: int = 126, hedge_step: float = 0.02, min_coverage: float = 0.5,
            retrim_k: float = 2.0, **_) -> dict:
    """Target weights (long > 0, short < 0), every name's borrow rate and the long cap.

    Args:
        panel: Daily panel with ``features`` ``short_interest`` (days-to-cover) and
            ``fund_shares``. SPY, if present, is never ranked; it must be present when
            ``ah_hedge`` or ``hedge_max`` is set (lab: ``--hedge``).
        sleeves: Comma-separated subset of :data:`SLEEVES`, no repeats.
        budgets: ``""`` (equal: ``(gross − hedge_max) / n`` each) or one positive number per
            sleeve, in ``sleeves`` order, summing to at most ``gross − hedge_max``.
        gross: Nominal gross of the whole account, the hedge included (<= 1).
        ah_variant: ``"tuned"`` (TUNEDv2) or ``"u0"`` (adaptive_hedge's config values).
        ah_part: ``"all"`` or ``"shorts"`` (AH's short weights only, ``b_ah / max_shorts`` a
            slot).
        ah_hedge: ``""`` or ``"SPY"`` (AH's index mode: its short slots held as SPY).
        book_vol: Yearly volatility the overlay is scaled down to (0 = off).
        vol_window, vol_step, min_scale: The overlay's volatility window, scale step and floor.
        beta_cap: Ex-ante beta above which the SPY hedge starts.
        hedge_max: Largest SPY short (0 = no hedge); the sleeves share ``gross − hedge_max``.
        beta_window: Bars of the trailing betas the hedge reads.
        hedge_step: The hedge is rounded up in size to a multiple of this.
        min_coverage: Share of the names closing >= $5 that must carry a days-to-cover, or
            the SI and AH sleeves sit in cash on the next bar.
        retrim_k: A long that grows to ``k ×`` its target is trimmed back (engine cap; 0 = off).

    Returns:
        ``{"weights", "borrow", "retrim_k"}`` for the engine's portfolio mode (``retrim_k``
        only when > 0).
    """
    parts = compose(panel, sleeves=sleeves, budgets=budgets, gross=gross,
                    ah_variant=ah_variant, ah_part=ah_part, ah_hedge=ah_hedge,
                    book_vol=book_vol, vol_window=vol_window, vol_step=vol_step,
                    min_scale=min_scale, beta_cap=beta_cap, hedge_max=hedge_max,
                    beta_window=beta_window, hedge_step=hedge_step, min_coverage=min_coverage)
    out = {"weights": parts["weights"], "borrow": parts["borrow"]}
    if retrim_k > 0:
        out["retrim_k"] = retrim_k
    return out


_DEFAULTS = {k: v.default for k, v in inspect.signature(signals).parameters.items()
             if v.default is not inspect.Parameter.empty}


def valid(params: dict) -> bool:
    """Reject parameter sets the account cannot express (missing keys read as ``signals()``'s
    defaults, so a partial dict is judged too)."""
    p = {**_DEFAULTS, **params}
    try:
        _budgets(_names(p["sleeves"]), p["budgets"], p["gross"] - p["hedge_max"])
    except (TypeError, ValueError):
        return False
    return (p["ah_variant"] in AH_VARIANTS and p["ah_part"] in AH_PARTS
            and p["ah_hedge"] in ("", HEDGE) and 0.0 < p["gross"] <= 1.0
            and 0.0 <= p["hedge_max"] < p["gross"] and p["beta_cap"] >= 0.0
            and p["book_vol"] >= 0.0 and p["vol_window"] >= 10 and 0.0 < p["vol_step"] <= 1.0
            and 0.0 < p["min_scale"] <= 1.0 and p["beta_window"] >= 4
            and p["hedge_step"] > 0.0 and 0.0 <= p["min_coverage"] <= 1.0
            and (p["retrim_k"] == 0 or p["retrim_k"] > 1))


def _names(sleeves: str) -> list[str]:
    """The sleeve keys of a ``sleeves`` string; raises ``ValueError`` on an empty, repeated or
    unknown key."""
    names = [s.strip() for s in str(sleeves).split(",") if s.strip()]
    if not names or len(set(names)) != len(names) or not set(names) <= set(SLEEVES):
        raise ValueError(f"sleeves {sleeves!r}: a non-empty subset of {SLEEVES}, no repeats")
    return names


def _budgets(names: list[str], budgets, room: float) -> dict[str, float]:
    """Each sleeve's budget: ``room / n`` each for ``""``, else one positive number per sleeve
    summing to at most ``room``; raises ``ValueError`` otherwise."""
    text = str(budgets).strip()
    if not text:
        return {k: room / len(names) for k in names}
    values = [float(x) for x in text.split(",")]
    if len(values) != len(names) or min(values) <= 0 or sum(values) > room + 1e-9:
        raise ValueError(f"budgets {budgets!r}: one positive number per sleeve, sum <= {room}")
    return dict(zip(names, values, strict=True))


def compose(panel, *, sleeves: str, budgets: str, gross: float, ah_variant: str, ah_part: str,
            ah_hedge: str, book_vol: float, vol_window: int, vol_step: float, min_scale: float,
            beta_cap: float, hedge_max: float, beta_window: int, hedge_step: float,
            min_coverage: float, **_) -> dict:
    """Everything the account reads, bar by bar (pure; also what ``research.py`` inspects).

    Returns:
        ``sleeves`` (key -> its weights as held: budget-scaled, switched, volatility-scaled, at
        the fill bar, on the panel's columns), ``coverage`` and ``on`` (the coverage switch at
        the fill bar), ``scale``, ``beta`` (the overlay's ex-ante beta at each bar; NaN without
        the hedge), ``hedge`` (the SPY weight), ``borrow`` and ``weights`` (Σ sleeves + hedge,
        each change moved to the name's next priced open).
    """
    names = _names(sleeves)
    budget = _budgets(names, budgets, gross - hedge_max)
    index, columns = panel.close.index, panel.close.columns
    if (ah_hedge or hedge_max > 0) and HEDGE not in columns:
        raise ValueError(f"{HEDGE} is not a column of the panel (lab: --hedge)")
    core = component("si")._without(panel, HEDGE)
    cov = coverage(core)
    on = (cov >= min_coverage).shift(1, fill_value=False)
    held, borrow = {}, pd.DataFrame(GC, index=index, columns=columns)
    for key in names:
        params = component_params(key, ah_variant)
        if key == "ah" and ah_hedge:
            sig = component("ah").signals(panel, **{**params, "hedge": ah_hedge})
        else:
            sig = component(key).signals(core, **params)
        w = sig["weights"].reindex(index=index, columns=columns).fillna(0.0)
        if key == "ah" and ah_part == "shorts":
            w = w.clip(upper=0.0) * ((budget[key] / max(params["max_shorts"], 1))
                                     / (params["gross"] / params["slots"]))
        else:
            w = w * (budget[key] / params["gross"])
        held[key] = w.mul(on.astype(float), axis=0) if key in ("si", "ah") else w
        rate = sig["borrow"].reindex(index=index, columns=columns).fillna(GC)
        borrow = borrow.where(borrow >= rate, rate)
    overlay = sum(held.values())
    rm = component("mom")
    rows = rm._full_rows(core.close)
    month_end = rm._month_ends(index, rows)
    fills = month_end[rows].shift(1, fill_value=False).reindex(index, fill_value=False)
    scale = pd.Series(1.0, index=index)
    if book_vol > 0:
        c = panel.close.where(rows, axis=0)
        decided = rm.book_scale(overlay, c / c.ffill().shift(1) - 1.0, rows, month_end,
                                vol_target=book_vol, vol_window=vol_window,
                                vol_step=vol_step, min_scale=min_scale)
        scale = decided[rows].shift(1, fill_value=1.0).reindex(index).ffill().fillna(1.0)
        held = {k: w.mul(scale, axis=0) for k, w in held.items()}
        overlay = overlay.mul(scale, axis=0)
    beta = pd.Series(np.nan, index=index)
    hedge = pd.Series(0.0, index=index)
    if hedge_max > 0:
        betas = trailing_betas(panel.close, core.close.columns, beta_window)
        beta = (overlay * betas[overlay.columns]).sum(axis=1)
        hedge = beta_hedge(beta, betas[HEDGE], fills, beta_cap=beta_cap, hedge_max=hedge_max,
                           hedge_step=hedge_step)
    total = overlay.copy()
    if hedge_max > 0:
        total[HEDGE] = total[HEDGE] + hedge
    total = total.where(total.abs() > TINY, 0.0)        # a netted name holds nothing at all
    weights = component("ah")._on_priced_opens(total, panel.open_)
    return {"sleeves": held, "coverage": cov, "on": on, "scale": scale, "beta": beta,
            "hedge": hedge, "borrow": borrow, "weights": weights}


def coverage(core) -> pd.Series:
    """Share of the names closing at or above :data:`MIN_PRICE` that carry a days-to-cover
    (NaN on a bar where no name is priced; 0 without the ``short_interest`` feature)."""
    close = core.close
    priced = close >= MIN_PRICE
    dtc = (core.features or {}).get("short_interest")
    if dtc is None:
        return pd.Series(0.0, index=close.index)
    have = priced & dtc.reindex_like(close).notna()
    return have.sum(axis=1) / priced.sum(axis=1).replace(0, np.nan)


def trailing_betas(close: pd.DataFrame, market_columns, window: int) -> pd.DataFrame:
    """Every column's trailing ``window``-bar beta to the equal-weight daily return of
    ``market_columns``, shifted one bar: the value on bar t is known at bar t − 1's close
    (``lab._ex_ante_beta``'s estimator; at least half the window observed)."""
    ret = close.pct_change(fill_method=None)
    market = ret[list(market_columns)].mean(axis=1)
    lag = max(window // 2, 2)
    var = market.rolling(window, min_periods=lag).var()
    return ret.rolling(window, min_periods=lag).cov(market).div(var, axis=0).shift(1)


def beta_hedge(beta: pd.Series, beta_spy: pd.Series, fills: pd.Series, *, beta_cap: float,
               hedge_max: float, hedge_step: float) -> pd.Series:
    """The SPY weight of the beta cap: ``−min(max(β − beta_cap, 0) / β_SPY, hedge_max)``,
    rounded UP in size to ``hedge_step``, decided on ``fills`` bars (the first bar after each
    month end) and held to the next one; 0 before the first decision and where β_SPY <= 0."""
    spy = beta_spy.where(beta_spy > 0)
    need = ((beta - beta_cap).clip(lower=0.0) / spy).clip(upper=hedge_max).fillna(0.0)
    h = (-np.ceil(need / hedge_step - 1e-9) * hedge_step).clip(lower=-hedge_max) + 0.0
    return h.where(fills).ffill().fillna(0.0)
