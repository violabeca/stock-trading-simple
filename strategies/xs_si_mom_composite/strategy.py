"""xs_si_mom_composite — one dollar-neutral book ranked on short interest plus residual momentum.

A cross-strategy of the proposals lab's last round (``strategies/proposals_lab``): two alpha
sources that are nearly orthogonal in the cross-section are integrated at the SIGNAL level, into
one ranking that drives one book, instead of holding two books side by side.

* **SI** (``informed_flow_ls_p1``): minus days-to-cover averaged over ``dtc_window`` bars,
  percentile inside ``neutral_groups`` dollar-volume groups, silent below 50% coverage.
* **MOM** (``resid_momentum`` U0v2): ``Σe / sd(e)`` over bars t−251 .. t−21 with ``e = r − β·m``
  against the equal-weight market of names >= ``min_price``, ranked on month-end rows only and
  carried to the next month end (a name unranked at the last month end stays unranked).
* **PROF** (optional, ``quality_junk``): profitability ranked within sector and measure.

Per bar (decided at the close, filled at the next open):

1. Eligible: informed_flow_ls_p1's (close >= ``min_price``, a days-to-cover, ``fund_shares`` > 0).
2. Each side's composite ``Σ w_k × rank_k.fillna(0.5) / Σ w_k`` over its sleeves, averaged over
   ``score_smooth`` bars and re-ranked among the eligible names.
3. The book is informed_flow_ls_p1's ``_select``: a long enters at rank >= ``q_in`` and leaves
   below ``q_out``, a short mirrors it; exits first, then free slots best first; ``max_hold``
   (the clock restarts in the entry zone); the squeeze stop (``short_stop`` above the entry
   close, no re-entry before the rank leaves the short zone once); never short above
   ``short_max_si`` of shares sold short; a NaN score exits. Identical sleeves on both sides:
   one pass. Different sleeves: longs from the long score, then shorts from the short score
   with the longs masked out.
4. Sizing (``_weights``, equal): ``gross / 2 / slots`` a slot, constant while held.
5. A target change on a bar without an open waits for the name's next priced open
   (``adaptive_hedge._on_priced_opens``): the engine drops an order at a NaN price.

``combine="book"`` (the C_mix control) holds the single-sleeve books side by side instead: each
sleeve's own book at ``gross``, weighted ``w_k / Σ w_k`` and netted per name. ``short_gate=
"prof"`` (iteration I1) lets a name be shorted, and stay short, only while its profitability
rank is at most 0.5 (no profitability: not shortable); longs are untouched. ``neutral=
"sector"`` (iteration I2) ranks SI within sector × dollar-volume cells and takes MOM's residual
against the market AND the name's sector, so neither leg is an industry bet.

The four component files are loaded by path under prefixed module names and pinned by SHA-256
(:data:`SHA256`): this module never edits them, and a changed component refuses to load.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

#: Component key -> strategy folder (loaded by path, never imported, never edited).
COMPONENTS = {"p1": "informed_flow_ls_p1", "rm": "resid_momentum", "qj": "quality_junk",
              "ah": "adaptive_hedge"}
#: SHA-256 of each component's ``strategy.py`` as read on 2026-10-23 (the lab v2 finals).
SHA256 = {"p1": "a05e84d4176ef60bcf68c20eca41c4a6cb113e78d17089f37a3f3583b88dee67",
          "rm": "5cc54c69620cab771e1a268f2900ba0137a107220889794abb3ff6cc3bc52753",
          "qj": "1867bc4de9bf56f66e468bf19008b0f0bf93ae0417fdb7574b1d549c8e48fdc6",
          "ah": "52684a2118b5f6779fe3a0ddbd39c230fde2f1ee1a16cfd27251c013941b936c"}
SLEEVES = ("si", "mom", "prof")
COMBINES = ("rank", "book")
SHORT_GATES = ("", "prof")
NEUTRALS = ("", "sector")
#: ``neutral="sector"``: a (sector × liquidity) cell needs this many eligible names on a bar to
#: be ranked on its own; a smaller cell takes the plain liquidity-group rank (quality_junk's rule).
MIN_CELL = 5
#: ``short_gate="prof"``: a short needs a within-sector profitability rank at most this.
GATE_MAX = 0.5
_STRATEGIES = Path(__file__).resolve().parent.parent


@cache
def component(key: str):
    """A component's ``strategy.py``, loaded by path as ``_xsc_<key>`` (cached).

    Raises:
        RuntimeError: When the file's SHA-256 is not the pinned one.
    """
    path = _STRATEGIES / COMPONENTS[key] / "strategy.py"
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != SHA256[key]:
        raise RuntimeError(f"{path} changed: sha256 {digest}, pinned {SHA256[key]}")
    spec = importlib.util.spec_from_file_location(f"_xsc_{key}", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)   # the bytes that were hashed
    return module


def signals(panel, *, long_sleeves: str = "si+mom", short_sleeves: str = "si+mom",
            w_si: float = 1.0, w_mom: float = 1.0, w_prof: float = 1.0, combine: str = "rank",
            short_gate: str = "", neutral: str = "", gross: float = 0.90, slots: int = 15,
            q_in: float = 0.9,
            q_out: float = 0.8, max_hold: int = 126, score_smooth: int = 5,
            dtc_window: int = 42, neutral_groups: int = 5, short_stop: float = 1.0,
            short_max_si: float = 0.3, si_mid: float = 0.1, si_high: float = 0.2,
            borrow_gc: float = 0.005, borrow_mid: float = 0.03, borrow_high: float = 0.1,
            min_price: float = 5.0, retrim_k: float = 2.0, **_) -> dict:
    """Target weights (long > 0, short < 0), every name's borrow rate and the long cap.

    Args:
        panel: Daily panel with ``features`` ``short_interest`` and ``fund_shares`` (and
            ``fund_gp_ttm``, ``fund_assets``, ``fund_ni_ttm``, ``fund_equity``, ``sector``
            for the ``prof`` sleeve or gate).
        long_sleeves, short_sleeves: ``"+"``-joined sleeves of each side's ranking
            (:data:`SLEEVES`), e.g. ``"si+mom"``.
        w_si, w_mom, w_prof: Sleeve weights in a side's composite (relative).
        combine: ``"rank"`` (one composite ranking) or ``"book"`` (the single-sleeve books
            side by side, netted; needs ``long_sleeves == short_sleeves``).
        short_gate: ``""`` or ``"prof"`` (short only names with profitability rank <= 0.5).
        neutral: ``""`` or ``"sector"`` (SI ranked within sector × liquidity cells, MOM's
            residual against market + sector).
        gross: Long plus short exposure as a share of the book (<= 1: shorts lock cash).
        slots: Positions per side.
        q_in, q_out: Long entry / exit ranks; the short side mirrors them (``1 - q``).
        max_hold: Bars a position is held unless it still ranks in its entry zone.
        score_smooth: Bars a side's composite is averaged over before it is ranked.
        dtc_window: Bars days-to-cover is averaged over.
        neutral_groups: Dollar-volume groups the SI sleeve is ranked within.
        short_stop: A short whose close rises this far above its entry close is bought back.
        short_max_si: Short interest / shares outstanding above which a name is not shorted.
        si_mid, si_high: Crowding tiers of the borrow rate.
        borrow_gc, borrow_mid, borrow_high: Yearly borrow rates of the three tiers.
        min_price: Minimum close to be ranked (and to count in MOM's market return).
        retrim_k: A long grown to ``k ×`` its slot is trimmed back (engine cap, longs only).

    Returns:
        ``{"weights", "borrow", "retrim_k"}`` for the engine's portfolio mode.
    """
    p1 = component("p1")
    longs_k, shorts_k = long_sleeves.split("+"), short_sleeves.split("+")
    r = ranks(panel, set(longs_k) | set(shorts_k) | ({"prof"} if short_gate else set()),
              dtc_window=dtc_window, neutral_groups=neutral_groups, short_max_si=short_max_si,
              si_mid=si_mid, si_high=si_high, borrow_gc=borrow_gc, borrow_mid=borrow_mid,
              borrow_high=borrow_high, min_price=min_price, neutral=neutral)
    base = r["base"]
    shortable = base["shortable"]
    if short_gate == "prof":                 # junk only: no profitability = not shortable
        shortable = shortable & r["ranks"]["prof"].le(GATE_MAX)
    w = {"si": w_si, "mom": w_mom, "prof": w_prof}
    rules = {"q_in": q_in, "q_out": q_out, "slots": slots, "max_hold": max_hold,
             "short_stop": short_stop}

    def book(lo: list, sh: list):
        long_score = side_score(r["ranks"], lo, w, base["eligible"], score_smooth)
        short_score = long_score if sh == lo else side_score(r["ranks"], sh, w,
                                                             base["eligible"], score_smooth)
        held_l, held_s = select(long_score, short_score, panel.close, shortable,
                                two_pass=sh != lo, **rules)
        return p1._weights(held_l, held_s, base["beta"], gross=gross, long_share=0.5,
                           slots=slots, sizing="equal")

    if combine == "book":
        total = sum(w[k] for k in longs_k)
        weights = sum(w[k] / total * book([k], [k]) for k in longs_k)
    else:
        weights = book(longs_k, shorts_k)
    weights = component("ah")._on_priced_opens(weights, panel.open_)
    return {"weights": weights, "borrow": base["borrow"], "retrim_k": retrim_k}


_DEFAULTS = {k: v.default for k, v in inspect.signature(signals).parameters.items()
             if v.default is not inspect.Parameter.empty}


def _sleeves(value) -> list | None:
    """The sleeves of a ``"+"``-joined string, or ``None`` when it is not a non-empty set of
    known sleeves."""
    keys = value.split("+") if isinstance(value, str) else []
    ok = keys and len(set(keys)) == len(keys) and set(keys) <= set(SLEEVES)
    return keys if ok else None


def valid(params: dict) -> bool:
    """Reject parameter sets the book cannot express (missing keys read as ``signals()``'s
    defaults, so a partial dict is judged too)."""
    p = {**_DEFAULTS, **params}
    lo, sh = _sleeves(p["long_sleeves"]), _sleeves(p["short_sleeves"])
    if lo is None or sh is None:
        return False
    w = {"si": p["w_si"], "mom": p["w_mom"], "prof": p["w_prof"]}
    return (min(w.values()) >= 0 and sum(w[k] for k in lo) > 0 and sum(w[k] for k in sh) > 0
            and p["combine"] in COMBINES and (p["combine"] == "rank" or lo == sh)
            and p["short_gate"] in SHORT_GATES and p["neutral"] in NEUTRALS
            and 0.5 < p["q_out"] < p["q_in"] < 1.0 and 0.0 < p["gross"] <= 1.0
            and p["slots"] >= 1 and p["score_smooth"] >= 1 and p["max_hold"] >= 1
            and p["dtc_window"] >= 1 and p["short_stop"] > 0 and p["retrim_k"] > 1
            and 0.0 < p["si_mid"] < p["si_high"] <= p["short_max_si"])


def ranks(panel, sleeves, *, dtc_window: int = 42, neutral_groups: int = 5,
          short_max_si: float = 0.3, si_mid: float = 0.1, si_high: float = 0.2,
          borrow_gc: float = 0.005, borrow_mid: float = 0.03, borrow_high: float = 0.1,
          min_price: float = 5.0, neutral: str = "") -> dict:
    """Every sleeve's rank frame (high = long) and informed_flow_ls_p1's base frames.

    Returns:
        ``{"ranks": {sleeve: frame}, "base": p1.compose(...)}``; ``si`` always, ``mom`` and
        ``prof`` only when in ``sleeves``. ``base`` carries ``eligible``, ``shortable``,
        ``si_share``, ``borrow`` and ``beta``.
    """
    base = component("p1").compose(
        panel, dtc_window=dtc_window, skew_window=20, w_si=1.0, w_skew=0.0, skew_smooth=1,
        score_smooth=1, neutral_groups=neutral_groups, short_max_si=short_max_si,
        si_mid=si_mid, si_high=si_high, borrow_gc=borrow_gc, borrow_mid=borrow_mid,
        borrow_high=borrow_high, min_price=min_price, beta_window=126)
    out = {"si": base["ranks"]["si"]}
    if neutral == "sector":
        out["si"] = _sector_rank(panel, base, out["si"], neutral_groups)
    if "mom" in sleeves:
        factors = "market+sector" if neutral == "sector" else "market"
        parts = component("rm").compose(panel, min_price=min_price, factors=factors)
        month_end = parts["rank"].loc[parts["rebalance"].to_numpy()]
        out["mom"] = month_end.reindex(panel.close.index, method="ffill").where(
            base["eligible"])
    if "prof" in sleeves:
        out["prof"] = component("qj").compose(panel, min_price=min_price)["ranks"]["prof"]
    return {"ranks": out, "base": base}


def _sector_rank(panel, base: dict, plain, neutral_groups: int):
    """SI ranked within (sector × dollar-volume group) cells with p1's ``_sleeve_rank``; a cell
    with fewer than :data:`MIN_CELL` eligible names on a bar keeps the ``plain`` (liquidity-
    group) rank. A name without a sector code is its own sector."""
    p1 = component("p1")
    close = panel.close
    sector = (panel.features or {}).get("sector")
    codes = (pd.DataFrame(-1.0, index=close.index, columns=close.columns) if sector is None
             else sector.reindex(index=close.index, columns=close.columns).fillna(-1.0))
    groups = p1._liquidity_groups(panel, neutral_groups)
    cells = codes * 100.0 + (1.0 if groups is None else groups)
    raw, ok = base["raw"]["si"], base["eligible"]
    within = p1._sleeve_rank(raw, ok, cells)
    labels, present = cells.to_numpy(), raw.where(ok).notna().to_numpy()
    count = np.zeros(labels.shape)
    for g in np.unique(labels[~np.isnan(labels)]):
        inside = labels == g
        count = np.where(inside, (inside & present).sum(axis=1, keepdims=True), count)
    return within.where(count >= MIN_CELL, plain)


def side_score(rank: dict, sleeves: list, weights: dict, ok, smooth: int):
    """One side's ranking: ``Σ w_k × rank_k.fillna(0.5) / Σ w_k`` over ``sleeves``, averaged
    over ``smooth`` bars among the eligible names, then re-ranked (NaN where ineligible)."""
    total = sum(weights[k] for k in sleeves)
    c = sum(weights[k] * rank[k].fillna(0.5) for k in sleeves) / total
    return c.where(ok).rolling(smooth, min_periods=1).mean().where(ok).rank(axis=1, pct=True)


def select(long_score, short_score, close, shortable, *, two_pass: bool, **rules):
    """The held long and short frames (at the deciding bar) from informed_flow_ls_p1's book.

    One pass of ``_select`` when both sides read the same score; with ``two_pass`` the longs
    come from ``long_score`` alone and the shorts from ``short_score`` masked where a name is
    held long (so no name is ever on both sides).
    """
    p1 = component("p1")
    if not two_pass:
        longs, shorts, _ = p1._select(long_score, close, shortable, sides="both", **rules)
        return longs, shorts
    longs, _, _ = p1._select(long_score, close, shortable, sides="long", **rules)
    _, shorts, _ = p1._select(short_score.mask(longs), close, shortable, sides="short", **rules)
    return longs, shorts
