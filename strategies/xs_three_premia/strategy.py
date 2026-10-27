"""xs_three_premia — short interest × residual momentum × within-sector profitability.

A cross-strategy of the proposals lab's last round (``strategies/proposals_lab``, lab v2). The
question: does a THIRD signal family, within-sector profitability, make the two-signal composite
of ``xs_si_mom_composite`` regime-robust? Profitability is the one lab signal with a TRAIN
premise, it is orthogonal to short interest, and it gained in 2026-H2, when SI, MOM and
adaptive_hedge all lost.

A thin wrapper: every rule is ``xs_si_mom_composite``'s ``signals`` (its ``strategy.py`` is
loaded by path as ``_xsp_core`` and pinned by SHA-256, :data:`CORE_SHA256`; a changed file
refuses to load). ``config.yaml`` carries the design: ``long_sleeves`` = ``short_sleeves`` =
``si+mom+prof`` with equal weights, so the composite is ``(si + mom + prof) / 3`` (each rank's
missing cells at 0.5), averaged over 5 bars and re-ranked among informed_flow_ls_p1's eligible
names, driving the core's 15 + 15 dollar-neutral book. The PROF sleeve itself, quality_junk's
``compose(panel, min_price=5)["ranks"]["prof"]`` (gross profit TTM / assets, ROE where gross
profit is untagged, ranked within sector × measure), lives in the core's ``ranks()``.

One knob of this folder's own, ``w_growth`` (iteration I2, RESEARCH_LOG.md §0.7; 0 = off, the
shipped value): quality_junk's profitability GROWTH rank (the 252-bar change of the name's own
ratio, ranked like profitability) added as a further rank to the LONG side's composite only;
the short side keeps the plain composite and the book runs the core's two passes (longs from the
long score, shorts from the short score with the longs masked). With ``w_growth = 0`` a call is
the core's call, unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
from functools import cache
from pathlib import Path

#: The composite core this folder wraps (loaded by path, never imported, never edited).
CORE = Path(__file__).resolve().parent.parent / "xs_si_mom_composite" / "strategy.py"
#: SHA-256 of the core's ``strategy.py`` as read on 2026-10-23 (its declared final, §8 there).
CORE_SHA256 = "c6061153760a6caffe65766c68a91bb9e258e134bd32953b75ec6a6a3341ef65"


@cache
def core():
    """``xs_si_mom_composite/strategy.py``, loaded by path as ``_xsp_core`` (cached).

    Raises:
        RuntimeError: When the file's SHA-256 is not :data:`CORE_SHA256`.
    """
    source = CORE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != CORE_SHA256:
        raise RuntimeError(f"{CORE} changed: sha256 {digest}, pinned {CORE_SHA256}")
    spec = importlib.util.spec_from_file_location("_xsp_core", CORE)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(CORE), "exec"), module.__dict__)   # the bytes that were hashed
    return module


def signals(panel, *, long_sleeves: str = "si+mom+prof", short_sleeves: str = "si+mom+prof",
            w_si: float = 1.0, w_mom: float = 1.0, w_prof: float = 1.0, w_growth: float = 0.0,
            combine: str = "rank", short_gate: str = "", neutral: str = "", gross: float = 0.90,
            slots: int = 15, q_in: float = 0.9, q_out: float = 0.8, max_hold: int = 126,
            score_smooth: int = 5, dtc_window: int = 42, neutral_groups: int = 5,
            short_stop: float = 1.0, short_max_si: float = 0.3, si_mid: float = 0.1,
            si_high: float = 0.2, borrow_gc: float = 0.005, borrow_mid: float = 0.03,
            borrow_high: float = 0.1, min_price: float = 5.0, retrim_k: float = 2.0,
            **_) -> dict:
    """Target weights (long > 0, short < 0), every name's borrow rate and the long cap.

    Every argument but ``w_growth`` is passed to ``xs_si_mom_composite.signals`` unchanged
    (its docstring documents them); the defaults here are this design's (``si+mom+prof``).

    Args:
        panel: Daily panel with ``features`` ``short_interest``, ``fund_shares``,
            ``fund_gp_ttm``, ``fund_assets``, ``fund_ni_ttm``, ``fund_equity`` and ``sector``.
        w_growth: Weight of quality_junk's profitability-growth rank, added to the LONG
            side's composite only (0 = off: the core's book exactly). Needs ``combine="rank"``.
        **_: Ignored (the lab passes nothing else; unknown keys never reach the core).

    Returns:
        ``{"weights", "borrow", "retrim_k"}`` for the engine's portfolio mode.
    """
    kw = {k: v for k, v in locals().items() if k in _CORE_KEYS}
    if not w_growth:
        return core().signals(panel, **kw)
    return _growth_book(panel, w_growth, kw)


_DEFAULTS = {k: v.default for k, v in inspect.signature(signals).parameters.items()
             if v.default is not inspect.Parameter.empty}
_CORE_KEYS = frozenset(_DEFAULTS) - {"w_growth"}


def valid(params: dict) -> bool:
    """The core's ``valid()`` plus ``w_growth >= 0`` (and ``combine="rank"`` when it is on).
    Missing keys read as this module's defaults, so a partial dict is judged too."""
    p = {**_DEFAULTS, **params}
    g = p["w_growth"]
    return (core().valid({k: v for k, v in p.items() if k in _CORE_KEYS})
            and isinstance(g, int | float) and g >= 0 and (g == 0 or p["combine"] == "rank"))


def _growth_book(panel, w_growth: float, p: dict) -> dict:
    """Iteration I2: the core's rank book with the growth rank in the LONG composite only.

    The same steps as the core's ``signals`` (``ranks`` → ``side_score`` → ``select`` →
    p1's equal ``_weights`` → ``_on_priced_opens``), with two passes because the sides read
    different sleeves; the growth rank is ``quality_junk.compose(..., w_growth=1)
    ["ranks"]["growth"]`` (NaN without a year of history: 0.5 in the composite).
    """
    c = core()
    lo, sh = p["long_sleeves"].split("+"), p["short_sleeves"].split("+")
    r = c.ranks(panel, set(lo) | set(sh) | ({"prof"} if p["short_gate"] else set()),
                dtc_window=p["dtc_window"], neutral_groups=p["neutral_groups"],
                short_max_si=p["short_max_si"], si_mid=p["si_mid"], si_high=p["si_high"],
                borrow_gc=p["borrow_gc"], borrow_mid=p["borrow_mid"],
                borrow_high=p["borrow_high"], min_price=p["min_price"], neutral=p["neutral"])
    base = r["base"]
    growth = c.component("qj").compose(panel, min_price=p["min_price"], w_growth=1.0)
    rank = {**r["ranks"], "growth": growth["ranks"]["growth"]}
    shortable = base["shortable"]
    if p["short_gate"] == "prof":
        shortable = shortable & rank["prof"].le(c.GATE_MAX)
    w = {"si": p["w_si"], "mom": p["w_mom"], "prof": p["w_prof"], "growth": w_growth}
    ok = base["eligible"]
    long_score = c.side_score(rank, [*lo, "growth"], w, ok, p["score_smooth"])
    short_score = c.side_score(rank, sh, w, ok, p["score_smooth"])
    held_l, held_s = c.select(long_score, short_score, panel.close, shortable, two_pass=True,
                              q_in=p["q_in"], q_out=p["q_out"], slots=p["slots"],
                              max_hold=p["max_hold"], short_stop=p["short_stop"])
    weights = c.component("p1")._weights(held_l, held_s, base["beta"], gross=p["gross"],
                                         long_share=0.5, slots=p["slots"], sizing="equal")
    weights = c.component("ah")._on_priced_opens(weights, panel.open_)
    return {"weights": weights, "borrow": base["borrow"], "retrim_k": p["retrim_k"]}
