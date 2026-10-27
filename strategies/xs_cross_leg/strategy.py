"""xs_cross_leg — short-interest longs, residual-momentum shorts (proposals lab cross-strategy).

Each signal trades only the side where its documented edge and its beta sit: the LONG leg is
informed_flow_ls_p1's short-interest rank (SI: low days-to-cover is good), the SHORT leg is
resid_momentum's month-end residual-momentum rank (MOM: the weakest residual 12-1 momentum).
Both are re-ranked among p1-eligible names after a 5-bar average, longs are chosen first, and
the short score is masked where a name is held long (no name is ever on both sides; a held
short that becomes a long is covered on that bar). Book: 15 + 15 equal slots of 3% (gross 0.90,
dollar-neutral), p1's bands, ``max_hold``, squeeze stop, crowding cap and borrow tiers.

This file is a thin wrapper: every rule of the book lives in
``strategies/xs_si_mom_composite/strategy.py`` (the ``long_sleeves`` / ``short_sleeves``
switches; config.yaml sets ``si`` / ``mom``; the reverse cross ``mom`` / ``si`` is the built-in
falsification control). The core is loaded by path as ``_xsl_core`` and pinned by SHA-256
(:data:`SHA256`): a changed core refuses to load, and the core pins its own four components the
same way. With ``short_sizing="equal"`` (the default) ``signals`` returns the core's book.

``short_sizing="beta"`` is the pre-registered iteration I2b (RESEARCH_LOG §0.7): the long leg
stays equal; each short slot is scaled at entry by the bar's median beta ÷ the name's beta
(informed_flow_ls_p1's ``"beta"`` sizing, clipped to 1/1.5 .. 1.5, frozen while held), with the
short side's nominal gross capped at half of ``gross``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path

#: The composite's strategy.py (another research folder: loaded by path, never edited).
CORE = Path(__file__).resolve().parent.parent / "xs_si_mom_composite" / "strategy.py"
#: Its SHA-256 as read on 2026-10-23 (the composite's declared-final code, prereg.log 13:38).
SHA256 = "c6061153760a6caffe65766c68a91bb9e258e134bd32953b75ec6a6a3341ef65"
SHORT_SIZINGS = ("equal", "beta")
#: I2b's slot bounds: a short is scaled between 1 / MAX_SIZE and MAX_SIZE (p1's ``max_size``).
MAX_SIZE = 1.5


def load_core():
    """The core module, executed from the exact bytes that were hashed.

    Raises:
        RuntimeError: When the file's SHA-256 is not :data:`SHA256`.
    """
    source = CORE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"{CORE} changed: sha256 {digest}, pinned {SHA256}")
    spec = importlib.util.spec_from_file_location("_xsl_core", CORE)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(CORE), "exec"), module.__dict__)
    return module


core = load_core()


def signals(panel, *, long_sleeves: str = "si", short_sleeves: str = "mom", w_si: float = 1.0,
            w_mom: float = 1.0, w_prof: float = 1.0, combine: str = "rank",
            short_gate: str = "", neutral: str = "", gross: float = 0.90, slots: int = 15,
            q_in: float = 0.9, q_out: float = 0.8, max_hold: int = 126,
            score_smooth: int = 5, dtc_window: int = 42, neutral_groups: int = 5,
            short_stop: float = 1.0, short_max_si: float = 0.3, si_mid: float = 0.1,
            si_high: float = 0.2, borrow_gc: float = 0.005, borrow_mid: float = 0.03,
            borrow_high: float = 0.1, min_price: float = 5.0, retrim_k: float = 2.0,
            short_sizing: str = "equal", **_) -> dict:
    """Target weights (long > 0, short < 0), every name's borrow rate and the long cap.

    Args:
        panel: Daily panel with ``features`` ``short_interest`` and ``fund_shares``.
        long_sleeves .. retrim_k: The core's knobs (``xs_si_mom_composite.signals``).
        short_sizing: ``"equal"`` (the core's book) or ``"beta"`` (I2b): a short slot scaled
            at entry by the bar's median beta ÷ the name's (p1's 126-bar beta to the
            equal-weight eligible names), clipped to ``1 / 1.5 .. 1.5`` (beta <= 0: 1 / 1.5; no
            beta: 1), frozen while held; a bar's new shorts are scaled down together when they
            would lift the short side's nominal gross above ``gross / 2``. Longs stay equal.

    Returns:
        ``{"weights", "borrow", "retrim_k"}`` for the engine's portfolio mode.
    """
    params = {k: v for k, v in locals().items() if k in core._DEFAULTS}   # the core's knobs
    if short_sizing == "equal":
        return core.signals(panel, **params)
    p1, ah = core.component("p1"), core.component("ah")
    lo, sh = long_sleeves.split("+"), short_sleeves.split("+")
    r = core.ranks(panel, set(lo) | set(sh), dtc_window=dtc_window,
                   neutral_groups=neutral_groups, short_max_si=short_max_si, si_mid=si_mid,
                   si_high=si_high, borrow_gc=borrow_gc, borrow_mid=borrow_mid,
                   borrow_high=borrow_high, min_price=min_price, neutral=neutral)
    base = r["base"]
    w = {"si": w_si, "mom": w_mom, "prof": w_prof}
    long_score = core.side_score(r["ranks"], lo, w, base["eligible"], score_smooth)
    short_score = core.side_score(r["ranks"], sh, w, base["eligible"], score_smooth)
    longs, shorts = core.select(long_score, short_score, panel.close, base["shortable"],
                                two_pass=True, q_in=q_in, q_out=q_out, slots=slots,
                                max_hold=max_hold, short_stop=short_stop)
    none = longs & False
    weights = (p1._weights(longs, none, base["beta"], gross=gross, long_share=0.5, slots=slots,
                           sizing="equal")
               + p1._weights(none, shorts, base["beta"], gross=gross / 2, long_share=0.0,
                             slots=slots, sizing="beta", max_size=MAX_SIZE))
    weights = ah._on_priced_opens(weights, panel.open_)
    return {"weights": weights, "borrow": base["borrow"], "retrim_k": retrim_k}


_DEFAULTS = {k: v.default for k, v in inspect.signature(signals).parameters.items()
             if v.default is not inspect.Parameter.empty}


def valid(params: dict) -> bool:
    """The core's ``valid()`` plus I2b's knob: ``short_sizing`` in :data:`SHORT_SIZINGS`, and
    ``"beta"`` only on one ranking per side (``combine: rank``) without the profitability gate
    (missing keys read as ``signals()``'s defaults)."""
    p = {**_DEFAULTS, **params}
    return bool(core.valid({k: v for k, v in p.items() if k in core._DEFAULTS})
                and p["short_sizing"] in SHORT_SIZINGS
                and (p["short_sizing"] == "equal"
                     or (p["combine"] == "rank" and p["short_gate"] == "")))
