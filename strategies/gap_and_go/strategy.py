"""Gap-and-go, intraday: opening-range breakouts on overnight gaps — and its inverse.

Thesis (``polarity: "go"``): an overnight gap on elevated opening participation
tends to CONTINUE intraday, and the break of the first-``or_minutes`` opening
range is the trigger that filters continuation from fade. Every other 1m
scanner in this repo blocks the first minutes of the session (``skip_open``);
this one trades ONLY a window just after the open — the untraded region where
liquid names make their biggest, fastest moves. That is deliberate and
double-edged: the open is also where spreads are widest (the classic intraday
spread U-shape), so the global 2 bps cost model is most understated exactly
where every entry here happens.

``polarity: "fade"`` takes the mirror position on the identical event: it buys
the gap-DOWN that breaks the range low and shorts the gap-UP that breaks the
range high. It exists because a market-demeaned event study over 200 names and
10 months says the continuation leg loses (long break, gap >= 2%, rvol >= 1.5,
60 bars: **-11.8 bps** of excess return, t = -3.3; the short leg is worse), and
that the effect strengthens with gap size and survives entry delays of 1-6
bars, so it is not bid-ask bounce. The two polarities are searched side by side
rather than one being assumed: the event study is in-sample evidence about a
*signal*, and only a backtest at stressed costs on the untouched holdout is
evidence about a *strategy*.

A long under ``go`` needs, at the signal close (short is the mirror on a gap
down; under ``fade`` the two legs swap):

1. **Gap** — today's open at least ``gap_min`` above the prior session's close
   (a plain fraction; prices are adjusted, so corporate actions do not
   manufacture gaps). ``0`` disables the gate, which is the "does ANY morning
   breakout work?" control.
2. **Participation and tradability** — the first-``or_minutes`` cumulative
   volume at least ``vol_mult``x its time-of-day expectation
   (``bt.volume_profile``, so the routine opening surge is already expected),
   the opening range's dollar volume above ``min_or_dollar_volume``, and the
   price above ``min_price``. All three are frozen at bar ``or_minutes - 1``.
3. **A slot on the morning watchlist** — see below.
4. **Breakout** — close above the opening-range high, frozen at bar
   ``or_minutes - 1`` and NaN while forming, so no signal bar is ever compared
   against a range that includes it. Signals live only in bars
   ``[or_minutes, or_minutes + entry_span]`` — the INVERSE of the siblings'
   ``skip_open``.

**The morning watchlist is what lets this run on hundreds of names.** One
shared wallet can fund only a handful of positions, so on a market-wide gap day
a naive rule emits far more signals than there is cash and the engine funds
whichever columns it reaches first — alphabetical luck deciding the book. Here
every symbol that clears the gates is instead ranked cross-sectionally at bar
``or_minutes - 1`` by ``rank_by``, and only the top ``max_candidates`` may trade
that session; the wallet deploys ``1 / max_candidates`` per entry, so a full
watchlist is a fully invested book and never more. The ranking uses ONLY
information frozen at bar ``or_minutes - 1`` — the gap (known at bar 0) and the
opening-range relative volume — which is exactly the pre-open watchlist a real
gap scanner builds, and it keeps the rule causal. Ranking *inside* the session
by which name eventually moved most would be look-ahead.

``entry_mode: "time"`` is the ablation control, not a thesis: enter at bar
``or_minutes`` on the gates alone (does the ORB trigger add anything over
buying every qualifying gap?). ``"break"`` with both gates off is the other
control (does ANY morning breakout work?).

Exits: ATR stop (trailing by default — continuation is a let-it-run thesis;
``sl_trail: false`` makes it a fixed bracket), optional ``tp_ratio`` target
expressed as a multiple of the stop, ``horizon`` time stop, and a mandatory
session-close flat — overnight risk is this strategy's own entry signal, never
its hold.

Engine semantics worth knowing: ``bt.atr``'s true range on each session's
first bar includes the overnight jump (|high - prev close|), so the Wilder EMA
is gap-inflated exactly on big-gap mornings — stops auto-widen with the
event's risk; a feature, documented rather than suppressed. And as everywhere
here, ``exit_after`` schedules a time exit for every raw signal, including
ones ignored while a position was open — the first-of-session edge below keeps
raw signals sparse, but phantom exits are nonzero (see squeeze_breakout).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel


def _session_scaffold(index: pd.DatetimeIndex, tz: str) -> tuple[pd.Series, pd.Series]:
    """Per-bar session key and 0-based bar number within each session.

    Args:
        index: The panel's (UTC) timestamp index.
        tz: Exchange-local timezone (``panel.tz``).

    Returns:
        ``(day, bar_no)`` — the exchange-local session date per bar, and the
        bar's position within its session (0 = the opening bar).
    """
    day = pd.Series(index.tz_convert(tz).date, index=index)
    return day, day.groupby(day).cumcount()


def signals(
    panel: Panel,
    *,
    or_minutes: int = 15,
    entry_span: int = 75,
    gap_min: float = 0.02,
    vol_mult: float = 1.5,
    profile_days: int = 10,
    entry_mode: str = "break",
    polarity: str = "go",
    rank_by: str = "gap",
    max_candidates: int = 8,
    min_price: float = 5.0,
    min_or_dollar_volume: float = 1e6,
    atr_window: int = 14,
    sl_mult: float = 2.0,
    sl_trail: bool = True,
    tp_ratio: float = 0.0,
    horizon: int = 120,
    direction: str = "both",
    eod_exit: bool = True,
) -> dict:
    """Build gap-gated opening-range-break entries and the exit ladder.

    Args:
        panel: Price panel (uses open/high/low/close/volume; 1m bars).
        or_minutes: Opening-range length in bars — the range is the high/low
            of bars ``0 .. or_minutes-1``, frozen at ``or_minutes - 1``.
        entry_span: Bars AFTER the range closes during which a raw signal may
            fire; the entry window is ``[or_minutes, or_minutes + entry_span]``.
            Expressed as an offset, not an absolute bar, so an entry window
            that ends before its own range closes is unrepresentable rather
            than rejected (the old absolute ``entry_until`` had the optimizer
            drawing — and wasting — combos that :func:`valid` then threw away).
        gap_min: Minimum |overnight gap| as a fraction of the prior session
            close (the up leg needs >= +gap_min, the down leg <= -gap_min).
            ``0`` disables the gate, leaving a pure morning breakout.
        vol_mult: Opening-participation gate — first-``or_minutes`` cumulative
            volume at least this multiple of its time-of-day expectation.
            The baseline already EXPECTS the opening surge, so this asks
            "elevated even for the open". ``0`` disables.
        profile_days: Trailing sessions in the volume baseline.
        entry_mode: ``"break"`` = trade the range break (the thesis) |
            ``"time"`` = enter at bar ``or_minutes`` on the gates alone (the
            ablation control; requires ``gap_min > 0``, see :func:`valid`).
        polarity: ``"go"`` = take the position the gap points at (continuation,
            the original thesis) | ``"fade"`` = take the opposite position on
            the identical event (buy the gap-down break, short the gap-up
            break). Searched, not assumed — see the module docstring.
        rank_by: How the morning watchlist is ordered — ``"gap"`` (|gap|),
            ``"rvol"`` (opening relative volume) or ``"gap_x_rvol"`` (the
            product). All three are known at bar ``or_minutes - 1``.
        max_candidates: Size of the morning watchlist. Only the top
            ``max_candidates`` gate-passing names by ``rank_by`` may trade a
            session, and each entry deploys ``1 / max_candidates`` of the
            shared wallet — so the slot count, not a separate cash fraction,
            is the single knob that sizes the book. This is what makes the
            rule scale-invariant: adding symbols improves *selection* instead
            of multiplying churn.
        min_price: Price floor, read at bar ``or_minutes - 1``. A quiet
            low-priced name's 1-minute spread is nothing like 2 bps.
        min_or_dollar_volume: Floor on the opening range's dollar volume
            (shares traded in bars ``0..or_minutes-1`` x the price at
            ``or_minutes - 1``). Per session, so it also drops a normally
            liquid name on a morning it simply is not trading.
        atr_window: ATR smoothing window in bars (stop scale; gap-inflated on
            gap mornings by construction — see module docstring).
        sl_mult: Stop distance in ATRs at entry.
        sl_trail: Trail the stop behind the best price (default — let the move
            run) or keep it fixed at the entry bracket.
        tp_ratio: Profit target as a MULTIPLE OF THE STOP (``2`` = take profit
            at twice the stop distance); ``0`` = no target. A ratio rather
            than an independent ATR distance so a degenerate short-gamma
            bracket cannot be drawn at all.
        horizon: Time stop — exit ``horizon`` bars after entry (the move
            never came).
        direction: Which POSITIONS are allowed: ``"long"`` | ``"short"`` |
            ``"both"``. Applied after ``polarity``, so under ``fade`` a
            ``"long"`` run trades gap-down bounces.
        eod_exit: Force flat on each session's final bar. A param only for
            contract-shape parity with the siblings — never searched, never
            off in a shipped config (holding overnight would trade against
            the thesis).

    Returns:
        Dict with ``entries`` / ``exits`` / ``short_entries`` / ``short_exits``,
        ``sl_stop`` (+ ``sl_trail``), optional ``tp_stop``, and ``cash_frac``.
    """
    index, columns = panel.close.index, panel.close.columns
    day, bar_no = _session_scaffold(index, panel.tz)

    def wide(mask: pd.Series) -> pd.DataFrame:
        """One bool per bar -> full (bar x symbol) frame (`.where` cannot broadcast)."""
        return pd.DataFrame(np.repeat(mask.to_numpy()[:, None], len(columns), axis=1),
                            index=index, columns=columns)

    # Prior session's close, taken ONE SESSION back and no further: a symbol that did
    # not trade yesterday gets NaN (and so cannot gap) instead of a week-old reference
    # price that manufactures a huge fake gap. Session 0 is NaN for everyone, so
    # nothing can trade before one full prior session.
    prev_close = panel.close.groupby(day).last().shift(1).reindex(day.to_numpy())
    prev_close.index = index
    # transform("first") is backward-looking within the session: known from bar 0 on.
    gap = panel.open_.groupby(day).transform("first") / prev_close - 1.0

    # Everything below is frozen at bar or_minutes-1 and held for the rest of the
    # session. The per-session ffill (not a plain ffill) is what keeps yesterday's
    # values from leaking into today's pre-formation bars.
    is_formed = bar_no == or_minutes - 1
    formed = wide(is_formed)
    freeze = lambda frame: frame.where(formed).groupby(day).ffill()  # noqa: E731
    or_high = freeze(panel.high.groupby(day).cummax())
    or_low = freeze(panel.low.groupby(day).cummin())
    or_price = freeze(panel.close)
    cum_volume = panel.volume.groupby(day).cumsum()
    rvol = None
    if vol_mult or rank_by != "gap":  # built once; both the gate and the ranking use it
        rvol = freeze(cum_volume / bt.volume_profile(
            panel.volume, panel.tz, days=profile_days).groupby(day).cumsum())

    # Gates. Warm-up sessions have a NaN volume baseline, so those comparisons fail
    # safe (NaN comparisons are False) rather than letting an unmeasured name through.
    gate = pd.DataFrame(True, index=index, columns=columns)
    if vol_mult:
        gate &= rvol >= vol_mult
    if min_price:
        gate &= or_price >= min_price
    if min_or_dollar_volume:
        gate &= freeze(cum_volume) * or_price >= min_or_dollar_volume
    up_gate = down_gate = gate
    if gap_min:                       # with the gate off, the break itself names the side
        up_gate, down_gate = gate & (gap >= gap_min), gate & (gap <= -gap_min)

    # The morning watchlist: rank the gate-passing names cross-sectionally at bar
    # or_minutes-1 and keep the top max_candidates for the session. Ranking on frozen
    # information only is what keeps this causal — it is the watchlist a scanner builds
    # at the open, not a hindsight pick of whichever name went on to move. Ranked on
    # the ~one row per session where the range closes, then held for the session.
    score = gap.abs() if rank_by in ("gap", "gap_x_rvol") else rvol
    if rank_by == "gap_x_rvol":
        score = score * rvol
    ranked = score.where(up_gate | down_gate)[is_formed.to_numpy()]
    on_watchlist = (ranked.rank(axis=1, ascending=False, method="first")
                    .reindex(index).groupby(day).ffill() <= max_candidates)
    up_gate, down_gate = up_gate & on_watchlist, down_gate & on_watchlist

    window = wide((bar_no >= or_minutes) & (bar_no <= or_minutes + entry_span))
    if entry_mode == "time":
        trigger_up = trigger_down = wide(bar_no == or_minutes)
    else:
        trigger_up, trigger_down = panel.close > or_high, panel.close < or_low
    # First trigger of the SESSION only: one break event, not a bar-by-bar barrage
    # while extended, and not a re-fire every time a chopping price recrosses the
    # level (which "not consecutive" would have allowed). Also keeps raw signals
    # sparse for exit_after (fewer phantom time exits).
    first = lambda raw: raw & (raw.groupby(day).cumsum() == 1)  # noqa: E731
    up_raw = first(trigger_up & up_gate & window)
    down_raw = first(trigger_down & down_gate & window)

    long_raw, short_raw = (down_raw, up_raw) if polarity == "fade" else (up_raw, down_raw)
    none = pd.DataFrame(False, index=index, columns=columns)
    entries = bt.next_bar(long_raw) if direction != "short" else none
    short_entries = bt.next_bar(short_raw) if direction != "long" else none

    exits = bt.exit_after(entries, horizon)
    short_exits = bt.exit_after(short_entries, horizon)
    if panel.bar != "1d" and eod_exit:
        end = bt.session_end(index, panel.tz).to_numpy()[:, None]
        exits = exits | end
        short_exits = short_exits | end

    rng = bt.atr(panel, atr_window)
    stop = (sl_mult * rng / panel.close).clip(lower=1e-4)
    out = {"entries": entries, "exits": exits,
           "short_entries": short_entries, "short_exits": short_exits,
           "sl_stop": stop, "sl_trail": bool(sl_trail),
           # One shared wallet, one slot per watchlist seat: a full watchlist is a
           # fully invested book and never more, whatever the universe size.
           "cash_frac": 1.0 / max_candidates}
    if tp_ratio:
        out["tp_stop"] = tp_ratio * stop
    return out


def valid(params: dict) -> bool:
    """Reject incoherent combos (tolerates partial dicts from sweep grids).

    Almost everything the old version rejected is now unrepresentable instead:
    ``entry_span`` is an offset from the range close, so the entry window
    cannot precede its own range, and ``tp_ratio`` is a multiple of the stop
    stepped in halves, so a degenerate short-gamma bracket cannot be drawn.
    Roughly a sixth of the v1 study's trials died on those two checks alone.

    Args:
        params: Candidate parameter combo (may hold only the swept keys).

    Returns:
        True unless ``entry_mode: "time"`` is paired with no gap gate — that
        combination is a daily timer buying everything at bar ``or_minutes``,
        a schedule rather than a thesis. Break mode with both gates off stays
        ALLOWED: that is the "any breakout works" ablation control.
    """
    return not (params.get("entry_mode", "break") == "time"
                and params.get("gap_min", 0.02) == 0)


def indicators(
    close: pd.Series,
    volume: pd.Series,
    *,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    or_minutes: int = 15,
    vol_mult: float = 1.5,
    profile_days: int = 10,
    tz: str = "America/New_York",
    **_,
) -> dict:
    """Per-symbol overlays: opening range + prior close, volume expectation.

    Args:
        close: One symbol's adjusted close.
        volume: One symbol's volume.
        high: One symbol's adjusted high. Pass it — :func:`signals` builds the
            range from the true high/low, so without it the drawn range is a
            close-based proxy and the chart shows a level the signal never used.
        low: One symbol's adjusted low (same reason).
        or_minutes: Opening-range length in bars.
        vol_mult: Participation multiple (drawn on the volume panel).
        profile_days: Volume-profile lookback sessions.
        tz: The panel's exchange timezone.
        **_: Extra params ignored.

    Returns:
        ``{"price", "volume", "volume_marks"}`` overlay series: the frozen
        range bounds and prior session close on price, the expected-volume
        baseline and trigger on volume, and the bars whose frozen opening
        relative volume cleared the gate.
    """
    day, bar_no = _session_scaffold(close.index, tz)
    formed = bar_no == or_minutes - 1
    freeze = lambda s: s.where(formed).groupby(day).ffill()  # noqa: E731
    or_high = freeze((close if high is None else high).groupby(day).cummax())
    or_low = freeze((close if low is None else low).groupby(day).cummin())
    prev_close = close.groupby(day).last().shift(1).reindex(day.to_numpy())
    prev_close.index = close.index
    base = bt.volume_profile(volume.to_frame("v"), tz, days=profile_days)["v"]
    rvol = freeze(volume.groupby(day).cumsum() / base.groupby(day).cumsum())
    return {
        "price": {
            f"OR high ({or_minutes}m)": or_high,
            f"OR low ({or_minutes}m)": or_low,
            "prior session close": prev_close,
        },
        "volume": {"expected volume": base, f"vol trigger ({vol_mult:g}x)": vol_mult * base},
        "volume_marks": rvol >= vol_mult,
    }
