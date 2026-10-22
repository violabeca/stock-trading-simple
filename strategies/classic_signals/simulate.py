"""The trade simulator of the classic_signals strategy: one position per stock at a time, traded
in the regular session only.

* **Entry**: a signal at a candle's close fills at the NEXT candle's open, in the same session.
  A signal on a session's last candle is dropped: the next open comes after a night, when the
  signal is stale. Signals while a trade is open are ignored (one trade per stock at a time).
* **The stop**, one of three kinds (``stop_kind``):

  - ``"moves"``: a trailing stop ``stop_moves`` typical moves away from the entry; after every
    candle it moves to the same distance behind the best price since the entry (the highest high
    of a long, the lowest low of a short), but only ever in the trade's favour;
  - ``"daily trailing"``: the same, ``stop_days`` typical DAILY moves (as known at the entry,
    ``ctx["daily_move"]``) behind it;
  - ``"daily fixed"``: ``stop_days`` typical daily moves from the entry price, never moved.

  The kind and the distances may change from candle to candle (levers re-tuned while trading): a
  trade keeps the ones it opened with.
* **Profit target** (optional): a price fixed at the entry (``targets``, e.g. the next level or a
  number of typical daily moves from the entry).
  A candle that reaches it closes the trade there; one that OPENS beyond it fills at that better
  open.
* **Exit**: a candle trades through the stop. If a candle OPENS beyond the stop (after a night, a
  weekend or a jump), the trade fills at that open, which is the real loss. Optionally the trade
  closes at the last close of its ``max_sessions``-th session (1 = never overnight), and at the
  close of a candle a rule marked ``flat`` (``decisions["flat"]``: the earnings guard, before a
  report). A trade still open when the data ends closes at the last close.
* **Size**: ``notional`` dollars per trade (fractional shares); with ``risk``, as many shares as
  lose ``risk`` dollars if the first stop is hit (gaps and costs aside), at most ``notional``
  dollars' worth: a calm and a wild stock, or quarter, then count alike.
* **Costs**: Per order a commission of
  ``per_share`` dollars a share, at least ``commission`` dollars (a $10,000 order of a $10 stock
  is 1,000 shares: $5 at $0.005 a share). Per fill ``slippage`` (a fraction of the price), at
  least ``min_slip`` dollars a share (half a cent: half the smallest possible spread), against
  the trade.

Inside a candle the order is conservative: the stop in force is checked against the candle's open,
then the target against it, then the stop against the high and low, then the target; only then
does the stop trail with that candle's best price. The inner loop is compiled with numba (the
optimizer will run it many times).
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd

STOP_KINDS = ("moves", "daily trailing", "daily fixed")      # the kernel's codes 0, 1, 2
RISK, MOST = 100.0, 20_000.0      # dollars a trade risks at its first stop; the most it may hold
REASONS = ("the trailing stop was hit", "the price opened beyond the stop",
           "the holding-time limit was reached", "the data ended",
           "the profit target was reached", "the earnings guard closed it before a report")
_WORDS = {1.0: "up", 0.0: "sideways", -1.0: "down"}


@numba.njit(cache=False)
def _kernel(open_, high, low, close, signal, move, mult, kind, dist, session, maxs, up, down,
            flat):
    """The trades, as arrays (see the module doc for the rules); ``kind``: the stop's code in
    :data:`STOP_KINDS`, ``dist``: the daily stops' distance in price."""
    n = open_.shape[0]
    entries = np.full(n, -1, np.int64)
    exits = np.full(n, -1, np.int64)
    sides = np.zeros(n, np.int8)
    reasons = np.zeros(n, np.int8)
    entry_px = np.full(n, np.nan)
    exit_px = np.full(n, np.nan)
    first_stop = np.full(n, np.nan)
    targets = np.full(n, np.nan)
    stop_path = np.full(n, np.nan)
    held = np.zeros(n, np.int8)
    k, pos, t_in, s_in, k_in = 0, 0, -1, -1, 0
    stop, best, target, m_in, d_in, ms_in = np.nan, np.nan, np.nan, np.nan, np.nan, 0
    for t in range(1, n):
        fresh = session[t] == session[t - 1]                  # no entry across a night
        gap = mult[t] * move[t] if kind[t] == 0 else dist[t]          # entry to stop
        if pos == 0 and signal[t - 1] != 0 and fresh and gap > 0:
            pos, t_in, s_in, ms_in = signal[t - 1], t, session[t], maxs[t]
            k_in, m_in, d_in = kind[t], mult[t], gap
            entries[k], sides[k], entry_px[k] = t, pos, open_[t]
            stop = open_[t] - pos * gap
            best = open_[t]
            target = up[t] if pos == 1 else down[t]
            first_stop[k], targets[k] = stop, target
        if pos == 0:
            continue
        stop_path[t], held[t] = stop, pos
        why, out = -1, np.nan
        if pos == 1:
            if t > t_in and open_[t] <= stop:
                why, out = 1, open_[t]
            elif t > t_in and open_[t] >= target:
                why, out = 4, open_[t]
            elif low[t] <= stop:
                why, out = 0, stop
            elif high[t] >= target:
                why, out = 4, target
        else:
            if t > t_in and open_[t] >= stop:
                why, out = 1, open_[t]
            elif t > t_in and open_[t] <= target:
                why, out = 4, open_[t]
            elif high[t] >= stop:
                why, out = 0, stop
            elif low[t] <= target:
                why, out = 4, target
        last_of_session = t == n - 1 or session[t + 1] != session[t]
        too_long = ms_in > 0 and session[t] - s_in + 1 >= ms_in
        if why < 0 and flat[t]:
            why, out = 5, close[t]
        elif why < 0 and last_of_session and too_long:
            why, out = 2, close[t]
        if why >= 0:
            exits[k], exit_px[k], reasons[k] = t, out, why
            k += 1
            pos = 0
            continue
        if k_in == 0 and move[t] > 0:                          # trailing, typical moves
            if pos == 1:
                best = max(best, high[t])
                stop = max(stop, best - m_in * move[t])
            else:
                best = min(best, low[t])
                stop = min(stop, best + m_in * move[t])
        elif k_in == 1:                                       # trailing, daily moves
            if pos == 1:
                best = max(best, high[t])
                stop = max(stop, best - d_in)
            else:
                best = min(best, low[t])
                stop = min(stop, best + d_in)
    if pos != 0:
        exits[k], exit_px[k], reasons[k] = n - 1, close[n - 1], 3
        k += 1
    return (entries[:k], exits[:k], sides[:k], entry_px[:k], exit_px[:k], first_stop[:k],
            targets[:k], reasons[:k], stop_path, held)


def costs(settings) -> dict:
    """:func:`run`'s size and cost arguments, the same in both notebooks: every trade risks
    :data:`RISK` dollars at its first stop (at most :data:`MOST` dollars' worth), and the repo's
    cost model (``config/settings.yaml``): a commission a share with a minimum an order, and a
    loss on every fill of half the spread plus slippage, at least half a cent a share."""
    c = settings.costs
    return {"notional": MOST, "risk": RISK, "commission": c.min_commission,
            "per_share": c.commission_per_share,
            "slippage": (c.spread_bps / 2 + c.slippage_bps) / 1e4, "min_slip": 0.005}


def run(ctx: dict, decisions: pd.DataFrame, *, stop_moves, stop_kind="moves", stop_days=1.0,
        max_sessions=0, targets: pd.DataFrame | None = None, notional: float = 10_000.0,
        risk: float | None = None, commission: float = 1.0, per_share: float = 0.0,
        slippage: float = 0.0002, min_slip: float = 0.0) -> dict:
    """Trade one stock's signals.

    Args:
        ctx: ``combine.make_context`` (its candles, typical move and trends).
        decisions: ``combine.decide`` (its ``signal`` column, and ``flat`` if there).
        stop_moves: Distance of the ``"moves"`` stop, in typical moves: a number, or one per
            candle (a trade keeps the one of its entry candle; so for the next two).
        stop_kind: The stop, one of :data:`STOP_KINDS`: a name, or one per candle.
        stop_days: Distance of a daily stop, in typical daily moves (``ctx["daily_move"]``): a
            number, or one per candle.
        max_sessions: Close a trade at the end of its n-th session (0: no limit; 1: no night):
            a number, or one per candle (a trade keeps the one of its entry candle).
        targets: The profit target a trade entered at each candle would get: ``long`` and
            ``short`` prices (NaN, or no frame: no target).
        notional: Dollars per trade; with ``risk``, the most a trade may hold.
        risk: Dollars a trade loses if its first stop is hit (None: every trade ``notional``).
        commission: The smallest commission per order, dollars (two orders per trade).
        per_share: Commission per share, dollars.
        slippage: Fraction of the price lost on each fill.
        min_slip: The smallest loss per share on each fill, dollars.

    Returns:
        ``trades`` (one row per trade: side, signal / entry / exit time, entry and exit price
        after slippage, the first stop, the target, the exit reason in words, sessions held,
        return % and P&L $ after all costs, the market condition at the entry: the stock's own
        daily trend at the previous close); ``stop`` (the stop in force at each candle, NaN when
        flat) and ``position`` (+1 long, -1 short, 0 flat at each candle).
    """
    c = ctx["candles"]
    session = pd.factorize(c.index.tz_localize(None).normalize())[0].astype(np.int64)
    move = ctx["move"].to_numpy(float)
    def per_candle(x, dtype):
        """A number, or one per candle, as one value per candle."""
        return (x.reindex(c.index).to_numpy(dtype) if isinstance(x, pd.Series)
                else np.full(len(c), x, dtype))

    mult = per_candle(stop_moves, float)
    kind = per_candle(stop_kind.map(STOP_KINDS.index) if isinstance(stop_kind, pd.Series)
                      else STOP_KINDS.index(stop_kind), np.int8)
    daily = ctx.get("daily_move")
    dist = per_candle(stop_days, float) * (np.full(len(c), np.nan) if daily is None
                                           else daily.to_numpy(float))
    flat = (decisions["flat"].to_numpy(bool) if "flat" in decisions
            else np.zeros(len(c), np.bool_))
    maxs = per_candle(max_sessions, np.int64)
    none = np.full(len(c), np.nan)
    up = none if targets is None else targets["long"].reindex(c.index).to_numpy(float)
    down = none if targets is None else targets["short"].reindex(c.index).to_numpy(float)
    (entries, exits, sides, epx, xpx, fstop, tgt, reasons, stop_path, held) = _kernel(
        c["open"].to_numpy(float), c["high"].to_numpy(float), c["low"].to_numpy(float),
        c["close"].to_numpy(float), decisions["signal"].to_numpy(np.int8), move, mult, kind,
        dist, session, maxs, up, down, flat)
    side = sides.astype(float)
    fill_in = epx + side * np.maximum(epx * slippage, min_slip)
    fill_out = xpx - side * np.maximum(xpx * slippage, min_slip)
    shares = notional / fill_in
    if risk is not None and len(epx):                 # the first stop risks ``risk`` dollars
        shares = np.minimum(risk / np.abs(epx - fstop), shares)
    pnl = side * shares * (fill_out - fill_in) - 2 * np.maximum(commission, shares * per_share)
    state = ctx["trend"]["stock"].to_numpy(float)[entries - 1] if len(entries) else np.array([])
    trades = pd.DataFrame({
        "stock": ctx["names"]["stock"], "side": np.where(side > 0, "long", "short"),
        "signal time": c.index[entries - 1], "entry time": c.index[entries],
        "exit time": c.index[exits], "entry price": fill_in, "exit price": fill_out,
        "first stop": fstop, "target": tgt, "exit reason": [REASONS[r] for r in reasons],
        "sessions": session[exits] - session[entries] + 1, "shares": shares,
        "return %": pnl / (shares * fill_in) * 100, "P&L $": pnl,
        "condition": [_WORDS.get(v, "not known") if np.isfinite(v) else "not known"
                      for v in state]})
    return {"trades": trades, "stop": pd.Series(stop_path, index=c.index),
            "position": pd.Series(held, index=c.index)}
