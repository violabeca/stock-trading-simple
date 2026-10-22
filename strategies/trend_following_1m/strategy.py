"""trend_following on 1-minute bars: the same signals, an intraday configuration.

The code is ``trend_following``'s, loaded from its folder rather than copied, so the two can
never drift apart. This folder owns only what differs: the ten stocks (``Morning10``), the
minute-scale search space, the session rules (entry blackouts at the open and close, flat at
the close or not) and the objective — ``alpha_cagr`` against volatility, because a book that
is flat overnight cannot out-earn a fully invested basket by construction. Every window is in
bars, i.e. minutes here. See README.md for the results and why they came out the way they did.
"""

from __future__ import annotations

from trading_simple import strategies

_base = strategies.load("trend_following")
signals = _base.signals
valid = _base.valid
indicators = _base.indicators
