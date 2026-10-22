"""The indicators of the classic_signals strategy: the maths only, no trading decisions.

Every function reads candles and returns numbers per candle (or per day), using only candles up
to that one, never later ones. What the numbers MEAN for a trade (buy, sell, stay out) is decided
later by the strategy's rules.

* :func:`bollinger`: the Bollinger Bands, a moving average with a band above and below it;
* :func:`moving_average`: a simple or exponential moving average;
* :func:`typical_move`: how far a candle typically moves (the average true range), and
  :func:`explain_move`: the same, candle by candle, to draw how it is computed;
* :func:`trend_state`: is a daily chart going up, down or sideways;
* :func:`major_levels`: price levels where the daily chart turned several times;
  :func:`previous_range`: the previous session's high and low as levels of the same shape;
  :func:`nearest_levels`: the nearest level above and below a price, candle by candle;
* :func:`sessions_to_event`: how many sessions until the next event (a quarterly report);
* :func:`before_open`: what happened between the previous close and each open (the move, the
  night's and the pre-market's high and low, the volume against its usual level);
* :func:`previous_session`: what a daily number said at the close BEFORE a minute's session;
* :func:`available_at`: a coarser candle's value, known only once that candle has closed;
* :func:`session_context`: everything known at a session's open (previous high and low, trends).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TREND_WORDS = {1: "up", 0: "sideways", -1: "down"}


def bollinger(close: pd.Series, length: int = 20, width: float = 2.0) -> pd.DataFrame:
    """The Bollinger Bands of a price series.

    The middle line is the average close of the last ``length`` candles; the upper and lower
    bands sit ``width`` standard deviations above and below it, so they widen when prices swing
    and narrow when they calm down. The window runs across nights: the first candles of a session
    still read the last ones of the session before.

    Args:
        close: Closing prices.
        length: Candles in the average.
        width: Distance of each band from the middle line, in standard deviations.

    Returns:
        ``middle``, ``upper``, ``lower``; ``position``: where the close sits in the band (0 on
        the lower band, 1 on the upper band, below 0 or above 1 outside it); ``width_pct``: the
        distance between the bands as a percent of the middle line. NaN until ``length`` candles
        exist.
    """
    mid = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    upper, lower = mid + width * sd, mid - width * sd
    with np.errstate(invalid="ignore", divide="ignore"):
        position = (close - lower) / (upper - lower)
    return pd.DataFrame({"middle": mid, "upper": upper, "lower": lower,
                         "position": position.where(upper > lower),
                         "width_pct": (upper - lower) / mid * 100})


def moving_average(close: pd.Series, length: int, kind: str = "simple") -> pd.Series:
    """A moving average of the last ``length`` closes: ``"simple"`` weighs them all alike,
    ``"exponential"`` weighs recent closes more. NaN until ``length`` candles exist."""
    if kind == "simple":
        return close.rolling(length, min_periods=length).mean()
    if kind == "exponential":
        return close.ewm(span=length, adjust=False, min_periods=length).mean()
    raise ValueError(f"moving average kind {kind!r}: 'simple' or 'exponential'")


def true_range(candles: pd.DataFrame, *, across_nights: bool = True) -> pd.Series:
    """How far each candle moved, counting a jump from the previous close: the larger of the
    high and the previous close minus the smaller of the low and the previous close.

    ``across_nights=False`` (for intraday candles) leaves the overnight jump out: the first
    candle of a session counts only its own high minus low, so one gap at the open does not
    inflate the next hours.
    """
    prev = candles["close"].shift(1)
    if not across_nights:
        day = candles.index.tz_localize(None).normalize() if candles.index.tz is not None \
            else candles.index.normalize()
        prev = prev.where(pd.Series(day, index=candles.index).diff().eq(pd.Timedelta(0)))
    high = np.fmax(candles["high"], prev.fillna(candles["high"]))
    low = np.fmin(candles["low"], prev.fillna(candles["low"]))
    return high - low


def typical_move(candles: pd.DataFrame, length: int = 14, *,
                 across_nights: bool = True) -> pd.Series:
    """The typical move of a candle (the "average true range"): the average of the last
    ``length`` true ranges (:func:`true_range`), in price units. NaN until ``length`` exist."""
    return true_range(candles, across_nights=across_nights).rolling(
        length, min_periods=length).mean()


def explain_move(candles: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """The typical move of intraday candles, candle by candle, to draw how it is computed. The
    overnight gap is left out (:func:`true_range` with ``across_nights=False``).

    Args:
        candles: Intraday candles of one size (``data.resample``).
        length: Candles averaged.

    Returns:
        ``range`` (each candle's true range, in price) and ``range_pct`` (in percent of its
        close); ``typical`` (the average of the last ``length`` true ranges, this candle
        included) and ``typical_pct`` (in percent of this candle's close); ``known_pct``: the
        typical move the strategy knows WHILE a candle runs, i.e. the previous candle's
        ``typical_pct`` (a candle's own range is only known once it has closed).
    """
    tr = true_range(candles, across_nights=False)
    typical = tr.rolling(length, min_periods=length).mean()
    pct = typical / candles["close"] * 100
    return pd.DataFrame({"range": tr, "range_pct": tr / candles["close"] * 100,
                         "typical": typical, "typical_pct": pct, "known_pct": pct.shift(1)})


def trend_state(days: pd.DataFrame, *, length: int = 50, slope_days: int = 10,
                flat: float = 0.5) -> pd.DataFrame:
    """Whether a daily chart is going up, down or sideways, judged at each day's close.

    * **up**: the close is above its ``length``-day average AND that average rose over the last
      ``slope_days`` days by more than ``flat`` x a typical day's move;
    * **down**: the close is below the average AND the average fell by more than that;
    * **sideways**: everything else (a flat average, or price and average disagreeing).

    The typical day's move scales the "flat" threshold to each symbol, so a calm fund and a
    jumpy stock are judged alike.

    Returns:
        ``state`` (1 up, 0 sideways, -1 down; NaN until enough days exist), ``average``,
        ``rise`` (the average's change over ``slope_days``) and ``typical_move``.
    """
    close = days["close"]
    average = moving_average(close, length)
    rise = average - average.shift(slope_days)
    move = typical_move(days, 14, across_nights=True)
    up = (close > average) & (rise > flat * move)
    down = (close < average) & (rise < -flat * move)
    state = pd.Series(np.select([up, down], [1.0, -1.0], 0.0), index=days.index)
    state[average.isna() | rise.isna() | move.isna()] = np.nan
    return pd.DataFrame({"state": state, "average": average, "rise": rise,
                         "typical_move": move})


def implied_move(chains: pd.DataFrame, *, min_dte: int = 2) -> pd.Series:
    """The daily move the options market expects, per date, in percent of the price: the
    at-the-money straddle (the call and the put at the strike nearest the price, mid quotes) of
    the nearest expiry at least ``min_dte`` calendar days out, as a daily standard deviation.
    A straddle is worth about √(2/π) ≈ 0.798 × price × σ × √(trading days to expiry), so
    σ = straddle ÷ (0.798 × price × √days). An event before that expiry (a report) raises it. A
    date without a two-sided quote of both at that strike is left out.

    Args:
        chains: ``data.load_chains``.
        min_dte: The fewest calendar days to expiry (the last days' prices are erratic).
    """
    d = chains[(chains["bid"] > 0) & (chains["ask"] > 0) & (chains["dte"] >= min_dte)]
    out = {}
    for day, g in d.groupby("date"):
        g = g[g["expiration"] == g["expiration"].min()]
        price = float(g["underlyingPrice"].iloc[0])
        at = g[g["strike"] == g.loc[(g["strike"] - price).abs().idxmin(), "strike"]]
        mid = ((at["bid"] + at["ask"]) / 2).groupby(at["side"]).first()
        if {"call", "put"} <= set(mid.index):
            days_left = float(g["dte"].iloc[0]) * 252 / 365
            out[pd.Timestamp(day)] = float(mid["call"] + mid["put"]) / (
                0.798 * price * np.sqrt(days_left)) * 100
    return pd.Series(out, dtype=float).sort_index()


RATE = 0.04      # the yearly risk-free rate option prices are read with (US bills, 2025-2026)


def _ncdf(x):
    """The standard normal distribution function (Abramowitz & Stegun 7.1.26: within 1.5e-7)."""
    z = np.abs(x) / np.sqrt(2)
    t = 1 / (1 + 0.3275911 * z)
    erf = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1 + np.sign(x) * erf)


def _bs_price(s, k, t, sigma, call, r=RATE):
    """European option value by Black-Scholes, without dividends (arrays welcome)."""
    sq = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r + sigma ** 2 / 2) * t) / sq
    disc = k * np.exp(-r * t)
    return np.where(call, s * _ncdf(d1) - disc * _ncdf(d1 - sq),
                    disc * _ncdf(sq - d1) - s * _ncdf(-d1))


def black_scholes(s, k, t, sigma, call, r=RATE) -> dict:
    """European option value and Greeks (no dividends; arrays welcome).

    Args:
        s, k: The stock's price and the strike.
        t: Years to expiry.
        sigma: Yearly volatility (0.3 = 30%).
        call: True for a call, False for a put.
        r: The yearly risk-free rate.

    Returns:
        ``price``; ``delta`` (the option's move per $1 of the stock); ``gamma`` (delta's change
        per $1); ``theta`` (the value lost per calendar day); ``vega`` (the value gained per
        volatility point, 0.01).
    """
    sq = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r + sigma ** 2 / 2) * t) / sq
    d2 = d1 - sq
    disc = k * np.exp(-r * t)
    pdf = np.exp(-d1 ** 2 / 2) / np.sqrt(2 * np.pi)
    carry = np.where(call, r * disc * _ncdf(d2), -r * disc * _ncdf(-d2))
    return {"price": _bs_price(s, k, t, sigma, call, r),
            "delta": np.where(call, _ncdf(d1), _ncdf(d1) - 1), "gamma": pdf / (s * sq),
            "theta": (-s * pdf * sigma / (2 * np.sqrt(t)) - carry) / 365,
            "vega": s * pdf * np.sqrt(t) / 100}


def implied_vol(price, s, k, t, call, r=RATE, *, lo: float = 0.01, hi: float = 5.0,
                steps: int = 50):
    """The yearly volatility at which Black-Scholes gives ``price`` (arrays welcome), by halving
    ``[lo, hi]`` ``steps`` times; NaN where no volatility in that range gives the price (a quote
    below the option's value at ``lo``, a stale one)."""
    price = np.asarray(price, float)
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    for _ in range(steps):
        m = (a + b) / 2
        high = _bs_price(s, k, t, m, call, r) > price
        a, b = np.where(high, a, m), np.where(high, m, b)
    inside = (price > _bs_price(s, k, t, lo, call, r)) & (price < _bs_price(s, k, t, hi, call, r))
    return np.where(inside, (a + b) / 2, np.nan)


def _atm_iv(g: pd.DataFrame, price: float) -> float:
    """The implied volatility at the strike nearest the price, in percent (NaN if unknown)."""
    if not len(g):
        return np.nan
    at = g[g["strike"] == g.loc[(g["strike"] - price).abs().idxmin(), "strike"]]
    return float(at["iv"].mean() * 100)


def option_features(chains: pd.DataFrame, *, wall_days: int = 45) -> pd.DataFrame:
    """What a stock's end-of-day option chains say, one row per date (judged at its close).

    Every quoted contract (a bid and an ask) gets its implied volatility (:func:`implied_vol` of
    the mid quote) and Greeks (:func:`black_scholes`). A strike's volatility is read from its
    out-of-the-money side (the call above the price, the put below; in-the-money quotes are
    noisier and American puts carry an early-exercise premium) and used for both sides.

    Args:
        chains: ``data.load_chains``.
        wall_days: The expiries (calendar days) whose open interest makes the walls.

    Returns:
        ``move`` (:func:`implied_move`, %), ``front iv`` (at the money, nearest expiry at least
        2 days out, %), ``iv`` (at the money, the expiry nearest 30 days among 7 to 60, %),
        ``term`` (``front iv`` ÷ ``iv``: above 1, an event priced before the later expiry),
        ``skew`` (that expiry's 25-delta put volatility minus its 25-delta call volatility,
        points: what downside protection costs over upside), ``put/call volume`` and
        ``put/call oi`` (all expiries; NaN when one side has none: no ratio to read),
        ``gamma`` (dealers' gamma in dollars per 1% move, if they hold the calls customers
        sold and the puts customers bought: + call gamma − put gamma,
        × open interest × 100 × price² × 1%), ``call wall`` / ``put wall`` (the strike above /
        below the price with the most call / put open interest over the next ``wall_days``),
        ``theta`` and ``vega`` (all open interest: dollars lost per day, gained per volatility
        point) and ``price`` (the stock's price in the chain: before a split, in the old shares).
    """
    d = chains[(chains["bid"] > 0) & (chains["ask"] > 0)].copy()
    if not len(d):
        return pd.DataFrame()
    d["call"] = d["side"] == "call"
    d["t"] = np.maximum(d["dte"], 0.5) / 365
    s, k = d["underlyingPrice"].to_numpy(float), d["strike"].to_numpy(float)
    own = implied_vol((d["bid"] + d["ask"]).to_numpy(float) / 2, s, k, d["t"].to_numpy(float),
                      d["call"].to_numpy())
    d["own"] = own
    d["otm"] = np.where(d["call"], k >= s, k <= s)
    key = ["date", "expiration", "strike"]
    otm = d[d["otm"]].groupby(key)["own"].first().rename("otm iv")
    d = d.join(otm, on=key)
    d["iv"] = d["otm iv"].fillna(d["own"])
    d = d[d["iv"].notna()]
    g = black_scholes(d["underlyingPrice"].to_numpy(float), d["strike"].to_numpy(float),
                      d["t"].to_numpy(float), d["iv"].to_numpy(float), d["call"].to_numpy())
    for name in ("delta", "gamma", "theta", "vega"):
        d[name] = g[name]
    oi = d["openInterest"].to_numpy(float) * 100
    sp = d["underlyingPrice"].to_numpy(float)
    d["dealer gamma"] = np.where(d["call"], 1.0, -1.0) * d["gamma"] * oi * sp * sp * 0.01
    d["theta $"], d["vega $"] = d["theta"] * oi, d["vega"] * oi
    move = implied_move(chains)
    rows = {}
    for day, x in d.groupby("date"):
        price = float(x["underlyingPrice"].iloc[0])
        front = x[x["dte"] >= 2]
        front = front[front["expiration"] == front["expiration"].min()] if len(front) else front
        mid = x[x["dte"].between(7, 60)]
        if len(mid):
            mid = mid[mid["expiration"] == mid.loc[(mid["dte"] - 30).abs().idxmin(),
                                                   "expiration"]]
        puts, calls = mid[~mid["call"]], mid[mid["call"]]
        skew = (puts.loc[(puts["delta"] + 0.25).abs().idxmin(), "iv"]
                - calls.loc[(calls["delta"] - 0.25).abs().idxmin(), "iv"]) * 100 \
            if len(puts) and len(calls) else np.nan
        near = x[x["dte"] <= wall_days]
        up = near[near["call"] & (near["strike"] > price)].groupby("strike")["openInterest"].sum()
        dn = near[~near["call"] & (near["strike"] < price)].groupby("strike")["openInterest"].sum()
        vol = x.groupby("call")["volume"].sum()
        oi_ = x.groupby("call")["openInterest"].sum()
        front_iv, iv = _atm_iv(front, price), _atm_iv(mid, price)
        rows[pd.Timestamp(day)] = {
            "front iv": front_iv, "iv": iv, "term": front_iv / iv if iv else np.nan,
            "skew": skew,
            "put/call volume": vol[False] / vol[True] if min(vol.get(False, 0),
                                                              vol.get(True, 0)) > 0 else np.nan,
            "put/call oi": oi_[False] / oi_[True] if min(oi_.get(False, 0),
                                                         oi_.get(True, 0)) > 0 else np.nan,
            "gamma": float(x["dealer gamma"].sum()),
            "call wall": float(up.idxmax()) if len(up) else np.nan,
            "put wall": float(dn.idxmax()) if len(dn) else np.nan,
            "theta": float(x["theta $"].sum()), "vega": float(x["vega $"].sum()),
            "price": price}
    out = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    out.insert(0, "move", move.reindex(out.index))
    return out


def options_ratio(implied: pd.Series, days: pd.DataFrame, *, length: int = 20) -> pd.Series:
    """The options' expected daily move (:func:`implied_move`) ÷ the stock's own recent daily
    moves (the standard deviation of its last ``length`` daily close-to-close returns, in
    percent), per date, judged at that date's close. About 1.1 on an ordinary day (options cost
    a little more than the moves they insure), 2 to 5 before an expected event."""
    realized = np.log(days["close"]).diff().rolling(length, min_periods=length).std() * 100
    return (implied / realized.reindex(implied.index)).dropna()


def previous_session(values: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """For every minute of ``index``, the daily value of the LAST SESSION BEFORE that minute's
    session: what was known at the open. A daily trend judged at Tuesday's close guides
    Wednesday's minutes, never Tuesday's own.

    Args:
        values: A daily series indexed by session date.
        index: New York minute times.
    """
    days = index.tz_localize(None).normalize() if index.tz is not None else index.normalize()
    known = values.dropna().sort_index()
    pos = known.index.searchsorted(days, side="left") - 1      # the last date before the day
    out = np.where(pos >= 0, known.to_numpy()[np.clip(pos, 0, None)], np.nan) \
        if len(known) else np.full(len(index), np.nan)
    return pd.Series(out, index=index)


def before_open(minutes: pd.DataFrame, *, usual_days: int = 20) -> pd.DataFrame:
    """What happened between the previous session's close and each session's open: the
    after-hours, the overnight (where stored) and the pre-market. All of it is known at 09:30.

    Only minutes with a trade count: a minute without one just repeats an older price.

    Args:
        minutes: 1-minute candles WITH the hours outside the regular session
            (``data.load_minutes(..., extended=True)``: needs ``regular`` and ``traded``).
        usual_days: Sessions behind the usual volume.

    Returns:
        One row per session (date): ``previous close`` (the last regular-session price of the
        session before); ``last`` (the last trade before the open); ``move %`` (from the
        previous close to that last trade: the whole night); ``night high`` and ``night low``
        (the highest and lowest trade since the previous close); ``pre-market high`` and
        ``pre-market low`` (04:00-09:30 only, the levels traders watch at the open);
        ``volume`` (shares traded since the previous close) and ``volume vs usual`` (divided by
        its average over the previous ``usual_days`` sessions); ``minutes traded``.
    """
    t = minutes.index
    naive = t.tz_localize(None) if t.tz is not None else t
    minute = np.asarray(t.hour * 60 + t.minute)
    regular = minutes["regular"].to_numpy(bool)
    reg = minutes[regular]
    reg_day = naive[regular].normalize()
    opens = pd.Series(reg.index, index=reg_day).groupby(level=0).first()
    sessions = pd.DatetimeIndex(opens.index)
    closes = reg["close"].groupby(reg_day).last().reindex(sessions)
    outside = ~regular & minutes["traded"].to_numpy(bool)
    nxt = np.searchsorted(pd.DatetimeIndex(opens.to_numpy()).asi8, t[outside].asi8, side="right")
    inside = nxt < len(sessions)                    # minutes after the last open lead nowhere
    night = minutes[outside][inside]
    lead = sessions[nxt[inside]]                    # the session each outside minute leads into
    g = night.groupby(lead)
    out = pd.DataFrame({"last": g["close"].last(), "night high": g["high"].max(),
                        "night low": g["low"].min(), "volume": g["volume"].sum(),
                        "minutes traded": g.size()}).reindex(sessions)
    pre_mask = outside & (minute >= 240) & (minute < 570)
    pg = minutes[pre_mask].groupby(naive[pre_mask].normalize())
    out["pre-market high"] = pg["high"].max().reindex(sessions)
    out["pre-market low"] = pg["low"].min().reindex(sessions)
    out.insert(0, "previous close", closes.shift(1))
    out["move %"] = (out["last"] / out["previous close"] - 1) * 100
    out["volume"] = out["volume"].fillna(0.0)
    out["minutes traded"] = out["minutes traded"].fillna(0).astype(int)
    usual = out["volume"].rolling(usual_days, min_periods=usual_days // 2).mean().shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["volume vs usual"] = out["volume"] / usual
    out.index.name = "day"
    return out


def available_at(values: pd.Series, size: int, index: pd.DatetimeIndex) -> pd.Series:
    """Values of ``size``-minute candles as known at each time of ``index``: a candle's value
    only becomes known when the candle has CLOSED (its opening time + ``size`` minutes), so a
    minute never sees the candle it belongs to.

    Args:
        values: One value per candle, indexed by the candle's opening time.
        size: Minutes per candle.
        index: The times at which the values are read (e.g. 1-minute opening times).
    """
    known = values.dropna()
    closes = known.index + pd.Timedelta(minutes=size)
    pos = closes.searchsorted(index, side="right") - 1
    out = np.where(pos >= 0, known.to_numpy()[np.clip(pos, 0, None)], np.nan) \
        if len(known) else np.full(len(index), np.nan)
    return pd.Series(out, index=index)


def session_context(days: pd.DataFrame, trends: dict[str, pd.Series], sessions) -> pd.DataFrame:
    """What was known at the open of each session: the previous session's high and low, and each
    daily trend as judged at the previous close.

    Args:
        days: The stock's daily candles.
        trends: Label -> daily trend state (``trend_state(...)["state"]``).
        sessions: The session dates.

    Returns:
        ``day``, ``previous high``, ``previous low`` and one column of words per trend.
    """
    sessions = pd.DatetimeIndex(sessions).normalize()
    out = pd.DataFrame({"day": sessions})
    out["previous high"] = previous_session(days["high"], sessions).to_numpy()
    out["previous low"] = previous_session(days["low"], sessions).to_numpy()
    for label, state in trends.items():
        known = previous_session(state, sessions).to_numpy()
        out[label] = [TREND_WORDS.get(v, "not known yet") if np.isfinite(v) else "not known yet"
                      for v in known]
    return out


def pivots(days: pd.DataFrame, span: int = 5) -> pd.DataFrame:
    """The daily chart's turning points: a day whose high is the highest of the ``span`` days
    before and after it (a swing high), or whose low is the lowest (a swing low).

    A turning point is only CONFIRMED ``span`` days later: until those days have passed, nobody
    could know that the price would not go further. ``confirmed`` is the session at whose close
    it became known.

    Returns:
        ``day``, ``confirmed``, ``kind`` ("high" or "low"), ``price``, ``position`` (the day's
        row number in ``days``) and ``span``.
    """
    high, low = days["high"].to_numpy(float), days["low"].to_numpy(float)
    rows = []
    for i in range(span, len(days) - span):
        window = slice(i - span, i + span + 1)
        if high[i] == np.nanmax(high[window]):
            rows.append((i, "high", high[i]))
        if low[i] == np.nanmin(low[window]):
            rows.append((i, "low", low[i]))
    out = pd.DataFrame(rows, columns=["position", "kind", "price"])
    out["day"] = days.index[out["position"]] if len(out) else pd.DatetimeIndex([])
    out["confirmed"] = days.index[out["position"] + span] if len(out) else pd.DatetimeIndex([])
    out["span"] = span
    return out[["day", "confirmed", "kind", "price", "position", "span"]]


def _groups(price: np.ndarray, tol: float) -> list[np.ndarray]:
    """Positions into ``price`` grouped by closeness: sorted by price, a new group starts
    wherever the gap to the previous price is larger than ``tol``."""
    order = np.argsort(price, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(price[order]) > tol) + 1)


def _find_groups(price: np.ndarray, kind: np.ndarray, idx: np.ndarray, tol: float, *,
                 mix: bool) -> list[np.ndarray]:
    """The groups among the turning points ``idx`` (positions into ``price``), sorted by their
    average price: all of them together (``mix``), or the swing highs and the swing lows
    apart."""
    parts = [idx] if mix else [idx[kind[idx] == "high"], idx[kind[idx] == "low"]]
    groups = [part[g] for part in parts if len(part) for g in _groups(price[part], tol)]
    return sorted(groups, key=lambda g: float(price[g].mean()))


def _level_row(day, price: np.ndarray, kind: np.ndarray, when: np.ndarray) -> dict:
    """One major level from the prices, kinds and days of its turning points."""
    return {"day": day, "price": float(price.mean()), "touches": len(price),
            "highs": int((kind == "high").sum()), "lows": int((kind == "low").sum()),
            "bottom": float(price.min()), "top": float(price.max()),
            "last_touch": pd.Timestamp(when.max())}


LEVEL_COLUMNS = {"day": "datetime64[ns]", "price": float, "touches": int, "highs": int,
                 "lows": int, "bottom": float, "top": float, "last_touch": "datetime64[ns]"}


def _levels_frame(rows: list[dict]) -> pd.DataFrame:
    """Level rows as a frame with fixed column types (also when there is no row)."""
    return pd.DataFrame(rows, columns=list(LEVEL_COLUMNS)).astype(LEVEL_COLUMNS)


def major_levels(days: pd.DataFrame, *, span: int = 5, lookback: int = 120,
                 tolerance: float = 0.5, min_touches: int = 2, mix: bool = True) -> pd.DataFrame:
    """The major price levels known before each session opens.

    The turning points (:func:`pivots`) confirmed by the previous close and no older than
    ``lookback`` sessions are sorted by price; neighbours closer than ``tolerance`` x a typical
    day's move are grouped. A group of at least ``min_touches`` turning points is a major level:
    the price turned there several times. Whether it acts as resistance (above the price) or
    support (below) depends on where the price is at the moment.

    Args:
        days: Daily candles.
        span: Days on each side that define a turning point.
        lookback: How many sessions back a turning point still counts.
        tolerance: How close (in typical daily moves) turning points must be to form one level.
        min_touches: Turning points a level needs to count as major.
        mix: Let swing highs and swing lows form one level together (an old floor that breaks
            often becomes a ceiling); False keeps ceilings (from highs) and floors (from lows)
            apart.

    Returns:
        One row per session and level: ``day`` (the session it applies to), ``price`` (the
        group's average), ``touches``, ``highs`` and ``lows`` (how many turning points of each
        kind), ``bottom`` and ``top`` (the group's price range) and ``last_touch`` (the most
        recent turning point's day).
    """
    piv = pivots(days, span)
    move = typical_move(days, 14, across_nights=True).to_numpy(float)
    pos, conf = piv["position"].to_numpy(), piv["position"].to_numpy() + span
    price, kind, day = piv["price"].to_numpy(float), piv["kind"].to_numpy(), piv["day"].to_numpy()
    rows = []
    for t in range(1, len(days)):
        known = (conf <= t - 1) & (pos >= t - 1 - lookback)
        tol = tolerance * move[t - 1]
        if known.sum() < min_touches or not np.isfinite(tol):
            continue
        for g in _find_groups(price, kind, np.flatnonzero(known), tol, mix=mix):
            if len(g) >= min_touches:
                rows.append(_level_row(days.index[t], price[g], kind[g], day[g]))
    return _levels_frame(rows)


def previous_range(days: pd.DataFrame) -> pd.DataFrame:
    """The previous session's high and low as two levels of each session, in the shape of
    :func:`major_levels`: ``day`` (the session they apply to), ``price`` and ``kind``
    ("previous high" or "previous low"). Both are known at the session's open."""
    prev = days[["high", "low"]].shift(1).dropna()
    return pd.concat([pd.DataFrame({"day": prev.index, "price": prev[c].to_numpy(),
                                    "kind": f"previous {c}"}) for c in ("high", "low")],
                     ignore_index=True)


def nearest_levels(prices: pd.Series, levels: pd.DataFrame, *, gap=0.0) -> pd.DataFrame:
    """For every candle, the nearest level at or above ``price + gap`` and the nearest at or
    below ``price - gap``, among the levels known for that candle's session (the levels a
    session uses are fixed at its open, so no later candle is read).

    Args:
        prices: The prices to measure from (each candle's close, or its open), on New York times.
        levels: One row per session and level: ``day`` (the session it applies to) and
            ``price`` (:func:`major_levels`, :func:`previous_range`, or both stacked).
        gap: How far a level must at least be: a number, or a Series on the same times.

    Returns:
        ``above`` and ``below``: the level prices, NaN where there is none (or no price).
    """
    idx = prices.index
    if not len(idx):
        return pd.DataFrame({"above": [], "below": []}, index=idx, dtype=float)
    day = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    p = prices.to_numpy(float)
    g = gap.reindex(idx).to_numpy(float) if isinstance(gap, pd.Series) else np.full(len(p), gap)
    above, below = np.full(len(p), np.nan), np.full(len(p), np.nan)
    by_day = ({pd.Timestamp(d): np.sort(grp["price"].to_numpy(float))
               for d, grp in levels.groupby("day")} if len(levels) else {})
    cuts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1, len(p)]    # candles of one session
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        lv = by_day.get(day[a])
        if lv is None or not len(lv):
            continue
        hi, lo = p[a:b] + g[a:b], p[a:b] - g[a:b]
        i = np.searchsorted(lv, hi, side="left")                  # the first level >= hi
        j = np.searchsorted(lv, lo, side="right") - 1             # the last level <= lo
        above[a:b] = np.where(i < len(lv), lv[np.minimum(i, len(lv) - 1)], np.nan)
        below[a:b] = np.where(j >= 0, lv[np.maximum(j, 0)], np.nan)
    unknown = ~np.isfinite(p + g)
    above[unknown], below[unknown] = np.nan, np.nan
    return pd.DataFrame({"above": above, "below": below}, index=idx)


def sessions_to_event(index: pd.DatetimeIndex, events) -> pd.Series:
    """For every candle, how many sessions from its session to the next event day: 0 on the
    event day itself, 1 on the session before it, and so on; NaN when no event lies ahead. An
    event on a day without a session (a weekend, past the data) counts from the next session.

    Args:
        index: New York candle times.
        events: Event days (e.g. the reaction days of quarterly reports).
    """
    day = (index.tz_localize(None) if index.tz is not None else index).normalize()
    sessions = day.unique()
    at = np.unique(np.searchsorted(sessions, pd.DatetimeIndex(events).normalize()))
    pos = np.searchsorted(sessions, day)                       # each candle's session number
    j = np.searchsorted(at, pos)                                # the next event at or after it
    ahead = j < len(at)
    out = np.where(ahead, at[np.minimum(j, len(at) - 1)] - pos, np.nan) if len(at) else \
        np.full(len(day), np.nan)
    return pd.Series(out, index=index, dtype=float)


def explain_levels(days: pd.DataFrame, day, *, span: int = 5, lookback: int = 120,
                   tolerance: float = 0.5, min_touches: int = 2, mix: bool = True) -> dict:
    """Everything behind the major levels known at the open of the session ``day``, to draw how
    they are built: the turning points of the last ``lookback`` sessions (found on the candles
    BEFORE ``day`` only, so every one of them is confirmed), how they group, and which groups
    are major. The levels are exactly :func:`major_levels`' rows of that day.

    Returns:
        ``day``; ``known_until`` (the previous session); ``window_start`` (the oldest session a
        turning point may come from); ``typical_move`` (a typical day's move at the previous
        close); ``tolerance`` (the grouping distance, in price); ``turning_points``
        (:func:`pivots`' columns plus ``touches``, the size of the point's group, ``level``, the
        price of its major level, NaN for a point that is not part of one, and ``nearest``, the
        price distance to the nearest other turning point it could group with, i.e. of the
        same kind unless ``mix``: within the grouping distance for a point in a group, beyond it
        for one that stands alone); ``levels``; ``mix``.
    """
    day = pd.Timestamp(day).normalize()
    t = int(days.index.searchsorted(day))
    if t < 1:
        raise ValueError(f"no session before {day.date()}")
    piv = pivots(days.iloc[:t], span)
    piv = piv[piv["position"] >= t - 1 - lookback].reset_index(drop=True)
    move = float(typical_move(days, 14, across_nights=True).iloc[t - 1])
    tol = tolerance * move
    piv["touches"], piv["level"], piv["nearest"] = 0, np.nan, np.nan
    rows = []
    if len(piv) > 1:
        p, k = piv["price"].to_numpy(float), piv["kind"].to_numpy()
        gaps = np.abs(p[:, None] - p[None, :]) + np.diag(np.full(len(p), np.inf))
        if not mix:                       # only a point of the same kind can be a neighbour
            gaps[k[:, None] != k[None, :]] = np.inf
        near = gaps.min(axis=1)
        piv["nearest"] = np.where(np.isfinite(near), near, np.nan)
    if len(piv) and np.isfinite(tol):
        price, kind = piv["price"].to_numpy(float), piv["kind"].to_numpy()
        when = piv["day"].to_numpy()
        for g in _find_groups(price, kind, np.arange(len(piv)), tol, mix=mix):
            piv.loc[g, "touches"] = len(g)
            if len(g) >= min_touches:
                row = _level_row(day, price[g], kind[g], when[g])
                piv.loc[g, "level"] = row["price"]
                rows.append(row)
    return {"day": day, "known_until": days.index[t - 1],
            "window_start": days.index[max(t - 1 - lookback, 0)], "typical_move": move,
            "tolerance": tol, "turning_points": piv,
            "levels": _levels_frame(rows), "mix": mix}


def explain_trend(days: pd.DataFrame, trend: pd.DataFrame, *, length: int, slope_days: int,
                  flat: float) -> pd.Series:
    """Why each day's trend is what it is, in words (for hovering over a chart).

    Args:
        days: Daily candles.
        trend: :func:`trend_state` of those candles.
        length: Days in the average.
        slope_days: Days its rise is measured over.
        flat: The flat zone, in typical daily moves.
    """
    close = days["close"].reindex(trend.index).to_numpy(float)
    avg, rise = trend["average"].to_numpy(float), trend["rise"].to_numpy(float)
    zone = flat * trend["typical_move"].to_numpy(float)
    out = []
    for c, a, r, z, s in zip(close, avg, rise, zone, trend["state"].to_numpy(float), strict=True):
        if not np.isfinite(s):
            out.append(f"not known yet: needs {length + slope_days} days of history")
        elif s == 1:
            out.append(f"UP: the close {c:.2f} is above the {length}-day average {a:.2f}, and the "
                       f"average rose {r:+.2f} over {slope_days} days, more than the flat zone "
                       f"(±{z:.2f})")
        elif s == -1:
            out.append(f"DOWN: the close {c:.2f} is below the {length}-day average {a:.2f}, and "
                       f"the average fell {r:+.2f} over {slope_days} days, more than the flat "
                       f"zone (±{z:.2f})")
        elif abs(r) <= z:
            out.append(f"SIDEWAYS: the {length}-day average moved only {r:+.2f} over "
                       f"{slope_days} days, inside the flat zone (±{z:.2f})")
        else:
            out.append(f"SIDEWAYS: the average {'rose' if r > 0 else 'fell'} {r:+.2f}, but the "
                       f"close {c:.2f} is {'below' if c < a else 'above'} it ({a:.2f})")
    return pd.Series(out, index=trend.index)


def change_since_start(close: pd.Series) -> pd.Series:
    """The percent change of a price series since its first value (to compare a stock with its
    sector and the market on one scale)."""
    first = close.dropna()
    return (close / first.iloc[0] - 1) * 100 if len(first) else close * np.nan
