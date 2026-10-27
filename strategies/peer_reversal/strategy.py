"""peer_reversal — peer-relative short-term reversal that leans in when markets are stressed.

Buy the stocks that fell sharply against their own correlation peers over the past week, sell
short the ones that jumped against them, and hold for about a week. Book size depends on the
market: full size when it is stressed, half size otherwise.

* **Why it can pay.** Short-term reversal is the pay of liquidity providers: someone who must
  trade now moves the price, and whoever takes the other side earns the way back (Lehmann 1990;
  Lo & MacKinlay 1990). The pay rises when liquidity is scarce, i.e. when volatility is high
  (Nagel 2012). Measuring the move against peers removes the industry news that continues
  instead of reversing (Da, Liu & Schaumburg 2014; Hameed & Mian 2015).
* **Peers.** ``n_clusters`` groups from Ward hierarchical clustering of each name's trailing
  ``cluster_window``-bar market-residual daily returns (z-scored). They are re-formed every
  ``reform_every`` bars from past data only, on a schedule anchored at the panel's first bar.
  ``peers="sector"`` uses the panel's ``sector`` codes instead; ``peers="market"`` puts every
  name in one group (plain market-relative reversal).
* **Signal.** The ``formation``-bar return measured to the close ``skip`` bars before the
  decision bar, minus the mean return of the name's OTHER peers over the same bars. Skipping a
  day keeps bid-ask bounce and closing-print artifacts out. ``scale="vol"`` divides it by the
  name's own peer-relative daily volatility. It is ranked across the eligible names every bar.
* **Book.**
  - Entries: long the bottom ``q_in`` of the ranking, short the top ``q_in``, into ``slots``
    positions a side, most extreme first.
  - Exits: a long leaves when its rank rises above ``q_out``, a short when its rank falls below
    ``1 - q_out``; either way after ``max_hold`` bars. The clock restarts while the name is
    still in its entry zone.
  - Size: each slot is ``gross / 2 / slots`` of the book, dollar-neutral in its targets when
    the two sides hold the same number of slots (approximately otherwise; held positions also
    drift). ``gross`` is ``gross_stress`` when the market's ``vol_window``-bar realized
    volatility is above its trailing ``vol_median_window``-bar median, else ``gross_calm``.
    Shorts lock their proceeds. The repo engine's ``call_seq="auto"`` runs a bar's new
    shorts before the covers that free their cash, so at 0.98 gross some openings fill only
    partly (2% of I4's on T1, RESEARCH_LOG §13.6; 0.90 leaves the headroom). The lab v2
    runner runs covers first and under-fills none (R2.3).
  - The size is set at entry and frozen while held: the engine trades a name only when its
    target changes, so re-sizing would re-trade every holding.
* **Borrow.** Every short pays a yearly fee by crowding: short interest ÷ shares outstanding,
  in the tiers ``si_mid`` / ``si_high`` → ``borrow_gc`` / ``borrow_mid`` / ``borrow_high``.
  It is ``borrow_gc`` where there is no short-interest data. A name above ``short_max_si`` is
  never shorted.
* **Timing.** Decisions use bar t's close at the latest and fill at t+1's open (one-bar shift).
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd


def signals(panel, *, peers: str = "cluster", n_clusters: int = 30, cluster_window: int = 252,
            reform_every: int = 126, min_coverage: float = 0.8, min_peers: int = 2,
            formation: int = 5, skip: int = 1, scale: str = "none", scale_window: int = 63,
            q_in: float = 0.1, q_out: float = 0.5, slots: int = 20, max_hold: int = 10,
            gross_stress: float = 0.98,
            gross_calm: float = 0.49, vol_window: int = 21, vol_median_window: int = 252,
            volume_cap: float = 0.0, volume_base: int = 63, min_price: float = 5.0,
            short_max_si: float = 0.3, si_mid: float = 0.1, si_high: float = 0.2,
            borrow_gc: float = 0.005, borrow_mid: float = 0.03,
            borrow_high: float = 0.1) -> dict:
    """Target weights (long > 0, short < 0) and the yearly borrow rate of every name.

    Args:
        panel: Daily panel. ``features`` may carry ``short_interest`` (days-to-cover) and
            ``fund_shares`` for the borrow tiers, and ``sector`` (numeric codes) for
            ``peers="sector"``.
        peers: ``"cluster"`` (correlation clusters), ``"sector"`` (the ``sector`` feature) or
            ``"market"`` (every name one group).
        n_clusters: Maximum number of correlation clusters (``fcluster`` ``maxclust``).
        cluster_window: Bars of residual daily returns the clusters are formed on.
        reform_every: Bars between cluster re-formations (first one at bar
            ``cluster_window``).
        min_coverage: Share of valid returns a name needs in the window to be clustered.
        min_peers: Peers (other members with a valid return) a name needs for a signal.
        formation: Bars the return is measured over.
        skip: Bars between the end of the formation window and the decision bar.
        scale: ``"none"`` ranks the peer-relative move itself; ``"vol"`` divides it by the
            name's own volatility: the standard deviation of its daily peer-relative return
            over the ``scale_window`` bars ending with the formation window (Blitz, Huij,
            Lansdorp & Verbeek 2013), so a move unusual FOR THAT NAME ranks as extreme.
        scale_window: Bars of that volatility.
        q_in: Entry zone: long at rank <= ``q_in``, short at rank >= ``1 - q_in``.
        q_out: Exit band: a long leaves above ``q_out``, a short below ``1 - q_out``.
        slots: Positions per side.
        max_hold: Bars a position is held after it last ranked in its entry zone.
        gross_stress: Long plus short exposure for entries decided on a stressed bar.
        gross_calm: The same on a calm bar (0 = no new entries when calm).
        vol_window: Bars of the market's realized volatility.
        vol_median_window: Bars of the trailing median that volatility is compared with.
        volume_cap: When > 0, a name whose busiest formation bar traded more than this
            multiple of its median volume over the ``volume_base`` bars before the window is
            not entered (an information move, not price pressure). 0 = off.
        volume_base: Bars of the "normal volume" median for ``volume_cap``.
        min_price: Minimum close on the decision bar for a name to be ranked.
        short_max_si: Short interest / shares outstanding above which a name is not shorted.
        si_mid, si_high: Crowding tiers for the borrow rate.
        borrow_gc, borrow_mid, borrow_high: Yearly borrow rates below ``si_mid``, up to
            ``si_high``, and above it (``borrow_gc`` where there is no data).

    Returns:
        ``{"weights", "borrow"}`` for the engine's portfolio mode.
    """
    parts = compose(panel, peers=peers, n_clusters=n_clusters, cluster_window=cluster_window,
                    reform_every=reform_every, min_coverage=min_coverage, min_peers=min_peers,
                    formation=formation, skip=skip, scale=scale, scale_window=scale_window,
                    slots=slots, gross_stress=gross_stress,
                    gross_calm=gross_calm, vol_window=vol_window,
                    vol_median_window=vol_median_window, volume_cap=volume_cap,
                    volume_base=volume_base, min_price=min_price, short_max_si=short_max_si,
                    si_mid=si_mid, si_high=si_high, borrow_gc=borrow_gc,
                    borrow_mid=borrow_mid, borrow_high=borrow_high)
    longs, shorts, _ = _select(parts["rank"], parts["shortable"], parts["enter_ok"],
                               parts["slot_size"] > 0, q_in=q_in, q_out=q_out, slots=slots,
                               max_hold=max_hold)
    return {"weights": _weights(longs, shorts, parts["slot_size"]), "borrow": parts["borrow"]}


_DEFAULTS = {k: v.default for k, v in inspect.signature(signals).parameters.items()
             if v.default is not inspect.Parameter.empty}


def valid(params: dict) -> bool:
    """Reject parameter sets the book cannot express (missing keys read as ``signals()``'s
    defaults, so a partial dict is judged too)."""
    p = {**_DEFAULTS, **params}
    return (p["peers"] in ("cluster", "sector", "market") and p["n_clusters"] >= 2
            and p["cluster_window"] >= 20 and p["reform_every"] >= 1
            and 0.0 < p["min_coverage"] <= 1.0 and p["min_peers"] >= 1
            and p["formation"] >= 1 and p["skip"] >= 0
            and p["scale"] in ("none", "vol") and p["scale_window"] >= 5
            and 0.0 < p["q_in"] < p["q_out"] < 1.0 - p["q_in"]
            and p["slots"] >= 1 and p["max_hold"] >= 1
            and 0.0 < p["gross_stress"] <= 1.0 and 0.0 <= p["gross_calm"] <= 1.0
            and p["vol_window"] >= 2 and p["vol_median_window"] >= 2
            and p["volume_cap"] >= 0.0 and p["volume_base"] >= 2
            and 0.0 < p["si_mid"] < p["si_high"] <= p["short_max_si"])


def compose(panel, *, peers: str, n_clusters: int, cluster_window: int, reform_every: int,
            min_coverage: float, min_peers: int, formation: int, skip: int, slots: int,
            scale: str = "none", scale_window: int = 63,
            gross_stress: float, gross_calm: float, vol_window: int, vol_median_window: int,
            volume_cap: float, volume_base: int, min_price: float, short_max_si: float,
            si_mid: float, si_high: float, borrow_gc: float, borrow_mid: float,
            borrow_high: float) -> dict:
    """Everything the book reads, bar by bar (pure; also what ``premise.py`` inspects).

    Every frame sits on the DECISION bar: it uses closes up to that bar at the latest.

    Returns:
        ``labels`` (peer group per bar and name, -1 = none), ``move`` (the formation-window
        return), ``peer`` (the other peers' mean of it), ``signal`` (``move - peer``),
        ``eligible``, ``rank`` (percentile among eligible names, NaN elsewhere),
        ``enter_ok`` (the volume screen), ``market`` (equal-weight daily return), ``stress``
        (bool series), ``slot_size`` (per-bar size of a new slot), ``si_share``,
        ``shortable`` and ``borrow``.
    """
    close = panel.close
    idx, cols = close.index, close.columns
    ret = close.pct_change(fill_method=None)
    market = ret.mean(axis=1)
    labels = _labels(panel, ret, market, peers=peers, n_clusters=n_clusters,
                     cluster_window=cluster_window, reform_every=reform_every,
                     min_coverage=min_coverage)
    move = close.shift(skip) / close.shift(skip + formation) - 1.0
    peer = pd.DataFrame(_peer_mean(move.to_numpy(dtype=float), labels, min_peers),
                        index=idx, columns=cols)
    signal = move - peer
    if scale == "vol":                      # each name's own peer-relative daily volatility
        daily = ret - pd.DataFrame(_peer_mean(ret.to_numpy(dtype=float), labels, min_peers),
                                   index=idx, columns=cols)
        sigma = daily.shift(skip).rolling(scale_window, min_periods=scale_window // 2).std()
        signal = signal / sigma.where(sigma > 0)
    eligible = signal.notna() & (close >= min_price)
    rank = signal.where(eligible).rank(axis=1, pct=True)

    enter_ok = pd.DataFrame(True, index=idx, columns=cols)
    if volume_cap > 0:
        vol = panel.volume.reindex_like(close)
        busiest = vol.shift(skip).rolling(formation, min_periods=1).max()
        normal = vol.shift(skip + formation).rolling(volume_base,
                                                     min_periods=volume_base // 2).median()
        enter_ok = ~(busiest > volume_cap * normal)

    realized = market.rolling(vol_window, min_periods=vol_window).std()
    median = realized.rolling(vol_median_window, min_periods=vol_median_window // 2).median()
    stress = realized > median
    slot_size = pd.Series(np.where(stress, gross_stress, gross_calm) / 2.0 / slots, index=idx)

    feats = panel.features or {}
    dtc, shares = feats.get("short_interest"), feats.get("fund_shares")
    if dtc is None or shares is None:
        si_share = pd.DataFrame(np.nan, index=idx, columns=cols)
    else:
        adv = panel.volume.reindex_like(close).rolling(20, min_periods=10).median()
        shares = shares.reindex_like(close)
        si_share = dtc.reindex_like(close) * adv / shares.where(shares > 0)
    borrow = pd.DataFrame(np.where(si_share > si_high, borrow_high,
                                   np.where(si_share > si_mid, borrow_mid, borrow_gc)),
                          index=idx, columns=cols)
    shortable = ~(si_share > short_max_si)
    return {"labels": labels, "move": move, "peer": peer, "signal": signal,
            "eligible": eligible, "rank": rank, "enter_ok": enter_ok, "market": market,
            "stress": stress, "slot_size": slot_size, "si_share": si_share,
            "shortable": shortable, "borrow": borrow}


def _labels(panel, ret: pd.DataFrame, market: pd.Series, *, peers: str, n_clusters: int,
            cluster_window: int, reform_every: int, min_coverage: float) -> np.ndarray:
    """Peer-group label of every name on every decision bar (int array, -1 = no group).

    ``"cluster"``: the groups formed at bar b (from returns up to b's close) hold from b
    until the next formation; bars before the first formation have none.
    """
    n, m = ret.shape
    if peers == "market":
        return np.zeros((n, m), dtype=np.int32)
    if peers == "sector":
        sector = (panel.features or {}).get("sector")
        if sector is None:
            raise ValueError("peers='sector' needs the panel's 'sector' feature")
        s = sector.reindex_like(panel.close).to_numpy(dtype=float)
        return np.where(np.isfinite(s), s, -1).astype(np.int32)
    from scipy.cluster.hierarchy import fcluster, linkage

    out = np.full((n, m), -1, dtype=np.int32)
    r, mk = ret.to_numpy(dtype=float), market.to_numpy(dtype=float)
    starts = list(range(cluster_window, n, reform_every))
    for k, b in enumerate(starts):
        z = _residual_z(r[b - cluster_window + 1:b + 1], mk[b - cluster_window + 1:b + 1],
                        min_coverage)
        if z is None:
            continue
        keep, x = z
        end = starts[k + 1] if k + 1 < len(starts) else n
        groups = fcluster(linkage(x, method="ward"), t=n_clusters, criterion="maxclust")
        out[b:end, keep] = groups - 1
    return out


def _residual_z(r: np.ndarray, mk: np.ndarray, min_coverage: float):
    """The clustered names and their z-scored market-residual returns over one window.

    Returns ``(columns, matrix names x bars)`` (missing returns read as 0, the residual
    mean), or ``None`` when fewer than three names have the coverage.
    """
    valid = np.isfinite(r) & np.isfinite(mk)[:, None]
    keep = np.flatnonzero(valid.mean(axis=0) >= min_coverage)
    if len(keep) < 3:
        return None
    x, v = r[:, keep], valid[:, keep]
    m = np.where(np.isfinite(mk), mk, 0.0)[:, None]
    cnt = v.sum(axis=0)
    dx = np.where(v, x - np.where(v, x, 0.0).sum(axis=0) / cnt, 0.0)
    dm = np.where(v, m - np.where(v, m, 0.0).sum(axis=0) / cnt, 0.0)
    beta = (dx * dm).sum(axis=0) / np.maximum((dm * dm).sum(axis=0), 1e-18)
    e = np.where(v, x - beta * m, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = (e - np.nanmean(e, axis=0)) / np.nanstd(e, axis=0)
    return keep, np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0).T


def _peer_mean(move: np.ndarray, labels: np.ndarray, min_peers: int) -> np.ndarray:
    """Mean ``move`` of each name's OTHER group members with a valid value (leave-one-out);
    NaN where the name has no group or fewer than ``min_peers`` such peers."""
    n, m = move.shape
    out = np.full((n, m), np.nan)
    if n == 0:
        return out
    change = np.flatnonzero(np.any(labels[1:] != labels[:-1], axis=1)) + 1
    bounds = [0, *change.tolist(), n]
    for a, z in zip(bounds[:-1], bounds[1:], strict=True):
        lab = labels[a]
        grouped = lab >= 0
        if not grouped.any():
            continue
        member = np.zeros((m, int(lab.max()) + 1))
        member[np.flatnonzero(grouped), lab[grouped]] = 1.0
        s = move[a:z]
        v = np.isfinite(s) & grouped
        sv = np.where(v, s, 0.0)
        g = np.where(grouped, lab, 0)
        total = (sv @ member)[:, g] - sv
        count = (v.astype(float) @ member)[:, g] - v
        with np.errstate(invalid="ignore", divide="ignore"):
            out[a:z] = np.where(grouped & (count >= min_peers), total / count, np.nan)
    return out


def _select(rank: pd.DataFrame, shortable: pd.DataFrame, enter_ok: pd.DataFrame,
            can_open: pd.Series, *, q_in: float, q_out: float, slots: int, max_hold: int):
    """The long and the short book, bar by bar (decided at the close, traded next bar).

    Exits come first, so a slot freed today can be refilled today:
    - a NaN rank (ineligible);
    - a rank past the exit band (``q_out``);
    - ``max_hold`` bars since the name last ranked in its entry zone.

    Then each side fills its free slots with entry-zone candidates, most extreme first.
    Candidates must pass ``enter_ok``, and for a short also ``shortable``. Nothing opens on a
    bar where ``can_open`` is False.

    Returns:
        ``(longs, shorts, exits)``: boolean held frames on the deciding bar (before the
        one-bar shift) and a list of ``(side, symbol, opened, closed, cause)``.
    """
    s = rank.to_numpy(dtype=float)
    ok_short = shortable.reindex_like(rank).fillna(False).to_numpy(dtype=bool)
    ok_enter = enter_ok.reindex_like(rank).fillna(False).to_numpy(dtype=bool)
    can = can_open.reindex(rank.index).fillna(False).to_numpy(dtype=bool)
    n, m = s.shape
    held = {+1: np.zeros((n, m), bool), -1: np.zeros((n, m), bool)}
    book = {+1: {}, -1: {}}                         # column -> [opened, last bar in the zone]
    exits = []
    eps = 1e-9                  # ranks are exact fractions (0.1 vs 1 - 0.9)
    for t in range(n):
        row = s[t]
        for side in (+1, -1):
            zone = row <= q_in + eps if side > 0 else row >= 1.0 - q_in - eps
            for j in list(book[side]):
                if zone[j]:
                    book[side][j][1] = t                # a fresh extreme: the clock restarts
                cause = None
                if np.isnan(row[j]):
                    cause = "ineligible"
                elif (row[j] > q_out + eps) if side > 0 else (row[j] < 1.0 - q_out - eps):
                    cause = "band"
                elif t - book[side][j][1] >= max_hold:
                    cause = "max_hold"
                if cause:
                    exits.append((side, j, book[side][j][0], t, cause))
                    del book[side][j]
            free = slots - len(book[side])
            if free > 0 and can[t]:
                cand = zone & ok_enter[t] & ~np.isnan(row)
                cand[list(book[side]) + list(book[-side])] = False
                if side < 0:
                    cand &= ok_short[t]
                idx = np.flatnonzero(cand)
                order = idx[np.argsort(row[idx] if side > 0 else -row[idx], kind="stable")]
                for j in order[:free]:
                    book[side][j] = [t, t]
            held[side][t, list(book[side])] = True
    frame = lambda a: pd.DataFrame(a, index=rank.index, columns=rank.columns)  # noqa: E731
    exits = [(side, rank.columns[j], rank.index[a], rank.index[z], cause)
             for side, j, a, z, cause in exits]
    return frame(held[+1]), frame(held[-1]), exits


def _weights(longs: pd.DataFrame, shorts: pd.DataFrame, slot_size: pd.Series) -> pd.DataFrame:
    """Target weights at the fill bar from the held frames (decided at the close).

    A slot takes the deciding bar's ``slot_size`` when it opens and keeps it while held, so
    the engine never re-trades a holding when the stress state flips.
    """
    held_l, held_s = longs.to_numpy(dtype=bool), shorts.to_numpy(dtype=bool)
    size = slot_size.reindex(longs.index).fillna(0.0).to_numpy(dtype=float)
    out = np.zeros(held_l.shape)
    prev = np.zeros(held_l.shape[1])
    for t in range(len(out)):
        side = held_l[t].astype(float) - held_s[t].astype(float)
        keep = (side != 0) & (np.sign(prev) == side)
        out[t] = prev = np.where(keep, prev, side * size[t])
    return pd.DataFrame(out, index=longs.index, columns=longs.columns).shift(1, fill_value=0.0)
