"""xs_si_mom_composite tuning: the helpers behind ``xs_si_mom_composite_tune_marimo.py``.

Loaded by path from that notebook; ``strategy.py`` never imports it, and it changes no rule.
``strategy.py`` is pinned by SHA-256 in two cross-strategies, so the knobs it does not expose are
added HERE: :func:`weights` rebuilds the book from the same components (informed_flow_ls_p1's
ranks, resid_momentum's momentum, the composite's ``side_score``, adaptive_hedge's
``_on_priced_opens``) with every knob of :data:`KNOBS`, and equals ``signals()`` exactly at the
shipped values (``tests/test_xs_si_mom_composite.py``); its additions are each side's own
slots and bands, the long/short split, beta sizing, momentum's windows, the universe, volatility
stop-losses on both sides and the account size. Books run on the engine (``backtest.run``:
portfolio mode, working orders, the retrim cap, borrow fees) with every order priced by
:func:`order_fees`. The engine's single fee fraction cannot make the account size matter, so a
bigger account would only ever look cheaper: here the $1 minimum commission falls with the
order's size while the market impact grows with it.

The search runs around version B (:class:`Search`): every knob alone (:func:`sweep_plan`), the
single changes that helped combined best-first (:func:`improve`), then small random combinations
of a few changes (:func:`pool_plan`). The checks a winner must pass before paper trading live
here too (:meth:`Search.walk_forward`, :meth:`Search.pbo`, :func:`luck`, :func:`neighbours`,
:func:`stress`, :func:`noise`, :func:`capacity`). They answer one question: would the tuned
setting have beaten the standard one (version B) WITHOUT knowing the future?
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from trading_simple import backtest as bt
from trading_simple import strategies

HERE = Path(__file__).resolve().parent


def _module(name: str, path: Path):
    """A research module loaded by path (no package import)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = _module("xs_si_mom_composite_explain_t", HERE / "explain.py")
lab = _module("xs_si_mom_composite_lab_t", HERE.parent / "proposals_lab" / "lab.py")
STRAT = strategies.load("xs_si_mom_composite")
_MOD = STRAT.signals.__globals__                           # the loaded strategy.py

#: First traded day: before mid-2020 the DB holds only ~500 equities, and momentum needs a year.
FIRST = "2022-01-01"
WARMUP = "2019-07-01"
#: The search sees only the bars up to TRAIN_END; the months after it are the reserved check.
TRAIN_END = "2025-12-31"
FINAL = "2026-01-01"
#: Every input the knobs can read (the profitability sleeve needs the four fund_* frames).
FEATURES = ["short_interest", "fund_shares", "fund_gp_ttm", "fund_assets", "fund_ni_ttm",
            "fund_equity"]
#: The borrow-fee model: what borrowing costs by crowding tier. Assumptions, never searched.
FIXED = {"si_mid": 0.1, "si_high": 0.2, "borrow_gc": 0.005, "borrow_mid": 0.03,
         "borrow_high": 0.1}
#: Fewer trades a year than this and a setting is not eligible (the config's floor).
MIN_TRADES = 20
#: Sessions of daily returns behind the volatility stops and the impact cost.
VOL_WINDOW = 20
#: The per-order cost model (:func:`order_fees`), fixed assumptions like the borrow fees: the
#: full quoted spread by the stock's daily dollar volume (1m-bar medians, RESEARCH_LOG §13.4),
#: never below one cent, and the square-root impact ``IMPACT × daily volatility ×
#: sqrt(order ÷ daily dollar volume)``.
SPREAD_BY_VOLUME = [(175e6, 2.5), (55e6, 3.9), (18e6, 5.0), (0.0, 6.4)]
IMPACT = 1.0

SLEEVES = ["si+mom", "si", "mom", "si+mom+prof"]
NEVER = 100_000                    # max_hold without a limit (sessions)
#: (name, group, standard value = version B, the values tried, what it does). Every knob is
#: searched over its listed values, version B's included; ``0`` for ``retrim_k`` means never trim.
KNOBS = [
    ("top_n", "universe", 2000, [500, 750, 1000, 1500, 2000],
     "Stocks in the universe: the N most traded, re-formed every six months."),
    ("min_price", "universe", 5.0, [0.0, 1.0, 2.0, 5.0, 10.0],
     "Lowest share price that is ranked, bought or shorted (stored prices are split-adjusted)."),
    ("dtc_window", "short interest", 42, [10, 21, 42, 63, 84, 126],
     "Sessions days-to-cover is averaged over before it is ranked."),
    ("neutral_groups", "short interest", 5, [1, 3, 5, 7, 10],
     "Dollar-volume groups the short-interest rank is taken within (1 = one group)."),
    ("short_max_si", "short interest", 0.3, [0.15, 0.2, 0.3, 0.5, 1.0],
     "Short interest ÷ shares above which a stock is never shorted (a held short is covered)."),
    ("mom_lookback", "momentum", 252, [126, 189, 252],
     "Sessions of returns momentum is measured over (the stored history allows a year)."),
    ("mom_skip", "momentum", 21, [0, 5, 10, 21, 42],
     "Most recent sessions left out of momentum."),
    ("mom_beta_window", "momentum", 252, [63, 126, 252],
     "Sessions the market beta is estimated over, to take the market out of the returns."),
    ("neutral", "ranking", "", ["", "sector"],
     "'sector': short interest ranked within sector cells, momentum net of the sector too."),
    ("long_sleeves", "ranking", "si+mom", SLEEVES,
     "Signals in the long ranking: si = short interest, mom = momentum, prof = profitability."),
    ("short_sleeves", "ranking", "si+mom", SLEEVES, "Signals in the short ranking."),
    ("w_mom", "ranking", 1.0, [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
     "Weight of momentum against short interest (1 = equal)."),
    ("w_prof", "ranking", 1.0, [0.5, 1.0, 2.0],
     "Weight of profitability (only when a side lists prof)."),
    ("combine", "ranking", "rank", ["rank", "book"],
     "rank: one combined ranking; book: one book per signal, netted (needs equal sides)."),
    ("short_gate", "ranking", "", ["", "prof"],
     "'prof': only unprofitable companies may be shorted."),
    ("score_smooth", "ranking", 5, [1, 3, 5, 10, 15, 21],
     "Sessions the combined score is averaged over before it is ranked."),
    ("long_entry", "entries & exits", 0.9, [0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 0.97],
     "Rank a stock must reach to be bought (1 = best)."),
    ("long_exit_gap", "entries & exits", 0.1, [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
     "A long is sold when its rank falls this far below the entry rank."),
    ("short_entry", "entries & exits", 0.1, [0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2],
     "Rank a stock must fall to before it is shorted."),
    ("short_exit_gap", "entries & exits", 0.1, [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
     "A short is covered when its rank rises this far above the entry rank."),
    ("max_hold", "entries & exits", 126, [42, 63, 126, 189, 252, NEVER],
     f"Sessions a position is held unless it still ranks in its entry zone ({NEVER} = no "
     "limit)."),
    ("long_stop", "stops", 0.0, [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
     "Long stop-loss in daily volatilities below the entry (0 = none). A stock that moves 2% a "
     "day is sold 10% down at 5."),
    ("short_vol_stop", "stops", 0.0, [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
     "Short stop-loss in daily volatilities above the entry (0 = none)."),
    ("stop_from", "stops", "entry", ["entry", "best close"],
     "Where the volatility stops are measured from: the entry close, or the best close since "
     "the entry (a trailing stop)."),
    ("short_stop", "stops", 1.0, [0.5, 0.75, 1.0, 1.5, 2.0, 1000.0],
     "Squeeze stop: a short that rises this much above its entry is covered (1.0 = +100%, "
     "1000 = never)."),
    ("long_slots", "book", 15, [5, 8, 10, 12, 15, 20, 25, 30, 40, 50], "Number of long positions."),
    ("short_slots", "book", 15, [5, 8, 10, 12, 15, 20, 25, 30, 40, 50],
     "Number of short positions."),
    ("gross", "book", 0.9, [0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
     "Longs plus shorts as a share of the account (above ~0.90 a full book runs out of cash)."),
    ("long_share", "book", 0.5, [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0],
     "Share of the gross on the long side (0.5 = as much short as long, 1.0 = long only)."),
    ("sizing", "book", "equal", ["equal", "beta"],
     "equal: every slot the same; beta: sized at entry for equal market risk per position."),
    ("max_size", "book", 1.5, [1.25, 1.5, 2.0, 3.0],
     "Largest beta scaling of a slot (beta sizing only)."),
    ("retrim_k", "book", 2.0, [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0, 0.0],
     "A long grown to k × its slot is trimmed back to its slot (0 = never)."),
    ("account", "account", 33_000, [10_000, 25_000, 33_000, 50_000, 75_000, 100_000],
     "Starting account in dollars (at most $100k). Bigger orders pay less commission per "
     "dollar (the $1 minimum) but move thin stocks more (market impact)."),
]
NAMES = [k[0] for k in KNOBS]
GRID = {k[0]: k[3] for k in KNOBS}
_SHORT = {"short_entry", "short_exit_gap", "short_slots", "short_stop", "short_vol_stop",
          "short_max_si", "short_sleeves", "short_gate"}


def standard() -> dict:
    """Version B: the shipped A_U0 rules on the top 2000 (RESEARCH_LOG §13.5)."""
    return {name: value for name, _, value, _, _ in KNOBS}


def knob_table() -> pd.DataFrame:
    """Every knob with its group, standard value, the values tried and its meaning."""
    return pd.DataFrame([{"knob": n, "group": g, "standard (B)": v,
                          "values tried": ", ".join(str(c) for c in s), "what it does": d}
                         for n, g, v, s, d in KNOBS])


def valid(k: dict) -> bool:
    """Settings the book can express: each side's signals carry weight, and two books side by
    side need the same signals on both sides."""
    w = {"si": 1.0, "mom": k["w_mom"], "prof": k["w_prof"]}
    sides = (k["long_sleeves"].split("+"), k["short_sleeves"].split("+"))
    return (all(sum(w[s] for s in side) > 0 for side in sides)
            and (k["combine"] == "rank" or sides[0] == sides[1]))


def active(k: dict) -> set[str]:
    """The knobs that change the book in setting ``k`` (the others are ignored by it): the
    profitability weight needs a prof sleeve, the beta cap beta sizing, ``stop_from`` a stop,
    momentum's knobs a mom sleeve, and the short side's knobs a short side."""
    sleeves = set(k["long_sleeves"].split("+")) | set(k["short_sleeves"].split("+"))
    off = set()
    if "prof" not in sleeves:
        off.add("w_prof")
    if "mom" not in sleeves:
        off |= {"w_mom", "mom_lookback", "mom_skip", "mom_beta_window"}
    if k["sizing"] != "beta":
        off.add("max_size")
    if not (k["long_stop"] or k["short_vol_stop"]):
        off.add("stop_from")
    if k["long_share"] >= 1.0:
        off |= _SHORT
    return set(NAMES) - off


def changes(k: dict, base: dict) -> dict:
    """The knobs where ``k`` differs from ``base``."""
    return {n: k[n] for n in NAMES if k[n] != base[n]}


def label(k: dict, base: dict) -> str:
    """A setting's changes from ``base`` as text ("version B" when none)."""
    diff = changes(k, base)
    return ", ".join(f"{n} = {v}" for n, v in diff.items()) or "version B"


def usable(k: dict, base: dict) -> bool:
    """A valid setting whose every change from ``base`` actually changes the book."""
    diff = changes(k, base)
    return bool(diff) and valid(k) and set(diff) <= active(k)


def sweep_plan(base: dict, include: list[str]) -> list[dict]:
    """Stage 1: every value of every included knob, one knob at a time, the rest at ``base``."""
    out = []
    for name in include:
        for value in GRID[name]:
            k = {**base, name: value}
            if usable(k, base):
                out.append(k)
    return out


def pool_plan(base: dict, include: list[str], seed: int = 1):
    """Stage 3: endless small random combinations around ``base``, the same sequence every run
    (drawn without looking at any result): 2 to 5 included knobs, each to another of its values,
    redrawn until the setting is usable."""
    rng = np.random.default_rng(seed)
    while True:
        names = rng.choice(include, size=min(len(include), int(rng.integers(2, 6))),
                           replace=False)
        k = dict(base)
        for name in names:
            others = [v for v in GRID[name] if v != base[name]]
            k[name] = others[int(rng.integers(len(others)))]
        if usable(k, base):
            yield k


def improve(base: dict, singles: list[dict], fit, run, max_tries: int = 30) -> tuple[dict, list]:
    """Stage 2: start at ``base`` and add the single changes that beat it, best first, keeping
    each one only if the combined setting scores higher still.

    Args:
        base: Version B.
        singles: Stage-1 settings (one change each).
        fit: ``knobs -> score`` on the half-years the choice may read (``None``: not eligible).
        run: ``knobs -> None``: makes sure the setting has been backtested.
        max_tries: Most single changes tried.

    Returns:
        ``(final knobs, path)``: the path lists every change tried and whether it was kept.
    """
    current, best = dict(base), fit(base)
    scored = [(fit(k), k) for k in singles]
    ranked = sorted([(f, k) for f, k in scored if f is not None and f > best],
                    key=lambda x: -x[0])
    path = []
    for _, single in ranked[:max_tries]:
        (name, value), = changes(single, base).items()
        if current[name] != base[name]:
            continue                                   # this knob was already changed
        trial = {**current, name: value}
        if not usable(trial, base):
            continue
        run(trial)
        f = fit(trial)
        kept = f is not None and f > best
        path.append({"change": f"{name} = {value}", "score": f, "kept": kept})
        if kept:
            current, best = trial, f
    return current, path


# --- data --------------------------------------------------------------------------------------

def load(engine, settings, *, refresh: bool = False, cache: Path | None = None) -> dict:
    """The data every setting reads, cached as a pickle (the first load takes ~15 minutes).

    One panel holds every name the widest universe (top 2000, no price screen) ever held from
    :data:`FIRST` on, from :data:`WARMUP`, with :data:`FEATURES` and the sector codes. The
    formations' ranking tables let any ``top_n`` / ``min_price`` be formed without the DB.

    Args:
        engine: Database engine.
        settings: Loaded settings.
        refresh: Reload from the database.
        cache: Folder of the pickle (default ``var/xs_si_mom_composite/tune``).

    Returns:
        ``{"panel", "ranked": {formation: table}, "spy"}``.
    """
    cache = cache or HERE.parents[1] / "var" / "xs_si_mom_composite" / "tune"

    def build():
        dates = pd.date_range(FIRST, pd.Timestamp.today(), freq="6MS")
        ranked = {d: ex.ranked(engine, d) for d in dates}
        widest = {**ex.ROLL_B, "min_price": 0.0}
        names = sorted(set().union(*(ex.pick(r, widest) for r in ranked.values())))
        panel = ex.load(engine, settings, {dates[0]: names}, warmup=WARMUP, feats=FEATURES)
        sec = pd.read_sql(text("SELECT symbol, sector FROM securities"), engine).set_index(
            "symbol")["sector"].reindex(panel.close.columns)
        codes = {s: i for i, s in enumerate(sorted(sec.dropna().unique()))}
        panel.features["sector"] = pd.DataFrame(
            np.tile(sec.map(codes).to_numpy(dtype=float), (len(panel.close), 1)),
            index=panel.close.index, columns=panel.close.columns)
        del panel.features["member"]
        spy = ex.spy_returns(engine, settings, panel.close.index)
        return {"panel": panel, "ranked": ranked, "spy": spy}

    data = ex.cached(cache / f"data-{FIRST}-{WARMUP}.pkl", build, refresh=refresh)
    return {**data, "panel": ex.positive(data["panel"])}       # also for a cache built before


def universe(data: dict, top_n: int = 2000, min_price: float = 5.0) -> dict:
    """The point-in-time universe ``{formation: members}`` for ``top_n`` and ``min_price``."""
    roll = {**ex.ROLL_B, "top_n": int(top_n), "min_price": float(min_price)}
    return {d: ex.pick(r, roll) for d, r in data["ranked"].items()}


_PANELS: dict = {}


def panel_for(data: dict, top_n: int, min_price: float):
    """The data's panel with that universe's ``member`` mask (cached per universe)."""
    key = (id(data), int(top_n), float(min_price))
    if key not in _PANELS:
        p = data["panel"]
        member = ex.member_mask(p.close.index, p.close.columns,
                                universe(data, top_n, min_price))
        _PANELS[key] = replace(p, features={**p.features, "member": member})
    return _PANELS[key]


# --- the strategy with every knob -------------------------------------------------------------

def _select(score, close, shortable, *, long_entry: float, long_exit: float, short_entry: float,
            short_exit: float, long_slots: int, short_slots: int, max_hold: int,
            short_stop: float, long_stop: float = 0.0, short_vol_stop: float = 0.0,
            trailing: bool = False, sigma=None, sides: str = "both"):
    """informed_flow_ls_p1's ``_select`` with each side's own band and slot count, plus a
    volatility stop-loss on either side.

    At ``short_entry = 1 - long_entry``, ``short_exit = 1 - long_exit``, equal slots and no
    volatility stops it is p1's code path line for line (pinned by the tests).

    The volatility stop (``long_stop`` / ``short_vol_stop`` > 0): a long is sold when its close
    falls ``k`` daily volatilities (``sigma`` at the entry bar) below its entry close, a short
    is covered when its close rises ``k`` above it; ``trailing`` measures from the best close
    since entry instead. Like every decision here it is read at the close and traded at the
    next open. A stopped name is not re-entered on that side until its rank has left the entry
    zone once (as p1's squeeze stop).

    Returns:
        ``(longs, shorts, exits)``: held frames at the deciding bar and one
        ``(side, symbol, opened, closed, cause)`` per closed position.
    """
    s = score.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    vol = np.full(s.shape, np.nan) if sigma is None else np.asarray(sigma, dtype=float)
    ok_short = shortable.reindex_like(score).fillna(False).to_numpy(dtype=bool)
    n, m = s.shape
    held = {+1: np.zeros((n, m), bool), -1: np.zeros((n, m), bool)}
    book = {+1: {}, -1: {}}             # column -> [opened, clock, entry, best close, sigma]
    slots = {+1: long_slots, -1: short_slots}
    k_stop = {+1: long_stop, -1: short_vol_stop}
    stopped = {+1: set(), -1: set()}     # stopped out: no re-entry until out of the entry zone
    exits = []
    use = {"both": (+1, -1), "long": (+1,), "short": (-1,)}[sides]
    eps = 1e-9
    for t in range(n):
        row = s[t]
        for side in use:
            enter = row >= long_entry - eps if side > 0 else row <= short_entry + eps
            for j in list(book[side]):
                opened, clock, entry, best, sig = book[side][j]
                if enter[j]:
                    book[side][j][1] = t
                if np.isfinite(c[t, j]):
                    best = book[side][j][3] = (max(best, c[t, j]) if side > 0
                                               else min(best, c[t, j]))
                ref = best if trailing else entry
                cause = None
                if np.isnan(row[j]):
                    cause = "ineligible"
                elif (row[j] < long_exit - eps) if side > 0 else (row[j] > short_exit + eps):
                    cause = "band"
                elif k_stop[side] > 0 and np.isfinite(sig) and np.isfinite(c[t, j]) and (
                        c[t, j] <= ref * (1 - k_stop[side] * sig) if side > 0
                        else c[t, j] >= ref * (1 + k_stop[side] * sig)):
                    cause = "stop"
                elif t - book[side][j][1] >= max_hold:
                    cause = "max_hold"
                elif side < 0 and c[t, j] >= entry * (1 + short_stop):
                    cause = "short_stop"
                elif side < 0 and not ok_short[t, j]:
                    cause = "crowded"
                if cause:
                    exits.append((side, score.columns[j], score.index[opened], score.index[t],
                                  cause))
                    del book[side][j]
                    if cause in ("short_stop", "stop"):
                        stopped[side].add(j)
            stopped[side] -= {j for j in stopped[side] if not enter[j]}
            free = slots[side] - len(book[side])
            if free > 0:
                cand = enter & ~np.isnan(row) & ~np.isin(np.arange(m), list(book[side]))
                cand &= ~np.isin(np.arange(m), list(book[-side]))
                cand &= ~np.isin(np.arange(m), list(stopped[side]))
                if side < 0:
                    cand &= ok_short[t]
                idx = np.flatnonzero(cand)
                order = idx[np.argsort(-row[idx] if side > 0 else row[idx], kind="stable")]
                for j in order[:free]:
                    book[side][j] = [t, t, c[t, j], c[t, j], vol[t, j]]
            held[side][t, list(book[side])] = True
    frame = lambda a: pd.DataFrame(a, index=score.index, columns=score.columns)  # noqa: E731
    return frame(held[+1]), frame(held[-1]), exits


def _sizes(longs, shorts, beta, *, gross: float, long_share: float, long_slots: int,
           short_slots: int, sizing: str, max_size: float) -> pd.DataFrame:
    """informed_flow_ls_p1's ``_weights`` with each side's own slot count (equal or beta)."""
    size_l = gross * long_share / long_slots
    size_s = gross * (1 - long_share) / short_slots
    if sizing == "equal":
        return (bt.next_bar(longs).astype(float) * size_l
                - bt.next_bar(shorts).astype(float) * size_s)
    b = beta.reindex_like(longs).to_numpy(dtype=float)
    median = pd.DataFrame(np.where(b > 0, b, np.nan)).median(axis=1).to_numpy()[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = median / b
    ratio = np.clip(np.where(np.isfinite(ratio), ratio, 1.0), 1.0 / max_size, max_size)
    ratio[b <= 0] = 1.0 / max_size
    held_l, held_s = longs.to_numpy(dtype=bool), shorts.to_numpy(dtype=bool)
    out = np.zeros(held_l.shape)
    prev = np.zeros(held_l.shape[1])
    for t in range(len(out)):
        side = held_l[t].astype(float) - held_s[t].astype(float)
        keep = (side != 0) & (np.sign(prev) == side)
        size = np.where(keep, prev, 0.0)
        new = np.where((side != 0) & ~keep, side * ratio[t] * np.where(side > 0, size_l, size_s),
                       0.0)
        need, room = np.abs(new).sum(), gross - np.abs(size).sum()
        if need > 0:
            size = size + new * min(1.0, max(room, 0.0) / need)
        out[t] = prev = size
    return pd.DataFrame(out, index=longs.index, columns=longs.columns).shift(1, fill_value=0.0)


_SIG: dict = {}


def weights(data: dict, knobs: dict | None = None, panel=None) -> dict:
    """The book's target weights for one setting (missing knobs = version B; the last few
    settings are kept in memory, so a cost stress does not rebuild the signals).

    Args:
        data: :func:`load` output.
        knobs: Any of :data:`NAMES`.
        panel: A panel to use instead of the data's (the missing-names check).

    Returns:
        ``{"weights", "borrow", "retrim_k", "exits"}``.
    """
    k = {**standard(), **(knobs or {})}
    p = panel if panel is not None else panel_for(data, k["top_n"], k["min_price"])
    key = (id(data), id(p), tuple(sorted((n, repr(v)) for n, v in k.items())))
    if key not in _SIG:
        if len(_SIG) >= 6:
            _SIG.pop(next(iter(_SIG)))
        _SIG[key] = _weights(p, k)
    return _SIG[key]


def _weights(p, k: dict) -> dict:
    """:func:`weights` without the cache."""
    rm, ah = _MOD["component"]("rm"), _MOD["component"]("ah")
    lo, sh = k["long_sleeves"].split("+"), k["short_sleeves"].split("+")
    need = set(lo) | set(sh) | ({"prof"} if k["short_gate"] else set())
    r = _MOD["ranks"](p, need - {"mom"}, dtc_window=k["dtc_window"],
                      neutral_groups=k["neutral_groups"], short_max_si=k["short_max_si"],
                      min_price=k["min_price"], neutral=k["neutral"], **FIXED)
    base = r["base"]
    if "mom" in need:
        parts = rm.compose(p, min_price=k["min_price"], lookback=k["mom_lookback"],
                           skip=k["mom_skip"], beta_window=k["mom_beta_window"],
                           factors="market+sector" if k["neutral"] == "sector" else "market")
        month_end = parts["rank"].loc[parts["rebalance"].to_numpy()]
        r["ranks"]["mom"] = month_end.reindex(p.close.index, method="ffill").where(
            base["eligible"])
    shortable = base["shortable"]
    if k["short_gate"] == "prof":
        shortable = shortable & r["ranks"]["prof"].le(_MOD["GATE_MAX"])
    w = {"si": 1.0, "mom": k["w_mom"], "prof": k["w_prof"]}
    rules = {"long_entry": k["long_entry"],
             "long_exit": round(k["long_entry"] - k["long_exit_gap"], 10),
             "short_entry": k["short_entry"],
             "short_exit": round(k["short_entry"] + k["short_exit_gap"], 10),
             "long_slots": int(k["long_slots"]), "short_slots": int(k["short_slots"]),
             "max_hold": int(k["max_hold"]), "short_stop": float(k["short_stop"]),
             "long_stop": float(k["long_stop"]), "short_vol_stop": float(k["short_vol_stop"]),
             "trailing": k["stop_from"] == "best close"}
    if rules["long_stop"] > 0 or rules["short_vol_stop"] > 0:     # daily volatility, known at
        rules["sigma"] = p.close.pct_change(fill_method=None).rolling(   # the deciding close
            VOL_WINDOW, min_periods=VOL_WINDOW // 2).std().to_numpy()
    size = {key: k[key] for key in ("gross", "long_share", "sizing", "max_size")}
    exits = []

    def book(lo_k, sh_k):
        ls = _MOD["side_score"](r["ranks"], lo_k, w, base["eligible"], int(k["score_smooth"]))
        ss = ls if sh_k == lo_k else _MOD["side_score"](r["ranks"], sh_k, w, base["eligible"],
                                                       int(k["score_smooth"]))
        if sh_k == lo_k:
            held_l, held_s, ex_ = _select(ls, p.close, shortable, **rules)
        else:
            held_l, _, ex_l = _select(ls, p.close, shortable, sides="long", **rules)
            _, held_s, ex_s = _select(ss.mask(held_l), p.close, shortable, sides="short",
                                      **rules)
            ex_ = ex_l + ex_s
        exits.extend(ex_)
        return _sizes(held_l, held_s, base["beta"], long_slots=rules["long_slots"],
                      short_slots=rules["short_slots"], **size)

    if k["combine"] == "book":
        total = sum(w[s] for s in lo)
        target = sum(w[s] / total * book([s], [s]) for s in lo)
    else:
        target = book(lo, sh)
    target = ah._on_priced_opens(target, p.open_)
    return {"weights": target, "borrow": base["borrow"],
            "retrim_k": None if not k["retrim_k"] else float(k["retrim_k"]), "exits": exits}


def check(data: dict) -> float:
    """Largest |difference| between :func:`weights` at version B and the shipped ``signals()``
    on the same panel (0 = this module trades exactly the strategy)."""
    k = standard()
    p = panel_for(data, k["top_n"], k["min_price"])
    mine = weights(data, k)["weights"]
    shipped = STRAT.signals(p, **STRAT.params)["weights"]
    return float((mine - shipped).abs().to_numpy().max())


# --- one book on the engine -------------------------------------------------------------------

def order_size(knobs: dict) -> float:
    """The average order in dollars: the account's gross spread over the slots that trade."""
    k = {**standard(), **knobs}
    slots = k["long_slots"] + (k["short_slots"] if k["long_share"] < 1 else 0)
    return k["account"] * k["gross"] / slots


def order_fees(panel, window, weights: pd.DataFrame, account: float, costs,
               mult: float = 1.0) -> pd.DataFrame:
    """Each order's cost as a fraction of its value (vectorbt's ``fees``), per stock and bar.

    The engine's single fee fraction cannot express the account size, so the costs are priced
    per order here (the per-order model of RESEARCH_LOG §13.5, plus market impact):
    - commission at the order's real size: ``commission_per_share`` a share, at least
      ``min_commission``;
    - half the quoted spread: by the stock's daily dollar volume (:data:`SPREAD_BY_VOLUME`),
      never below the engine's ``spread_bps`` or one cent;
    - market impact, the square-root rule: :data:`IMPACT` × daily volatility ×
      sqrt(order ÷ daily dollar volume).
    An order is sized at ``account`` × the larger of the target and the previous target weight
    (a whole slot in or out); volume and volatility are read up to the bar before the fill.

    Args:
        panel: The panel the window was cut from (warm rolling volume and volatility).
        window: The scored panel.
        weights: The target weights.
        account: Starting cash in dollars.
        costs: The ``costs:`` settings block.
        mult: Every cost multiplied by this (the stress test).

    Returns:
        A fee frame on the window.
    """
    idx, cols = window.close.index, window.close.columns
    w = weights.reindex(index=idx, columns=cols).ffill().fillna(0.0).abs()
    size = account * np.maximum(w, w.shift(1).fillna(0.0)).to_numpy()
    price = window.open_.to_numpy(dtype=float)
    dv = (panel.close * panel.volume).rolling(20, min_periods=5).median().shift(1)
    vol = panel.close.pct_change(fill_method=None).rolling(
        VOL_WINDOW, min_periods=VOL_WINDOW // 2).std().shift(1)
    dv, vol = (f.reindex(index=idx, columns=cols).to_numpy() for f in (dv, vol))
    with np.errstate(divide="ignore", invalid="ignore"):
        commission = np.maximum(costs.min_commission,
                                size / price * costs.commission_per_share) / size
        tier = np.select([dv >= low for low, _ in SPREAD_BY_VOLUME],
                         [bp for _, bp in SPREAD_BY_VOLUME], SPREAD_BY_VOLUME[-1][1]) / 1e4
        spread = np.maximum(np.maximum(tier * mult, costs.spread_bps / 1e4), 0.01 / price)
        impact = mult * IMPACT * np.nan_to_num(vol, nan=0.02) * np.sqrt(size / dv)
    fee = commission + spread / 2.0 + np.nan_to_num(impact, nan=0.0)
    return pd.DataFrame(np.where(np.isfinite(fee) & (size > 0), fee, 0.0), index=idx,
                        columns=cols)


def run(data: dict, sig: dict, *, start, end=None, knobs: dict | None = None, settings,
        late_bps: float = 0.0, cost_mult: float = 1.0, whole_shares: bool = False,
        panel=None) -> dict:
    """One continuous book from ``start`` to ``end`` (signals warm from the panel's first bar),
    with ``knobs["account"]`` of starting cash and every order priced by :func:`order_fees`.

    Args:
        data: :func:`load` output.
        sig: :func:`weights` output (or a shuffled copy).
        start, end: The scored window.
        knobs: The setting (its ``account``).
        settings: Loaded settings (costs).
        late_bps: Extra cost per leg (every fill this much worse than the open).
        cost_mult: Every cost multiplied by this.
        whole_shares: Trade whole shares only.
        panel: The panel the signals were built on (default: the data's).

    Returns:
        ``{"returns", "pf", "window"}``: daily returns count day 0 from the cash.
    """
    capital = float({**standard(), **(knobs or {})}["account"])
    p = panel if panel is not None else data["panel"]
    keep = {key: sig[key] for key in ("weights", "borrow", "retrim_k")
            if sig.get(key) is not None}
    if end is not None:
        stop = pd.Timestamp(end, tz="UTC")
        rows = p.close.index <= stop
        p = replace(p, close=p.close[rows], open_=p.open_[rows], high=p.high[rows],
                    low=p.low[rows], volume=p.volume[rows],
                    features={n: f[rows] for n, f in p.features.items()})
        keep = {key: (v.loc[:stop] if isinstance(v, pd.DataFrame) else v)
                for key, v in keep.items()}
    window, s2 = bt._score_window(p, keep, start)
    s = lab.with_costs(settings, cost_mult, late_bps)
    fees = order_fees(p, window, s2["weights"], capital, s.costs, cost_mult)
    pf = bt.run(window, s2, fees=fees, slippage=s.costs.slippage_bps / 1e4, init_cash=capital,
                fill=STRAT.fill, whole_shares=whole_shares)
    value = bt.equity_curve(pf)["value"]
    return {"returns": value / value.shift(1).fillna(capital) - 1.0, "pf": pf,
            "window": window}


_BENCH: dict = {}


def benchmark(data: dict, index: pd.DatetimeIndex) -> pd.Series:
    """The common yardstick: equal-weight buy-and-hold of version B's universe, point in time
    (every setting is scored against it, so different universes stay comparable)."""
    key = (id(data), index[0], index[-1], len(index))
    if key not in _BENCH:
        p = data["panel"]
        rows = (p.close.index >= index[0]) & (p.close.index <= index[-1])
        window = replace(p, close=p.close[rows], open_=p.open_[rows], high=p.high[rows],
                         low=p.low[rows], volume=p.volume[rows], features={})
        _BENCH[key] = ex.member_benchmark(window, universe(data)).reindex(index).fillna(0.0)
    return _BENCH[key]


def halves(index: pd.DatetimeIndex, full: bool = False) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The calendar half-years (Jan-Jun, Jul-Dec) touching ``index`` as ``(start, next start)``;
    ``full``: only those with more than 20 of its sessions."""
    first = pd.Timestamp(index[0].year, 1 if index[0].month <= 6 else 7, 1, tz="UTC")
    out = []
    while first <= index[-1]:
        nxt = first + pd.DateOffset(months=6)
        if not full or ((index >= first) & (index < nxt)).sum() > 20:
            out.append((first, nxt))
        first = nxt
    return out


def score(r: pd.Series, b: pd.Series, how: str) -> float:
    """The optimizer's target on daily returns ``r`` against the yardstick ``b``.

    - ``alpha``: yearly return beyond what the book's market exposure explains;
    - ``robust``: the half-years' alphas, mean minus half their spread (steady beats lumpy);
    - ``alpha_ratio``: alpha per unit of the book's own stock-specific risk.
    """
    b = b.reindex(r.index).fillna(0.0)
    years = lab.years_of(r.index)
    if how == "robust":
        alphas = [lab.period_alpha(r[(r.index >= a) & (r.index < z)],
                                   b[(b.index >= a) & (b.index < z)])
                  for a, z in halves(r.index, full=True)]
        return lab.robust(alphas)
    alpha = float(lab.alpha_of(r.to_numpy(), b.to_numpy(), years)[3][0])
    if how == "alpha":
        return alpha
    beta = float(lab.alpha_of(r.to_numpy(), b.to_numpy(), years)[2][0])
    resid = (r - beta * b).std() * np.sqrt(252)
    return alpha / resid if resid > 0 else float("nan")


def metrics(data: dict, res: dict, sig: dict, how: str) -> dict:
    """Everything worth comparing about one book (the objective first).

    Returns:
        A flat dict: the score, alpha / CAGR / beta against the yardstick, alpha against SPY,
        volatility, Sharpe, max drawdown, worst month, the half-years' alphas, trades, stop-loss
        exits and orders a year, median days held, costs paid a year (commissions, spread,
        impact, borrow),
        average long / short exposure, and how large the orders are against the stocks' daily
        dollar volume (95th percentile).
    """
    r = res["returns"]
    b = benchmark(data, r.index)
    years = lab.years_of(r.index)
    cagr, bcagr, beta, alpha = (float(x[0]) for x in lab.alpha_of(r.to_numpy(), b.to_numpy(),
                                                                   years))
    spy = data["spy"].reindex(r.index).fillna(0.0)
    alpha_spy = float(lab.alpha_of(r.to_numpy(), spy.to_numpy(), years)[3][0])
    curve = (1 + r).cumprod()
    month = r.index.tz_localize(None).to_period("M")
    pf, window = res["pf"], res["window"]
    value = pf.value()
    assets = pf.asset_value(group_by=False)
    o = pf.orders.records_readable
    traded = o["Size"].abs() * o["Price"]
    dv = (window.close * window.volume).rolling(20, min_periods=5).median()
    col = {c: i for i, c in enumerate(window.close.columns)}
    adv = dv.to_numpy()[window.close.index.get_indexer(o["Timestamp"]),
                         [col[c] for c in o["Column"]]]
    inside = [(o_, c, why) for _, _, o_, c, why in sig["exits"]
              if r.index[0] <= c <= r.index[-1]]
    held = [(c - o_).days for o_, c, _ in inside]
    halves_ = {f"alpha {a.year}-H{1 if a.month == 1 else 2}": lab.period_alpha(
        r[(r.index >= a) & (r.index < z)], b[(b.index >= a) & (b.index < z)])
        for a, z in halves(r.index, full=True)}
    return {"score": score(r, b, how), "alpha": alpha, "cagr": cagr, "buy & hold": bcagr,
            "beta": beta, "alpha vs SPY": alpha_spy,
            "volatility": float(r.std() * np.sqrt(252)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
            "max drawdown": float((curve / curve.cummax() - 1).min()),
            "worst month": float(((1 + r).groupby(month).prod() - 1).min()),
            "trades / yr": len(inside) / years,
            "stop exits / yr": sum(why == "stop" for _, _, why in inside) / years,
            "orders / yr": len(o) / years,
            "median days held": float(np.median(held)) if held else float("nan"),
            "costs / yr": float(o["Fees"].sum() / value.mean() / years),
            "long exposure": float((assets.clip(lower=0).sum(axis=1) / value).mean()),
            "short exposure": float((-assets.clip(upper=0).sum(axis=1) / value).mean()),
            "order % of daily $ volume (p95)": float(np.nanpercentile(traded / adv, 95) * 100)
            if len(o) else float("nan"),
            **halves_}


def evaluate(data: dict, knobs: dict, *, settings, how: str, start=FIRST, end=TRAIN_END,
             **stress) -> tuple[dict, pd.Series]:
    """One setting from signals to metrics: ``(metrics, daily returns)``."""
    sig = weights(data, knobs)
    res = run(data, sig, start=start, end=end, knobs=knobs, settings=settings, **stress)
    return metrics(data, res, sig, how), res["returns"]


# --- the checks -------------------------------------------------------------------------------

class Search:
    """One search around version B, kept on disk and resumable.

    Every backtest is one trial of an Optuna journal study (``<folder>/<name>.log``; trials are
    enqueued with fixed knobs, so the sampler never steers anything) and its daily returns on
    the search window sit in ``<name>.returns.pkl``. A setting already run is never run again.
    The checks' results go to ``<name>.results.pkl``.

    Args:
        data: :func:`load` output.
        settings: Loaded settings.
        how: The score (:func:`score`).
        base: Version B's knobs (with the user's account).
        folder: Where the study lives.
        name: The study's name.
    """

    def __init__(self, data: dict, settings, how: str, base: dict, folder: Path, name: str):
        import optuna
        self.data, self.settings, self.how, self.base = data, settings, how, base
        folder.mkdir(parents=True, exist_ok=True)
        log = str(folder / f"{name}.log")
        storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(
            log, lock_obj=optuna.storages.journal.JournalFileOpenLock(log)))
        self.study = optuna.create_study(study_name=name, storage=storage, direction="maximize",
                                         load_if_exists=True,
                                         sampler=optuna.samplers.RandomSampler(seed=0))
        self.returns_file = folder / f"{name}.returns.pkl"
        self.results_file = folder / f"{name}.results.pkl"
        self.returns = (pd.read_pickle(self.returns_file) if self.returns_file.exists() else {})
        self.runs: dict[str, dict] = {}
        for t in self.study.trials:
            if t.state.name == "COMPLETE" and t.number in self.returns:
                k = {**base, **t.params}
                self.runs[label(k, base)] = {"trial": t.number, "knobs": k, "stage":
                                             t.user_attrs.get("stage", "?"),
                                             "metrics": dict(t.user_attrs)}
        self.seconds = [(t.datetime_complete - t.datetime_start).total_seconds()
                        for t in self.study.trials if t.state.name == "COMPLETE"]

    # -- running ---------------------------------------------------------------------------

    def run(self, knobs: dict, stage: str) -> dict | None:
        """The setting's record, backtesting it first if it has never run (``None``: failed)."""
        key = label(knobs, self.base)
        if key in self.runs:
            return self.runs[key]
        import optuna
        self.study.enqueue_trial({n: knobs[n] for n in NAMES}, user_attrs={"stage": stage})

        def objective(trial):
            k = {n: trial.suggest_categorical(n, GRID[n]) for n in NAMES}
            m, r = evaluate(self.data, k, settings=self.settings, how=self.how)
            for name, value in m.items():
                trial.set_user_attr(name, value)
            self.returns[trial.number] = r
            return m["score"]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study.optimize(objective, n_trials=1, catch=(Exception,))
        t = self.study.trials[-1]
        if t.state.name != "COMPLETE":
            return None
        pd.to_pickle(self.returns, self.returns_file)
        self.seconds.append((t.datetime_complete - t.datetime_start).total_seconds())
        self.runs[key] = {"trial": t.number, "knobs": dict(knobs), "stage": stage,
                          "metrics": dict(t.user_attrs)}
        return self.runs[key]

    # -- scoring ---------------------------------------------------------------------------

    def halves(self) -> list:
        """The search window's half-years (from version B's run)."""
        return halves(self.returns[self.runs["version B"]["trial"]].index, full=True)

    def bench(self) -> pd.Series:
        """The yardstick's daily returns on the search window."""
        return benchmark(self.data, self.returns[self.runs["version B"]["trial"]].index)

    def fit(self, knobs: dict, hs: list) -> float | None:
        """The setting's score on the half-years ``hs`` only (``None``: never run, failed, or
        trading fewer than :data:`MIN_TRADES` times a year)."""
        rec = self.runs.get(label(knobs, self.base))
        if rec is None or rec["metrics"].get("trades / yr", 0) < MIN_TRADES:
            return None
        r = self.returns[rec["trial"]]
        inside = np.zeros(len(r), bool)
        for a, z in hs:
            inside |= (r.index >= a) & (r.index < z)
        return score(r[inside], self.bench()[inside], self.how)

    def pick(self, hs: list, extra: list[dict]) -> dict:
        """The best setting on ``hs`` among version B, stage 1, stage 3 and ``extra`` (a greedy
        result built on ``hs``). Greedy settings built on other half-years are left out: they
        were chosen with data the pick may not read."""
        pool = [r["knobs"] for r in self.runs.values() if r["stage"] in ("B", "sweep", "pool")]
        scored = [(self.fit(k, hs), k) for k in pool + extra]
        return max([x for x in scored if x[0] is not None], key=lambda x: x[0])[1]

    def combine(self, hs: list) -> tuple[dict, list]:
        """Stage 2 on the half-years ``hs`` only: ``(combined knobs, path)``. The tuning rule
        is this followed by :meth:`pick` with the result."""
        singles = [r["knobs"] for r in self.runs.values() if r["stage"] == "sweep"]
        return improve(self.base, singles, lambda k: self.fit(k, hs),
                       lambda k: self.run(k, "greedy"))

    def walk_forward(self, picks: dict, min_train: int = 4) -> dict:
        """Would the tuning rule, run on the past only, have beaten version B next half-year?

        Args:
            picks: ``{half-year index: the rule's pick on the half-years before it}``.
            min_train: Half-years before the first one traded.

        Returns:
            ``{"table", "tuned", "standard", "delta", "P(delta > 0)"}``.
        """
        hs, b = self.halves(), self.bench()
        base_r = self.returns[self.runs["version B"]["trial"]]
        rows, tuned, std = [], [], []
        for i in range(min_train, len(hs)):
            k = picks[i]
            r = self.returns[self.runs[label(k, self.base)]["trial"]]
            test = (r.index >= hs[i][0]) & (r.index < hs[i][1])
            rows.append({"half-year": f"{hs[i][0].year}-H{1 if hs[i][0].month == 1 else 2}",
                         "picked on the past": label(k, self.base),
                         "alpha of the pick": lab.period_alpha(r[test], b[test]),
                         "alpha of version B": lab.period_alpha(base_r[test], b[test])})
            tuned.append(r[test])
            std.append(base_r[test])
        t, s_ = pd.concat(tuned), pd.concat(std)
        bb = b.reindex(t.index).fillna(0.0)
        years = lab.years_of(t.index)
        pr = lab.paired(t.to_numpy(), s_.to_numpy(), bb.to_numpy(), years)
        return {"table": pd.DataFrame(rows),
                "tuned": float(lab.alpha_of(t.to_numpy(), bb.to_numpy(), years)[3][0]),
                "standard": float(lab.alpha_of(s_.to_numpy(), bb.to_numpy(), years)[3][0]),
                "delta": pr["delta alpha"], "P(delta > 0)": pr["P(delta > 0)"]}

    def pbo(self) -> float:
        """Probability of backtest overfitting (Bailey et al.) over every eligible setting of
        stages 1 and 3 (both drawn without looking at results)."""
        hs, b = self.halves(), self.bench()
        rows = []
        for r in self.runs.values():
            ok = r["metrics"].get("trades / yr", 0) >= MIN_TRADES
            if r["stage"] in ("B", "sweep", "pool") and ok:
                x = self.returns[r["trial"]]
                rows.append([lab.period_alpha(x[(x.index >= a) & (x.index < z)],
                                              b[(b.index >= a) & (b.index < z)]) for a, z in hs])
        return lab.pbo(np.array(rows))

    def table(self) -> pd.DataFrame:
        """Every finished run: stage, changes from version B and its metrics."""
        return pd.DataFrame([{"trial": r["trial"], "stage": r["stage"], "changes vs B": key,
                              **r["metrics"]} for key, r in self.runs.items()])

    def overnight(self, include: list[str], hours: float, n_luck: int, progress=None,
                  min_train: int = 4) -> dict:
        """The whole night: stage 1 (each knob alone), stage 2 (combine the winners, also on the
        past only for every walk-forward half-year), stage 3 (random small combinations until
        the time left is what the checks need), the pick, and the checks. Resumable: a setting
        already run is read back.

        Args:
            include: The knobs searched.
            hours: The time budget.
            n_luck: Shuffled-name books of check C.
            progress: ``(iterable, title) -> iterable`` to show progress.
            min_train: Half-years before the first walk-forward one.

        Returns:
            The results (also saved to ``<name>.results.pkl``).
        """
        import time
        show = progress or (lambda it, title: it)
        start = time.time()
        self.run(self.base, "B")
        for k in show(sweep_plan(self.base, include), "stage 1 · each knob alone"):
            self.run(k, "sweep")
        hs = self.halves()
        greedy = {}                      # half-years read -> (combined knobs, path)
        for i in show(list(range(min_train, len(hs))) + [len(hs)], "stage 2 · combining"):
            greedy[i] = self.combine(hs[:i])
        per_run = float(np.median(self.seconds[-50:])) if self.seconds else 20.0
        reserve = per_run * (0.5 * n_luck + 2 * len(include) + 20)
        stop_at = start + hours * 3600 - reserve
        n_pool, plan = 0, pool_plan(self.base, include)
        for _ in show(range(max(0, int((stop_at - time.time()) / per_run))),
                      "stage 3 · small random combinations"):
            if time.time() > stop_at:
                break
            if self.run(next(plan), "pool") is not None:
                n_pool += 1
        picks = {i: self.pick(hs[:i], [greedy[i][0]]) for i in greedy}  # with the full pool
        best = picks[len(hs)]
        res = {"best": best, "path": greedy[len(hs)][1], "wf": self.walk_forward(picks, min_train),
               "pbo": self.pbo(), "n_pool": n_pool}
        st = self.settings
        res["luck"] = luck(self.data, best, settings=st, n=n_luck,
                           progress=lambda it: show(it, "check C · luck books"))
        near = []
        for name, x, k in show(neighbours(best, include), "check D · neighbours"):
            rec = self.run(k, "neighbour")
            if rec is not None:
                near.append({"knob": name, "pick's value": best[name], "neighbour": x,
                             "alpha": rec["metrics"]["alpha"], "score": rec["metrics"]["score"]})
        res["near"] = pd.DataFrame(near)
        mine = stress(self.data, best, settings=st, how=self.how)
        theirs = stress(self.data, self.base, settings=st, how=self.how)
        res["costs"] = pd.DataFrame({"test": mine["test"], "pick alpha": mine["alpha"],
                                     "B alpha": theirs["alpha"],
                                     "pick costs / yr": mine["costs / yr"],
                                     "B costs / yr": theirs["costs / yr"]})
        res["missing"] = [m["alpha"] for m in noise(self.data, best, settings=st, how=self.how)]
        res["capacity"] = capacity(self.data, best, settings=st, how=self.how)
        pd.to_pickle(res, self.results_file)
        return res

    def results(self) -> dict | None:
        """The last night's results, if any."""
        return pd.read_pickle(self.results_file) if self.results_file.exists() else None


def shuffle(sig: dict, data: dict, knobs: dict, rng) -> dict:
    """The same book on randomly relabelled names: each formation's members are permuted
    within dollar-volume tenths (ranked on the 126 sessions before it), so timing, turnover,
    holding times and liquidity match and only the names are random (the lab's luck books)."""
    k = {**standard(), **knobs}
    p = data["panel"]
    w = sig["weights"]
    out = w.copy()
    dv = p.close * p.volume
    hist = universe(data, k["top_n"], k["min_price"])
    starts = sorted(pd.Timestamp(f, tz="UTC") for f in hist)
    for i, t0 in enumerate(starts):
        rows = (w.index >= t0) & (w.index < (starts[i + 1] if i + 1 < len(starts)
                                             else w.index[-1] + pd.Timedelta(days=1)))
        names = [c for c in hist[t0.tz_localize(None)] if c in w.columns]
        med = dv.loc[:t0].iloc[-126:][names].median().fillna(0.0)
        tenth = pd.qcut(med.rank(method="first"), 10, labels=False)
        mapping = {}
        for g in range(10):
            group = list(med.index[tenth == g])
            mapping.update(dict(zip(group, rng.permutation(group), strict=True)))
        block = w.loc[rows].rename(columns=mapping)
        out.loc[rows] = block[w.columns].to_numpy()
    return {**sig, "weights": out}


def luck(data: dict, knobs: dict, *, settings, n: int = 50, seed: int = 7, end=TRAIN_END,
         progress=None) -> dict:
    """``n`` shuffled-name books of one setting against the real one (alpha, search window).

    Returns:
        ``{"real", "books", "percentile"}``: the share of shuffled books the real one beats.
    """
    sig = weights(data, knobs)
    real = run(data, sig, start=FIRST, end=end, knobs=knobs, settings=settings)
    b = benchmark(data, real["returns"].index)
    years = lab.years_of(b.index)
    alpha = lambda r: float(lab.alpha_of(r.to_numpy(), b.to_numpy(), years)[3][0])  # noqa: E731
    rng = np.random.default_rng(seed)
    books = []
    for _ in (progress(range(n)) if progress else range(n)):
        res = run(data, shuffle(sig, data, knobs, rng), start=FIRST, end=end, knobs=knobs,
                  settings=settings)
        books.append(alpha(res["returns"]))
    a = alpha(real["returns"])
    return {"real": a, "books": np.array(books),
            "percentile": float((np.array(books) < a).mean() * 100)}


def neighbours(knobs: dict, searched: list[str]) -> list[tuple[str, object, dict]]:
    """Every usable setting one listed value away from ``knobs`` on one searched knob:
    ``(knob, value, knobs)``."""
    out = []
    for name in searched:
        grid = GRID[name]
        if name not in active(knobs) or knobs[name] not in grid:
            continue
        i = grid.index(knobs[name])
        for j in (i - 1, i + 1):
            if 0 <= j < len(grid):
                k = {**knobs, name: grid[j]}
                if valid(k) and name in active(k):
                    out.append((name, grid[j], k))
    return out


def noise(data: dict, knobs: dict, *, settings, how: str, share: float = 0.05,
          copies: int = 3, seed: int = 11) -> list[dict]:
    """The setting on panels missing ``share`` of the names at random (does a few stocks carry
    it?)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(copies):
        k = {**standard(), **knobs}
        p = panel_for(data, k["top_n"], k["min_price"])
        cols = p.close.columns
        keep = sorted(set(cols) - set(rng.choice(cols, int(len(cols) * share), replace=False)))
        q = replace(p, close=p.close[keep], open_=p.open_[keep], high=p.high[keep],
                    low=p.low[keep], volume=p.volume[keep],
                    features={n: f[keep] for n, f in p.features.items()})
        sig = weights(data, knobs, panel=q)
        res = run(data, sig, start=FIRST, end=TRAIN_END, knobs=knobs, settings=settings,
                  panel=q)
        out.append(metrics(data, res, sig, how))
    return out


def stress(data: dict, knobs: dict, *, settings, how: str) -> pd.DataFrame:
    """The setting under harsher trading: fills 0.5% worse than the open, every cost doubled,
    and whole shares at its account size (search window)."""
    rows = []
    for label, kw in (("as tested", {}), ("fills 0.5% worse (+50 bp a leg)", {"late_bps": 50.0}),
                      ("every cost × 2", {"cost_mult": 2.0}),
                      ("whole shares", {"whole_shares": True})):
        m, _ = evaluate(data, knobs, settings=settings, how=how, **kw)
        rows.append({"test": label, "alpha": m["alpha"], "score": m["score"],
                     "costs / yr": m["costs / yr"]})
    return pd.DataFrame(rows)


def capacity(data: dict, knobs: dict, *, settings, how: str,
             capitals=tuple(GRID["account"])) -> pd.DataFrame:
    """The setting at several account sizes: the $1 minimum commission weighs less as orders
    grow, and the market impact grows with the orders against the stocks' daily volume."""
    rows = []
    for c in capitals:
        k = {**knobs, "account": c}
        m, _ = evaluate(data, k, settings=settings, how=how)
        rows.append({"account $": c, "average order $": round(order_size(k)),
                     "alpha": m["alpha"], "costs / yr": m["costs / yr"],
                     "order % of daily $ volume (p95)": m["order % of daily $ volume (p95)"]})
    return pd.DataFrame(rows)
