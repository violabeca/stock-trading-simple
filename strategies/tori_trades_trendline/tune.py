"""Tune Tori Trades' trendline bounce on ten stocks at once, on 15-minute candles, with every
knob normalized so it means the same on every stock, and judge it on data it never saw.

The user (2026-10-20): fix the 15-minute chart; challenge the knobs (missing ones, merges, the
same meaning on every stock); be creative with metrics, derived from the strategy's goal and the
traps seen so far, and keep them all for post-processing; stocks GPI BIO ICUI AMG ESE R KALU CHE
CACC; optimize from 2026-01-01 to 2026-08-01 and validate on the later data. (A first version
with ASR, a later start and an earlier validation window was stopped before its first batch;
ASR is out entirely.)

Fixed before the first trial:

- **Windows.** TRAIN: 2026-01-01 to 2026-07-31, the nine stocks. VALID: 2026-08-01 to
  2026-09-28 (the last day all nine have 1-minute candles), about 40 sessions.
- **Fixed knobs**: the 15-minute chart; a plain price scale, 150 candles on screen and the session
  refresh (none mattered, or only by making the strategy trade less); a close-break distance of 0
  (her rule; it moved results by +-2 bp).
- **Tuned knobs** (:data:`SPACE`). Every distance is in typical candles (the median 15-minute
  high-to-low range of the stock as it was then), in R (the trade's own first risk), in
  sessions or in candles of the one chart, so a value means the same on a calm and on a jumpy
  stock: ``tol`` (the lines' close enough; the poke allowance is tied to it, 0.4 x ``tol``:
  two knobs merged), ``confirm``, ``band`` (the entry gap), ``room`` (the stop beyond the
  line), ``target`` (profit taking in R; 0 = her rule, none), ``reward`` (skip a signal with a
  line of the other kind less than ``reward`` R ahead: no room to win), ``skip_open`` (no
  signal on the session's first candles), ``trend`` and ``min_span``, and ``sides``.
- **Metrics** (:func:`metrics`), each answering a trap seen before: "do not trade" wins (score
  evidence: t and lower bound of the net return per trade; no trade scores worst); market drift
  passes for skill (every trade against random entries held as long, same side, stock and
  months: excess return, and wins above chance as a z, the user's win/loss idea made sound); a
  few trades decide (median, trimmed mean, best trade's share, return without each stock's best
  trade); win rate fights profit (payoff ratio, profit factor, expectancy in R); costs eat tight
  stops (costs in R, gross and net per trade); stocks and months differ (stocks positive, worst
  third, alpha, beta, book Sharpe, positive months).
- **Studies** (:data:`STUDIES`): random search (exploration; every trial also replayed on VALID,
  so each knob's importance is checked out of time); TPE on ``edge_t`` (excess over
  random holds, t), ``lcb`` (net return per trade, mean - 2 se), ``chance_z`` (wins above
  chance) and ``sharpe`` (the book), warm-started with the random trials; NSGA-II on
  ``edge_t`` against ``activity`` (trades per stock and week, capped at 3: "a few a week", none
  forced). Batches of 28 trials until two batches bring no gain (at least 3, at most 12).
- **Judgement**: the 5 best TRAIN trials of every study and her defaults are replayed on VALID;
  the three best on VALID also with every line moved one typical candle toward and away from
  price (the placebo). A setting **has an edge** only when, on VALID, its trades beat random
  holds (``edge_t`` >= 2), make money after costs (net bp per trade > 0) and beat the moved
  lines. VALID is about 40 sessions of nine stocks: only a sizeable edge can reach t >= 2 there.

**Version 2** (2026-10-21; the user approved, after v1 was read; v1's journal is
``studies-v1.log``). v1's search failed in two ways. (1) Settings with 5-9 lucky trades topped
every evidence objective and then set the bar that the warm start and the stopping rule
compared against, so the TPE studies stopped early. (2) The best active settings sat at the edge
of the space on five or more knobs. Changes:

- **A minimum sample in the scoring, not in the strategy:** a setting with fewer than
  :data:`MIN_TRADES` TRAIN trades keeps all its metrics but scores the worst value, so it cannot
  become "best" (a setting that does not trade is still allowed).
- **Wider bounds** where v1's best active settings sat at an edge: target 0.5 and 0.75 R, ``tol``
  to 2, ``confirm`` to 10, ``skip_open`` to 8, ``trend`` to 400, ``min_span`` to 80, ``room`` to
  5, ``band`` to 3. **New knob** ``entry``: ``next_open`` or ``limit`` (her entry at the line,
  ``bounce.replay``).
- **Objectives:** ``edge_t``, ``net_bp`` (net return per trade, with the minimum sample), ``lcb``
  and ``sharpe`` (TPE), and NSGA-II on ``edge_t`` against ``activity``. ``chance_z`` stays a
  metric only: v1 showed it follows the exit's geometry. A trade's length depends on its exit
  (a target ends a winner early, a stop a loser), so random holds of the same length are no fair
  baseline for its win rate. Excess return is fair: no exit rule makes money from random entries.
- **Budget:** 8 random batches; TPE and NSGA-II at least 6 and at most 16 batches.
- **One-shot confirmation** (:func:`confirm`) on EARLY = 2025 (2025-01-02 to 2025-12-31; AMG,
  BIO and KALU from their first 1-minute session in June 2025), never read by v1 or v2. Picks,
  fixed now: the best TRAIN trial of every v2 study, her defaults, and v1's three candidates
  (``tpe-chance_z#248``, ``nsga2-edge_t-activity#139``, ``tpe-lcb#195``). Each is replayed
  with her lines and with every line moved one typical candle toward and away from price.
  **It passes** only when, on EARLY: ``edge_t`` >= 2, ``net_bp`` > 0, and her lines' ``edge_bp``
  beats both moved lines'.

Usage::

    uv run python strategies/tori_trades_trendline/tune.py prepare
    uv run python strategies/tori_trades_trendline/tune.py run [--only random,tpe-edge_t]
    uv run python strategies/tori_trades_trendline/tune.py validate
    uv run python strategies/tori_trades_trendline/tune.py confirm   # once: reads 2025
    uv run python strategies/tori_trades_trendline/tune.py report

Everything goes to ``var/tori_trades_trendline/tune/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.distributions import CategoricalDistribution as Choice
from optuna.distributions import FloatDistribution as Range
from optuna.trial import TrialState

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "var" / "tori_trades_trendline" / "tune"
_spec = importlib.util.spec_from_file_location("tori_bounce_tune", HERE / "bounce.py")
bounce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bounce)
td = bounce.td

STOCKS = ["GPI", "BIO", "ICUI", "AMG", "ESE", "R", "KALU", "CHE", "CACC"]
#: name -> (first session traded, last day of data, stocks)
WINDOWS = {"train": ("2026-01-01", "2026-07-31", STOCKS),
           "valid": ("2026-08-01", "2026-09-28", STOCKS),
           "early": ("2025-01-02", "2025-12-31", STOCKS)}     # :func:`confirm` only, once
FIXED = {"tf": "15m", "log": False, "screen": 150, "refresh": True, "give": 0.0}
SPACE = {
    "sides": Choice(["long", "short", "both"]),
    "tol": Choice([0.1, 0.25, 0.5, 1.0, 1.5, 2.0]), "confirm": Choice([1, 2, 3, 5, 7, 10]),
    "band": Range(0.0, 3.0), "room": Range(0.1, 5.0, log=True),
    "target": Choice([0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]), "reward": Range(0.0, 5.0),
    "skip_open": Choice([0, 1, 2, 4, 6, 8]), "trend": Choice([0, 20, 50, 100, 200, 300, 400]),
    "min_span": Range(0.0, 80.0), "entry": Choice(["next_open", "limit"])}
DEFAULTS = {"sides": "both", "tol": 0.25, "confirm": 2, "band": 0.25, "room": 0.5,
            "target": 0.0, "reward": 0.0, "skip_open": 0, "trend": 0, "min_span": 0.0,
            "entry": "next_open"}
BATCH, IN_FLIGHT, MIN_BATCHES, MAX_BATCHES, RANDOM_BATCHES = 28, 6, 6, 16, 8
ACTIVITY_CAP = 3.0                       # trades per stock and week worth rewarding, at most
MIN_TRADES = 50                          # TRAIN trades (all stocks) a setting needs to be scored
#: v1's candidates for the confirmation: (study, trial) in ``studies-v1.log``
V1_CANDIDATES = [("tpe-chance_z", 248), ("nsga2-edge_t-activity", 139), ("tpe-lcb", 195)]


def _kwargs(params: dict) -> dict:
    """A trial's knobs as ``bounce.replay`` arguments (``tol`` sets touch and pierce)."""
    p = dict(params)
    tol = p.pop("tol")
    return {**FIXED, **p, "touch": tol, "pierce": 0.4 * tol}


# ---- worker side: one process per stock keeps its candles, charts and lines -------------------

_DATA: dict = {}
_BOOKS: dict = {}
_LINES: OrderedDict = OrderedDict()
_HOLDS: dict = {}
_COSTS = None


def _start_worker() -> None:
    """Each worker reads the cost model once."""
    global _COSTS
    from trading_simple.config import load_settings
    _COSTS = load_settings().costs


def _window(stock: str, window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A stock's candles up to the end of ``window``."""
    if (stock, window) not in _DATA:
        daily, minutes = pd.read_pickle(OUT / "data" / f"{stock}.pkl")
        end = pd.Timestamp(WINDOWS[window][1])
        _DATA[stock, window] = (daily[daily.index <= end],
                                minutes[minutes.index.tz_localize(None).normalize() <= end])
    return _DATA[stock, window]


def _book(stock: str, window: str) -> bounce.Chartbook:
    """The stock's charts down to 15 minutes over ``window``, built once."""
    if (stock, window) not in _BOOKS:
        daily, minutes = _window(stock, window)
        _BOOKS[stock, window] = bounce.Chartbook(daily, minutes, list(td.TIMEFRAMES)[:6],
                                                 td.clock(daily))
    return _BOOKS[stock, window]


def _lines(stock: str, window: str, params: dict) -> dict:
    """The lines drawn so far for this stock, window, ``tol`` and ``confirm``."""
    key = (stock, window, params["tol"], params["confirm"])
    if key in _LINES:
        _LINES.move_to_end(key)
    else:
        _LINES[key] = {}
        while len(_LINES) > 80:
            _LINES.popitem(last=False)
    return _LINES[key]


def _holds(stock: str, window: str, d: float, length: int, first: int) -> np.ndarray:
    """The gross returns of every random hold of ``length`` candles in ``window`` (in at a
    candle's open, out at the open ``length`` candles later; ``d`` +1 long, -1 short), sorted."""
    key = (stock, window, d, length)
    if key not in _HOLDS:
        o = _book(stock, window).full["15m"]["open"].to_numpy(float)[first:]
        length = min(length, len(o) - 1)
        _HOLDS[key] = np.sort(d * (o[length:] / o[:-length] - 1))
    return _HOLDS[key]


def replay_stock(params: dict, stock: str, window: str, shift: float = 0.0) -> dict:
    """One stock's replay of a trial in a window, boiled down for :func:`metrics`.

    Returns:
        ``trades`` (per trade: side, R, net and gross return, candles held, the mean gross
        return and the win chance of random holds as long on the same side, entry month),
        ``daily`` (the strategy's daily returns), and ``log`` / ``bh`` / ``beta`` / ``dd`` /
        ``inside`` (log(1 + return), the same for buying and holding, beta to the stock, max
        drawdown, share of candles in a trade), ``sessions``.
    """
    daily, minutes = _window(stock, window)
    book = _book(stock, window)
    # a stock whose 1-minute candles start inside the window starts the day after them
    start = max(pd.Timestamp(WINDOWS[window][0]),
                minutes.index[0].tz_localize(None).normalize() + pd.Timedelta(days=1))
    r = bounce.replay(daily, minutes, start=start, costs=_COSTS, book=book,
                      cache=_lines(stock, window, params), shift=shift,
                      **_kwargs({"entry": "next_open", **params}))
    t, s, eq = r["trades"], r["summary"], r["equity"]
    first = int(book.full["15m"].index.searchsorted(start))
    rows = []
    for row in t.itertuples(index=False):
        d = 1.0 if row.side == "long" else -1.0
        gross = d * (row.exit - row.entry) / row.entry
        cost = gross - row[t.columns.get_loc("return")]
        held = _holds(stock, window, d, max(int(row.exit_k - row.entry_k), 1), first)
        rows.append((d, row.R, row[t.columns.get_loc("return")], gross, row.candles,
                     float(held.mean()), float(1 - np.searchsorted(held, cost, "right")
                                               / len(held)),
                     row[t.columns.get_loc("entry time")].strftime("%Y-%m")))
    closes = eq.groupby(eq.index.normalize()).last()
    moves = closes[["strategy", "buy and hold"]].pct_change().dropna()
    var = float(moves["buy and hold"].var())
    return {"trades": rows, "daily": moves["strategy"],
            "log": float(np.log1p(s["strategy return"])),
            "bh": float(np.log1p(s["buy and hold return"])),
            "beta": float(moves["strategy"].cov(moves["buy and hold"]) / var) if var else 0.0,
            "dd": float(s["strategy max drawdown"]),
            "inside": float(t["candles"].sum() / len(eq)) if len(t) else 0.0,
            "sessions": int(closes.shape[0])}


# ---- metrics: every number a trap asked for ----------------------------------------------------

def metrics(per_stock: dict[str, dict]) -> dict:
    """Every metric of one trial in one window (see the module docstring)."""
    rows = [(*tr, stock) for stock, r in per_stock.items() for tr in r["trades"]]
    t = pd.DataFrame(rows, columns=["d", "R", "net", "gross", "candles", "rand_mean",
                                    "rand_win", "month", "stock"])
    stocks = pd.DataFrame({k: {f: v[f] for f in ("log", "bh", "beta", "dd", "inside",
                                                  "sessions")}
                           for k, v in per_stock.items()}).T.astype(float)
    weeks = float(stocks["sessions"].mean()) / 5.0
    n = len(t)
    out = {"trades": n, "activity": n / len(stocks) / weeks if weeks else 0.0,
           "long_share": float((t["d"] > 0).mean()) if n else np.nan,
           "time_in_market": float(stocks["inside"].mean()),
           "return": float(stocks["log"].mean()),
           "excess_bh": float((stocks["log"] - stocks["bh"]).mean()),
           "alpha": float((stocks["log"] - stocks["beta"] * stocks["bh"]).mean()),
           "beta": float(stocks["beta"].mean()), "median_stock": float(stocks["log"].median()),
           "worst_third": float(np.sort(stocks["log"])[:int(np.ceil(len(stocks) / 3))].mean()),
           "stocks_pos": float((stocks["log"] > 0).mean()),
           "max_dd": float(stocks["dd"].mean())}
    days = pd.concat([r["daily"].rename(s) for s, r in per_stock.items()], axis=1)
    book = days.fillna(0.0).mean(axis=1)
    sd = float(book.std())
    out["sharpe"] = float(book.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    curve = (1 + book).cumprod()
    out["book_dd"] = float((curve / curve.cummax() - 1).min()) if len(curve) else 0.0
    monthly = (1 + book).groupby(book.index.to_period("M")).prod() - 1
    out["months_pos"] = float((monthly > 0).mean()) if len(monthly) else np.nan
    if n:
        wins = t["net"] > 0
        se = float(t["net"].std(ddof=1) / np.sqrt(n)) if n > 1 else np.inf
        excess = t["gross"] - t["rand_mean"]
        ese = float(excess.std(ddof=1) / np.sqrt(n)) if n > 1 else np.inf
        p = t["rand_win"].clip(1e-6, 1 - 1e-6)
        loss = -t.loc[~wins, "net"].sum()
        best = t.groupby("stock")["net"].max()
        out |= {"win_rate": float(wins.mean()),
                "payoff": float(t.loc[wins, "net"].mean() / -t.loc[~wins, "net"].mean())
                if wins.any() and (~wins).any() else np.nan,
                "profit_factor": float(t.loc[wins, "net"].sum() / loss) if loss > 0 else np.inf,
                "mean_R": float(t["R"].mean()), "median_R": float(t["R"].median()),
                "net_bp": 1e4 * float(t["net"].mean()), "gross_bp": 1e4 * float(t["gross"].mean()),
                "median_bp": 1e4 * float(t["net"].median()),
                "trimmed_bp": 1e4 * float(pd.Series(np.sort(t["net"])[int(0.05 * n):
                                                                    n - int(0.05 * n)]).mean()),
                "cost_R": float(((t["gross"] - t["net"]) / (t["gross"] / t["R"]).abs()
                                 .replace(0, np.nan)).mean()),
                "t_net": float(t["net"].mean() / se) if se > 0 else 0.0,
                "lcb": 1e4 * float(t["net"].mean() - 2 * se) if n > 1 else -1000.0,
                "sign_z": float((2 * wins.sum() - n) / np.sqrt(n)),
                "edge_bp": 1e4 * float(excess.mean()),
                "edge_t": float(excess.mean() / ese) if n > 1 and ese > 0 else -10.0,
                "chance_z": float((wins.sum() - p.sum()) / np.sqrt((p * (1 - p)).sum())),
                "best_share": float(t["net"].max() / t["net"].clip(lower=0).sum())
                if (t["net"] > 0).any() else np.nan,
                "return_wo_best": float((stocks["log"] - np.log1p(best.reindex(stocks.index)
                                                                    .fillna(0).clip(lower=0)))
                                        .mean())}
    else:
        out |= {"lcb": -1000.0, "edge_t": -10.0, "chance_z": -10.0, "sharpe": -10.0}
    return out


#: objective -> (the number it reads, the gain a batch must beat to count as progress)
OBJECTIVES = {"edge_t": 0.1, "lcb": 1.0, "net_bp": 1.0, "sharpe": 0.05}
#: the score of a setting below :data:`MIN_TRADES` (the worst each metric gives)
WORST = {"edge_t": -10.0, "lcb": -1000.0, "net_bp": -1000.0, "sharpe": -10.0}
STUDIES = {
    "random": (lambda: optuna.samplers.RandomSampler(seed=0), ["edge_t"]),
    "tpe-edge_t": (lambda: _tpe(1), ["edge_t"]),
    "tpe-lcb": (lambda: _tpe(2), ["lcb"]),
    "tpe-net_bp": (lambda: _tpe(3), ["net_bp"]),
    "tpe-sharpe": (lambda: _tpe(4), ["sharpe"]),
    "nsga2-edge_t-activity": (lambda: optuna.samplers.NSGAIISampler(population_size=BATCH,
                                                                    seed=5),
                              ["edge_t", "activity"])}


def _tpe(seed: int) -> optuna.samplers.TPESampler:
    """TPE for trials evaluated in parallel."""
    return optuna.samplers.TPESampler(seed=seed, multivariate=True, constant_liar=True,
                                      n_startup_trials=BATCH)


def _values(m: dict, objectives: list[str]) -> list[float]:
    """A trial's scores on its objectives (activity capped: a few trades a week, none forced).
    Below :data:`MIN_TRADES` TRAIN trades every score but activity is the worst: too few trades
    to say anything, so such a setting can never become "best" (it may still not trade)."""
    small = m["trades"] < MIN_TRADES
    return [min(m["activity"], ACTIVITY_CAP) if o == "activity"
            else WORST[o] if small else m[o] for o in objectives]


# ---- driver side -------------------------------------------------------------------------------

def _log(text: str) -> None:
    """A progress line, to the screen and to ``progress.log``."""
    line = f"{time.strftime('%H:%M:%S')} {text}"
    print(line, flush=True)
    with open(OUT / "progress.log", "a") as f:
        f.write(line + "\n")


def _storage() -> optuna.storages.JournalStorage:
    """The Optuna journal of every study."""
    from optuna.storages.journal import JournalFileBackend
    return optuna.storages.JournalStorage(JournalFileBackend(str(OUT / "studies.log")))


class Farm:
    """One single-process executor per stock: a stock's charts and lines stay in one process."""

    def __init__(self):
        self.pools = {s: ProcessPoolExecutor(1, initializer=_start_worker) for s in STOCKS}

    def submit(self, params: dict, windows: list[str], shift: float = 0.0) -> dict:
        """Every (stock, window) replay of a setting: ``{(stock, window): future}``."""
        return {(s, w): self.pools[s].submit(replay_stock, params, s, w, shift)
                for w in windows for s in WINDOWS[w][2]}

    def close(self) -> None:
        for p in self.pools.values():
            p.shutdown()


def _gather(futures: dict) -> dict[str, dict]:
    """``{window: metrics}`` from finished replays."""
    windows = {w for _, w in futures}
    return {w: metrics({s: f.result() for (s, ww), f in futures.items() if ww == w})
            for w in windows}


def run_study(name: str, farm: Farm, storage) -> None:
    """Run one study in batches until it stops improving."""
    factory, objectives = STUDIES[name]
    study = optuna.create_study(study_name=name, storage=storage, sampler=factory(),
                                directions=["maximize"] * len(objectives), load_if_exists=True)
    if name != "random" and not study.trials and len(objectives) == 1:
        source = optuna.load_study(study_name="random", storage=storage)
        for t in source.trials:
            if "train" in t.user_attrs:
                study.add_trial(optuna.trial.create_trial(
                    params=t.params, distributions=SPACE, user_attrs={
                        "train": t.user_attrs["train"], "warm": True},
                    values=_values(t.user_attrs["train"], objectives)))
    windows = ["train", "valid"] if name == "random" else ["train"]
    best, stale, began = None, 0, time.time()
    for batch in range(1, (RANDOM_BATCHES if name == "random" else MAX_BATCHES) + 1):
        running, asked = {}, 0
        while asked < BATCH or running:
            while asked < BATCH and len(running) < IN_FLIGHT:
                trial = study.ask(SPACE)
                running[trial.number] = (trial, farm.submit(trial.params, windows))
                asked += 1
            done = [k for k, (_, fs) in running.items() if all(f.done() for f in fs.values())]
            if not done:
                wait([f for _, fs in running.values() for f in fs.values()],
                     return_when=FIRST_COMPLETED)
                continue
            for k in done:
                trial, fs = running.pop(k)
                result = _gather(fs)
                for window, m in result.items():
                    trial.set_user_attr(window, m)
                study.tell(trial, _values(result["train"], objectives))
        now = (study.best_value if len(objectives) == 1
               else max(t.values[0] for t in study.best_trials))
        gain = np.inf if best is None else now - best
        stale = 0 if gain > OBJECTIVES[objectives[0]] else stale + 1
        best = now if best is None else max(best, now)
        _log(f"{name}: batch {batch}, {len(study.trials)} trials, best TRAIN {now:.4f} "
             f"(gain {gain:+.4f}), {time.time() - began:.0f} s")
        if name != "random" and batch >= MIN_BATCHES and stale >= 2:
            _log(f"{name}: stopped, no gain for 2 batches")
            break


def _picks(storage, n: int = 5) -> dict[str, dict]:
    """Her defaults and the ``n`` best trials each study found itself."""
    picks = {"defaults": DEFAULTS}
    for name in STUDIES:
        try:
            study = optuna.load_study(study_name=name, storage=storage)
        except KeyError:
            continue
        found = (study.best_trials if len(study.directions) > 1 else
                 [t for t in study.trials if t.state == TrialState.COMPLETE])
        found = [t for t in found if not t.user_attrs.get("warm")]
        for t in sorted(found, key=lambda t: t.values[0], reverse=True)[:n]:
            if t.params not in picks.values():
                picks[f"{name}#{t.number}"] = t.params
    return picks


def validate(farm: Farm, storage) -> None:
    """Replay the picks on every window, then the three best on VALID with the lines moved one
    typical candle toward and away from price."""
    picks, began = _picks(storage), time.time()
    futures = {k: farm.submit(p, ["train", "valid"]) for k, p in picks.items()}
    rows = [{"setting": k, "shift": 0.0, **_gather(fs)} for k, fs in futures.items()]
    _log(f"validate: {len(rows)} settings in {time.time() - began:.0f} s")
    order = sorted(rows, key=lambda r: r["valid"].get("edge_t", -10), reverse=True)
    best = [r["setting"] for r in order if r["setting"] != "defaults"][:3]
    futures = {(k, s): farm.submit(picks[k], ["train", "valid"], s)
               for k in best for s in (-1.0, 1.0)}
    rows += [{"setting": k, "shift": s, **_gather(fs)} for (k, s), fs in futures.items()]
    pd.to_pickle({"picks": picks, "rows": rows}, OUT / "validation.pkl")
    _log(f"validate: placebo done, {time.time() - began:.0f} s")


def confirm(farm: Farm, storage) -> pd.DataFrame:
    """The one-shot check on EARLY (2025) of the picks fixed in the module docstring: her lines
    and every line moved one typical candle toward and away from price. Saves ``confirm.pkl``."""
    from optuna.storages.journal import JournalFileBackend

    picks, began = _picks(storage, n=1), time.time()
    old = optuna.storages.JournalStorage(JournalFileBackend(str(OUT / "studies-v1.log")))
    for name, number in V1_CANDIDATES:
        trial = optuna.load_study(study_name=name, storage=old).trials[number]
        picks[f"v1 {name}#{number}"] = {"entry": "next_open", **trial.params}
    futures = {(k, s): farm.submit(p, ["early"], s) for k, p in picks.items()
               for s in (0.0, -1.0, 1.0)}
    early = {key: _gather(fs)["early"] for key, fs in futures.items()}
    rows = []
    for k in picks:
        m, toward, away = early[k, 0.0], early[k, 1.0], early[k, -1.0]
        rows.append({"setting": k, **{c: m.get(c, np.nan) for c in (
                         "trades", "activity", "win_rate", "net_bp", "edge_bp", "edge_t",
                         "chance_z", "return", "stocks_pos")},
                     "edge_bp moved toward": toward.get("edge_bp", np.nan),
                     "edge_bp moved away": away.get("edge_bp", np.nan),
                     "passes": bool(m.get("edge_t", -10) >= 2 and m.get("net_bp", -1e9) > 0
                                    and m.get("edge_bp", -1e9) > max(
                                        toward.get("edge_bp", -1e9), away.get("edge_bp", -1e9)))})
    table = pd.DataFrame(rows).set_index("setting")
    pd.to_pickle({"picks": picks, "early": early, "table": table}, OUT / "confirm.pkl")
    _log(f"confirm: {len(picks)} settings on 2025, {time.time() - began:.0f} s; "
         f"passes: {table.index[table['passes']].tolist() or 'none'}")
    return table


KEY = ["trades", "activity", "win_rate", "payoff", "mean_R", "net_bp", "median_bp", "gross_bp",
       "cost_R", "lcb", "edge_bp", "edge_t", "chance_z", "return", "alpha", "stocks_pos",
       "sharpe", "best_share"]


def report(storage) -> str:
    """The tuning's tables, as text (also saved to ``report.txt``)."""
    from optuna.importance import FanovaImportanceEvaluator, get_param_importances

    out = []

    def show(title, frame):
        out.append(f"\n## {title}\n\n{frame.to_string()}\n")

    rows = []
    for name in STUDIES:
        try:
            s = optuna.load_study(study_name=name, storage=storage)
        except KeyError:
            continue
        own = [t for t in s.trials if not t.user_attrs.get("warm")]
        best = (s.best_value if len(s.directions) == 1
                else max(t.values[0] for t in s.best_trials))
        rows.append({"study": name, "objectives": ", ".join(STUDIES[name][1]),
                     "own trials": len(own), "best TRAIN": best})
    show("Studies", pd.DataFrame(rows).set_index("study").round(4))

    random = [t for t in optuna.load_study(study_name="random", storage=storage).trials
              if "valid" in t.user_attrs]
    params = pd.DataFrame([t.params for t in random])
    tr = pd.DataFrame([t.user_attrs["train"] for t in random])
    be = pd.DataFrame([t.user_attrs["valid"] for t in random])
    agree = pd.DataFrame({m: {"TRAIN vs VALID (rank corr over the random trials)":
                              tr[m].corr(be[m], method="spearman"),
                              "TRAIN median": tr[m].median(), "VALID median": be[m].median(),
                              "VALID best": be[m].max()} for m in KEY if m in tr}).T
    show(f"Random search ({len(random)} trials): does a metric on TRAIN predict it on VALID?",
         agree.round(3))
    show("Random search: how the metrics move together on TRAIN (rank correlation)",
         tr[[m for m in KEY if m in tr]].corr(method="spearman").round(2))
    for objective in ("edge_t", "net_bp", "chance_z", "sharpe", "return"):
        imp = {}
        for label, frame in (("TRAIN", tr), ("VALID", be)):
            s = optuna.create_study(direction="maximize")
            s.add_trials([optuna.trial.create_trial(
                params=t.params, distributions=SPACE, value=float(np.nan_to_num(v, nan=-10.0)))
                for t, v in zip(random, frame[objective], strict=True)])
            imp[label] = get_param_importances(s, evaluator=FanovaImportanceEvaluator(seed=0))
        show(f"fANOVA importance for {objective} (random trials)",
             pd.DataFrame(imp).sort_values("TRAIN", ascending=False).round(3))
        effects = []
        for knob, dist in SPACE.items():
            bins = (params[knob].astype(str) if isinstance(dist, Choice)
                    else pd.qcut(params[knob], 4, duplicates="drop").astype(str))
            g = pd.DataFrame({"knob": knob, "value": bins, "TRAIN": tr[objective],
                              "VALID": be[objective]})
            effects.append(g.groupby(["knob", "value"]).agg(
                trials=("TRAIN", "size"), TRAIN=("TRAIN", "median"),
                VALID=("VALID", "median")))
        show(f"Median {objective} by knob value (TRAIN and VALID, random trials)",
             pd.concat(effects).round(3))

    stored = pd.read_pickle(OUT / "validation.pkl")
    table = pd.DataFrame(stored["rows"])
    for window in ("train", "valid"):
        plain = table[table["shift"] == 0].set_index("setting")[window]
        frame = pd.DataFrame(list(plain), index=plain.index)
        show(f"Validation on {window.upper()}: the best TRAIN settings of every study",
             frame[[m for m in KEY if m in frame]].round(3))
    show("Validation: their knobs", pd.DataFrame(stored["picks"]).T)
    moved = table[table["setting"].isin(table.loc[table["shift"] != 0, "setting"])]
    rows = []
    for _, r in moved.iterrows():
        for window in ("train", "valid"):
            m = r[window]
            rows.append({"setting": r["setting"], "shift": r["shift"], "window": window,
                         **{k: m.get(k) for k in ("trades", "win_rate", "net_bp", "edge_bp",
                                                  "edge_t", "chance_z", "return")}})
    show("Placebo: the three best VALID settings, lines moved one typical candle toward (+1) "
         "and away from (-1) price",
         pd.DataFrame(rows).set_index(["setting", "window", "shift"]).sort_index().round(3))
    if (OUT / "confirm.pkl").exists():
        show("Confirmation on 2025 (one shot): passes = edge_t >= 2, net_bp > 0 and her lines' "
             "edge_bp above both moved lines'", pd.read_pickle(OUT / "confirm.pkl")["table"]
             .round(3))
    text = "".join(out)
    (OUT / "report.txt").write_text(text)
    return text


def _load_stock(stock: str) -> str:
    """Read one stock's candles from the database, up to the last window's end."""
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    daily, minutes = td.load(make_engine(load_settings()), stock)
    end = pd.Timestamp(WINDOWS["valid"][1])
    pd.to_pickle((daily[daily.index <= end],
                  minutes[minutes.index.tz_localize(None).normalize() <= end]),
                 OUT / "data" / f"{stock}.pkl")
    return stock


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=["prepare", "run", "validate", "confirm", "report"])
    parser.add_argument("--only", default=",".join(STUDIES))
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    if args.what == "prepare":
        with ProcessPoolExecutor(5) as pool:
            for stock in pool.map(_load_stock, STOCKS):
                _log(f"prepare: {stock}")
    elif args.what == "report":
        print(report(_storage()))
    else:
        farm = Farm()
        try:
            if args.what == "run":
                for name in args.only.split(","):
                    run_study(name, farm, _storage())
            elif args.what == "confirm":
                print(confirm(farm, _storage()).round(3).to_string())
            else:
                validate(farm, _storage())
        finally:
            farm.close()
