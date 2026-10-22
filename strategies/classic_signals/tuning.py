"""The tuning of the classic_signals strategy: any hyperparameter of ``hyperparams.py`` can be
tuned (the rules' on/off switches too), one stock at a time.

* :class:`Lab`: one stock's strategy for any settings. Its parts (candles of each size, bands,
  averages, typical moves, trends, levels, contexts, targets) are built on demand by
  ``pipeline.py`` (the computation the main notebook runs) and the most recent ones are kept,
  so trying many settings mostly reuses them.
* :func:`score`: the daily Sharpe ratio of a stretch: its profit per session (an open trade
  valued at every session's last close, :func:`daily_pnl`), mean ÷ standard deviation × √252;
  ``-1e9`` below a minimum number of trades per session, so "never trade" cannot win. It does not
  grow with the stretch's length, so a training and a validation window compare directly.
* Several stocks at once: wherever one :class:`Lab` goes, a list of them can too. Every trade
  risks the same dollars at its stop (``simulate.costs``), so the stocks' daily profits are on
  one scale and add up; a setting is scored on their sum, each stock's own score beside it.
* :func:`luck`: the score the best of n settings without any edge reaches by chance alone. A
  tuning whose best does not clear it has found nothing.
* :func:`optimize`: Optuna on the sessions before a day (the training window), choosing whatever
  the search space leaves free; every trial can also be scored on the sessions from that day on
  (the validation window), recorded but never used to choose. Written to a journal file that
  optuna-dashboard reads; a run of the same name is continued with more trials.
* :func:`walk`: :func:`optimize` every ``refit`` sessions; the next ``refit`` sessions trade with
  its winner. Stitched together, an honest record: every decision used settings chosen on
  earlier sessions only, and a trade keeps the exit it opened with.
* :func:`effects`: one hyperparameter at a time across its range, everything else as in a given
  settings dict: how much each one moves the score, and which ones do nothing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pickle
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

HERE = Path(__file__).resolve().parent
JOURNAL = HERE / "studies" / "tuning.log"   # every optimization run: optuna-dashboard opens it


def _sibling(name: str):
    """One of this strategy's helper files, loaded by path."""
    spec = importlib.util.spec_from_file_location(f"classic_signals_tu_{name}",
                                                  HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IND, COMBINE, SIM = _sibling("indicators"), _sibling("combine"), _sibling("simulate")
DATA, PIPE, HP = _sibling("data"), _sibling("pipeline"), _sibling("hyperparams")
WARMUP = 120          # candles before a stretch that the rules read to warm up (shifts, rolls)
FLOOR = 100.0         # the smallest worst drawdown "profit ÷ worst drawdown" divides by, $
YEAR = 252            # sessions in a year: the Sharpe ratio is per year
KEEP = 24             # parts of each kind a Lab keeps
optuna.logging.set_verbosity(optuna.logging.WARNING)


def daily_pnl(trades: pd.DataFrame, candles: pd.DataFrame,
              notional: float = 10_000.0) -> pd.Series:
    """The profit of every session of ``candles`` (marked to market): a closed trade counts with
    its P&L after costs on the session it closes, an open one at its value at every session's last
    close. Indexed by session day. A trade holds its ``shares`` (``notional`` dollars' worth when
    the trades have no such column)."""
    close = candles["close"].to_numpy(float)
    if not len(close):
        return pd.Series(dtype=float)
    equity = np.zeros(len(close))
    if len(trades):
        a = candles.index.get_indexer(trades["entry time"])
        b = candles.index.get_indexer(trades["exit time"])
        side = np.where(trades["side"] == "long", 1.0, -1.0)
        price = trades["entry price"].to_numpy(float)
        shares = (trades["shares"].to_numpy(float) if "shares" in trades
                  else notional / price)
        for k in range(len(trades)):                 # one trade at a time: open from a to b - 1
            equity[a[k]:b[k]] = side[k] * shares[k] * (close[a[k]:b[k]] - price[k])
        closed = np.zeros(len(close))
        np.add.at(closed, b, trades["P&L $"].to_numpy(float))
        equity += np.cumsum(closed)
    day = candles.index.tz_localize(None).normalize()
    last = np.r_[np.flatnonzero(day[1:] != day[:-1]), len(close) - 1]
    return pd.Series(np.diff(equity[last], prepend=0.0), index=day[last])


def score(daily: pd.Series, trades: int, *, min_per_day: float) -> float:
    """The daily Sharpe ratio of a stretch: the mean profit per session ÷ its standard deviation,
    × √252 (a year of sessions). ``-1e9`` when the stretch has fewer than ``min_per_day`` trades
    a session, or none, or a profit that never varies.

    Args:
        daily: The profit of every session of the stretch (:func:`daily_pnl`).
        trades: The trades entered in it.
        min_per_day: Trades per session the stretch needs to score at all.
    """
    sd = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    if trades == 0 or trades < min_per_day * len(daily) or not sd > 0:
        return -1e9
    return float(daily.mean() / sd * np.sqrt(YEAR))


def luck(trials: int, sessions: int, *, chance: float = 0.5) -> float:
    """The score the best of ``trials`` settings without any edge reaches by chance on
    ``sessions`` sessions: typically (``chance`` 0.5), or one time in twenty (0.95).

    Without an edge, a setting's t-statistic (its score × √(sessions / 252)) is a standard
    normal, and the best of n independent ones stays below Φ⁻¹(chance^(1/n)) with that chance.
    Settings that resemble each other are not independent, so the true line is somewhat lower
    (178 random settings of NOW came out 0.1 below it).
    """
    from statistics import NormalDist

    if trials < 1 or sessions < 2:
        return float("nan")
    return NormalDist().inv_cdf(chance ** (1 / trials)) * float(np.sqrt(YEAR / sessions))


def _risk(daily: pd.Series) -> dict:
    """The worst drawdown of a stretch's profit curve (marked to market) and profit ÷ it."""
    curve = daily.cumsum().to_numpy(float)
    worst = float((curve - np.maximum(np.maximum.accumulate(curve), 0.0)).min()) if len(curve) \
        else 0.0
    return {"worst drawdown $": worst,
            "profit ÷ worst drawdown": float(curve[-1] / max(-worst, FLOOR)) if len(curve)
            else 0.0}


def _cut(x, rows: slice):
    """Every series and frame in ``x`` (a context, or one of its parts) cut to ``rows``."""
    if isinstance(x, dict):
        return {k: _cut(v, rows) for k, v in x.items()}
    return x.iloc[rows] if isinstance(x, pd.Series | pd.DataFrame) else x


class Lab:
    """One stock's strategy for any settings, its parts built once and kept.

    Args:
        names: ``{"stock", "sector", "market"}``.
        minutes: The stock's 1-minute candles, every stored hour (``data.load_minutes`` with
            ``extended=True``).
        days: Daily candles of the three symbols (``data.load_days``).
        reports: ``data.earnings_dates`` (empty: no report known).
        rules: ``combine.load_rules``.
        costs: ``simulate.run``'s cost arguments.
        keep: Parts of each kind kept (the oldest is dropped).
        options: The stock's daily option features (``pipeline.options``); None or empty: no
            option data (the option rules then allow both sides, and there are no walls).
    """

    def __init__(self, names: dict, minutes: pd.DataFrame, days: dict, reports: pd.DataFrame,
                 rules: dict, costs: dict, *, keep: int = KEEP,
                 options: pd.DataFrame | None = None):
        self.names, self.minutes, self.reports, self.rules = dict(names), minutes, reports, rules
        self.days = {s: days[s] for s in dict.fromkeys(names.values())}
        self.costs, self.keep, self._parts = dict(costs), keep, {}
        self.options = options

    def _part(self, kind: str, key, build):
        """A kept part, or a new one built (and kept)."""
        store = self._parts.setdefault(kind, OrderedDict())
        if key in store:
            store.move_to_end(key)
            return store[key]
        store[key] = build()
        if len(store) > self.keep:
            store.popitem(last=False)
        return store[key]

    def candles(self, size: int) -> pd.DataFrame:
        """The stock's ``size``-minute candles, every stored hour."""
        return self._part("candles", size, lambda: DATA.resample(self.minutes, size))

    def sessions(self, size: int = 1) -> tuple[np.ndarray, pd.DatetimeIndex]:
        """The first regular-session candle (a row of the context) of every session, and the
        session days (the same for every candle size)."""
        def build():
            c = self.candles(size)
            idx = c.index[c["regular"].to_numpy(bool)]
            day = idx.tz_localize(None).normalize()
            starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1]
            return starts, day[starts]
        return self._part("sessions", size, build)

    @staticmethod
    def _keys(s: dict) -> dict:
        """The settings each part depends on."""
        size, ext = s["candle_size"], bool(s["extended_bands"])
        return {"bands": (size, s["band_length"], s["band_width"], ext),
                "average": (size, s["average_length"], ext),
                "move": (size, s["move_size"], s["move_length"]),
                "trends": (s["trend_length"], s["trend_slope"], s["trend_flat"]),
                "levels": (s["level_span"], s["level_lookback"], s["level_tolerance"],
                           s["level_touches"], bool(s["level_mix"]))}

    def levels(self, s: dict) -> pd.DataFrame:
        """The stock's major levels for these settings."""
        key = self._keys(s)["levels"]
        return self._part("levels", key,
                          lambda: PIPE.levels(self.days[self.names["stock"]], *key))

    def prev(self) -> pd.DataFrame:
        """The previous session's high and low, as levels."""
        return self._part("prev", 0, lambda: IND.previous_range(self.days[self.names["stock"]]))

    def walls(self) -> pd.DataFrame | None:
        """The option walls, as levels (``pipeline.walls``); None without option data."""
        if self.options is None or not len(self.options):
            return None
        return self._part("walls", 0, lambda: PIPE.walls(self.options,
                                                         self.days[self.names["stock"]]))

    def context(self, s: dict) -> dict:
        """What the rules read with these settings (``pipeline.context``)."""
        k, size = self._keys(s), s["candle_size"]

        def build():
            c = self.candles(size)
            b = self._part("bands", k["bands"], lambda: PIPE.bands(c, *k["bands"][1:]))
            a = self._part("average", k["average"], lambda: PIPE.average(c, *k["average"][1:]))
            m = self._part("move", k["move"],
                           lambda: PIPE.move(self.minutes, c, *k["move"][1:])[0])
            t = self._part("trends", k["trends"], lambda: PIPE.trends(self.days, *k["trends"]))
            return PIPE.context(self.names, c, b, m, t, a, self.levels(s), self.prev(),
                                self.reports, self.days[self.names["stock"]], self.options)
        return self._part("context", tuple(k.values()), build)

    def targets(self, s: dict):
        """The profit targets with these settings (``pipeline.targets``), or None."""
        kind = s["target_kind"]
        if kind == "none":
            return None
        key = (*self._keys(s).values(), kind, s["target_days"] if kind == "daily moves" else None)
        return self._part("targets", key, lambda: PIPE.targets(
            self.context(s), self.levels(s), self.prev(), kind, s["target_days"], self.walls()))

    def decide(self, s: dict, a: int, b: int) -> pd.DataFrame:
        """``combine.decide`` for context rows ``a`` .. ``b - 1`` (the rules read ``WARMUP``
        candles before ``a``: only earlier ones, as always)."""
        w = max(a - WARMUP, 0)
        return COMBINE.decide(_cut(self.context(s), slice(w, b)), self.rules,
                              HP.switches(s, self.rules), s, only_on=True).iloc[a - w:]

    def rows(self, s: dict, first: int, last: int | None) -> tuple[int, int]:
        """The context rows of sessions ``first`` .. ``last - 1`` (None: to the end)."""
        starts, _ = self.sessions(s["candle_size"])
        n = len(self.context(s)["candles"])
        return int(starts[first]), n if last is None or last >= len(starts) else int(
            starts[last])

    def simulate(self, s: dict, first: int = 0, last: int | None = None) -> dict:
        """``simulate.run`` of sessions ``first`` .. ``last - 1`` traded with settings ``s`` (a
        trade still open at the stretch's end closes at its last close), and the ``candles`` it
        ran on."""
        a, b = self.rows(s, first, last)
        tgt = self.targets(s)
        ctx = _cut(self.context(s), slice(a, b))
        out = SIM.run(ctx, self.decide(s, a, b), **HP.stop_args(s),
                      targets=None if tgt is None else tgt.iloc[a:b], **self.costs)
        return {**out, "candles": ctx["candles"]}

    def run(self, s: dict, first: int = 0, last: int | None = None) -> pd.DataFrame:
        """The trades of sessions ``first`` .. ``last - 1`` traded with settings ``s``."""
        return self.simulate(s, first, last)["trades"]

    def daily(self, s: dict, first: int = 0,
              last: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
        """The trades of sessions ``first`` .. ``last - 1`` and the profit of every session
        (:func:`daily_pnl`)."""
        out = self.simulate(s, first, last)
        return out["trades"], daily_pnl(out["trades"], out["candles"])


def labs(engine, stocks: list, *, training: tuple, validation: tuple, sectors: dict, rules: dict,
         costs: dict, market: str = "SPY", daily_from: str = "2024-01-01") -> tuple[list, str]:
    """Every stock's :class:`Lab`, built as the tuning notebook builds them (a headless run that
    builds them here gives the same labs, so the notebook can continue its runs).

    Every stock's 1-minute candles (every stored hour), its daily candles and its sector fund's
    and the market's, its option chains (``pipeline.options``) and the report dates, from
    ``training[0]`` to the validation window's end: the last day EVERY stock has 1-minute candles
    (at most ``validation[1]``), so no stock scores fewer validation sessions than the others.
    With many stocks every lab keeps fewer parts (:data:`KEEP` ÷ the stocks, at least 4), or 11
    labs would hold several GB.

    Returns:
        The labs and the validation window's last day ("YYYY-MM-DD").
    """
    counts = DATA.coverage_counts(engine, stocks, validation[0], validation[1])
    ends = [counts[s][counts[s] > 0].index.max() for s in stocks if (counts[s] > 0).any()]
    end = min(ends).strftime("%Y-%m-%d") if ends else validation[1]
    minutes = DATA.load_minutes(engine, stocks, training[0], end, extended=True)
    symbols = list(dict.fromkeys([*stocks, *(sectors[s] for s in stocks), market]))
    days = DATA.load_days(engine, symbols, daily_from, end)
    reports = DATA.earnings_dates(list(sectors))
    keep = max(4, KEEP // len(stocks))
    return [Lab({"stock": s, "sector": sectors[s], "market": market}, minutes[s], days, reports,
                rules, costs, keep=keep, options=PIPE.options(s, training[0], end))
            for s in stocks], end


def _labs(labs) -> list:
    """One :class:`Lab` or several, as a list."""
    return list(labs) if isinstance(labs, list | tuple) else [labs]


def group(labs) -> str:
    """The name of a stock or a group of stocks ("NOW + GEV")."""
    return " + ".join(lab.names["stock"] for lab in _labs(labs))


def _day(lab: Lab, s: dict, day) -> int:
    """A session number of ``lab``: ``day`` itself, or, for a date, the first session on or
    after it (stocks with gaps in their calendars still split at the same date)."""
    if isinstance(day, int | np.integer):
        return int(day)
    _, days = lab.sessions(s["candle_size"])
    return int((days < pd.Timestamp(day)).sum())


def _on_grid(row: dict, value):
    """``value`` as the tuner may draw it: one of the choices, or moved onto the search grid."""
    if "choices" in row:
        return value if value in row["choices"] else None
    low, high, step = row["search"]
    v = min(max(low + round((float(value) - low) / step) * step, low), high)
    return int(round(v)) if all(float(x).is_integer() for x in (low, high, step)) else round(v, 9)


def suggest(trial: optuna.Trial, rows: dict, space: dict, start: dict) -> dict:
    """One trial's settings: every hyperparameter the space leaves free drawn by Optuna, every
    fixed one at its value (``start`` fills the rest). A lever is only drawn while it acts in
    this trial (``hyperparams.active``: its rule on, or the stop or target kind it belongs to):
    otherwise it would do nothing."""
    s = dict(start)
    for name, r in rows.items():                 # settings, then switches, then levers
        if not HP.active(r, s):
            continue
        spec = space[name]
        if isinstance(spec, list):
            s[name] = spec[0] if len(spec) == 1 else trial.suggest_categorical(name, spec)
        elif spec[0] == spec[1]:
            s[name] = spec[0]
        elif isinstance(spec[0], int):
            s[name] = trial.suggest_int(name, spec[0], spec[1], step=spec[2])
        else:          # Optuna's float steps carry noise (1.2000000000000002): back on the grid
            s[name] = round(trial.suggest_float(name, spec[0], spec[1], step=spec[2]), 9)
    return s


def journal(path) -> optuna.storages.JournalStorage:
    """An Optuna journal file (the repo's study storage, as in ``optimize.py``): every trial is
    written to it when it finishes, so ``uvx optuna-dashboard <path>`` shows a run live."""
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return optuna.storages.JournalStorage(
        JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path))))


def judge(labs, s: dict, day, *, min_per_day: float = 0.5, validate: bool = False) -> dict:
    """One setting on one stock or several (``labs``): the training sessions (before ``day``, a
    session number or a date) and, with ``validate``, the validation sessions (from ``day`` on,
    the indicators running straight through).

    Returns:
        The stocks together (their daily profits added up): ``score`` (:func:`score`; -1e9 when
        a stock trades too little), ``P&L $``, ``trades``, ``worst drawdown $``, ``profit ÷
        worst drawdown`` and ``worst stock`` (the lowest of the stocks' own scores); with
        ``validate``, ``validation score`` (None: no trade or no swing), ``validation P&L $``,
        ``validation trades`` and ``stocks profitable on validation``. Then every stock's own:
        ``<stock> score`` (None: too few trades), ``<stock> P&L $``, ``<stock> trades`` and,
        with ``validate``, its validation ones.
    """
    out, train, val, own, n_train, n_val = {}, [], [], [], 0, 0
    for lab in _labs(labs):
        k, d = lab.names["stock"], _day(lab, s, day)
        _, days = lab.sessions(s["candle_size"])
        cut = days[d] if d < len(days) else None          # the first validation day
        trades, daily = lab.daily(s, 0, None if validate else d)
        entered = trades["entry time"].dt.tz_localize(None).dt.normalize()
        n = len(trades) if cut is None else int((entered < cut).sum())
        part = daily if cut is None else daily[daily.index < cut]
        own.append(score(part, n, min_per_day=min_per_day))
        out.update({f"{k} score": own[-1] if own[-1] > -1e8 else None,
                    f"{k} P&L $": float(part.sum()), f"{k} trades": n})
        train.append(part)
        n_train += n
        if validate and cut is not None:
            v_part = daily[daily.index >= cut]
            v = score(v_part, len(trades) - n, min_per_day=0.0)
            out.update({f"{k} validation score": v if v > -1e8 else None,
                        f"{k} validation P&L $": float(v_part.sum()),
                        f"{k} validation trades": len(trades) - n})
            val.append(v_part)
            n_val += len(trades) - n
    ok = min(own) > -1e8
    together = pd.concat(train, axis=1).fillna(0.0).sum(axis=1).sort_index()
    head = {"score": score(together, n_train, min_per_day=0.0) if ok else -1e9,
            "P&L $": float(together.sum()), "trades": n_train, **_risk(together),
            "worst stock": min(own) if ok else None}
    if val:
        v_all = pd.concat(val, axis=1).fillna(0.0).sum(axis=1).sort_index()
        v = score(v_all, n_val, min_per_day=0.0)
        head.update({"validation score": v if v > -1e8 else None,
                     "validation P&L $": float(v_all.sum()), "validation trades": n_val,
                     "stocks profitable on validation": sum(
                         out[f"{lab.names['stock']} validation P&L $"] > 0
                         for lab in _labs(labs))})
    return {**head, **out}


def by_stock(record: dict, stocks: list) -> pd.DataFrame:
    """A :func:`judge` record (or a trial's row of :func:`optimize`) as one row per stock and
    one for the stocks together: score, profit and trades, training and validation."""
    keys = ("score", "P&L $", "trades", "validation score", "validation P&L $",
            "validation trades")
    clean = lambda v: None if isinstance(v, float) and v <= -1e8 else v  # noqa: E731
    rows = [{"stock": k, **{c: clean(record.get(f"{k} {c}")) for c in keys}} for k in stocks]
    rows.append({"stock": "together", **{c: clean(record.get(c)) for c in keys}})
    return pd.DataFrame(rows)


def sides(labs, s: dict) -> pd.DataFrame:
    """Every stock's profit from its longs and from its shorts, quarter by quarter (by the
    entry), on every loaded session: a setting that only leans with a trend makes its money on
    one side."""
    frames = []
    for lab in _labs(labs):
        t = lab.run(s)
        q = t["entry time"].dt.tz_localize(None).dt.to_period("Q").astype(str)
        f = t.assign(quarter=q).pivot_table(index="quarter", columns="side", values="P&L $",
                                            aggfunc="sum", fill_value=0.0)
        frames.append(f.reindex(columns=["long", "short"], fill_value=0.0).assign(
            stock=lab.names["stock"]).reset_index())
    return pd.concat(frames, ignore_index=True)[["stock", "quarter", "long", "short"]]


def optimize(labs, day, *, space: dict, start: dict, trials: int = 100,
             min_per_day: float = 0.5, seed: int = 0, warm: tuple = (), validate: bool = False,
             storage=None, name: str = "", parallel: bool = False) -> dict:
    """Optuna on the sessions before ``day`` (the training window): the daily Sharpe ratio
    (:func:`score`) of every trial, on one stock or on several stocks' daily profits added up
    (:func:`judge`), the best one kept.

    Args:
        labs: One stock's :class:`Lab`, or several (one setting for all of them).
        day: The first session after the training window (a session number, or a date).
        space: What the optimizer may choose (``hyperparams.clean_space``): a number's
            ``(from, to, step)``, a choice's allowed values; one value: fixed.
        start: A complete settings dict: the first trial (moved into the space).
        trials: Optuna trials (more, when a run is continued).
        min_per_day: Trades per session the training window needs to score at all. A trial with
            fewer is recorded as pruned (no score), which keeps it out of the best and of the
            dashboard's plots.
        seed: Seeds the sampler, so a run repeats exactly.
        warm: More settings to try first (e.g. the previous winner).
        validate: Also score every trial on the sessions from ``day`` on (the validation
            window, the indicators running straight through): its score, profit and trades are
            recorded beside the training ones, and never used to choose.
        storage: A journal file (:func:`journal`); None: kept in memory. A run named ``name``
            already in it is continued with ``trials`` more trials, if it was made with the same
            search space, minimum trades, windows, data and code (its stamp); otherwise
            ValueError: a different experiment needs a new name.
        name: The run's name in the journal.
        parallel: Other processes add trials to the same run at the same time
            (:func:`continue_run`): the sampler then counts their unfinished trials as bad
            ("constant liar"), so two workers do not search the same spot. Give each its own
            ``seed``.

    Returns:
        ``best``: the winning settings (complete); ``score``: its training score and ``best
        trial`` its number; ``trials``: one row per trial (its number, score: NaN when pruned,
        training profit, trades, worst drawdown, profit ÷ worst drawdown, the validation ones
        and every drawn value; ``luck``: the luck line after it); ``sessions``: the training
        sessions; ``luck``: the score the best of that many scored trials reaches by chance
        (:func:`luck`); ``start``: the first trial (in the space); ``name``: the run's name.
    """
    labs = _labs(labs)
    rules = labs[0].rules
    rows = HP.table(rules)
    base = HP.project(start, space, rules)
    free = [n for n in rows if HP.is_free(space[n])]
    _, days = labs[0].sessions(base["candle_size"])
    day_ = _day(labs[0], base, day)
    cut = days[day_] if day_ < len(days) else None        # the first validation day
    store = None if storage is None else journal(storage)
    stamp = _fingerprint(labs, {"space": space, "min_per_day": min_per_day, "day": day,
                                "validate": validate, "objective": "daily Sharpe"})

    def evaluate(trial: optuna.Trial) -> float:
        out = judge(labs, suggest(trial, rows, space, base), day, min_per_day=min_per_day,
                    validate=validate)
        value = out.pop("score")
        for k, v in out.items():
            trial.set_user_attr(k, v)
        if value <= -1e8:
            raise optuna.TrialPruned()                   # too few trades: no score
        return value

    before = 0
    if store is not None and name in optuna.get_all_study_names(store):
        old = optuna.load_study(study_name=name, storage=store)
        if old.user_attrs.get("stamp") != stamp:
            raise ValueError(f"the run '{name}' was made with other inputs (search space, minimum "
                             "trades, windows, data or code): give the new run another name")
        before = len(old.trials)
    # TPE draws its first trials at random before it models the scores; Optuna's default of 10
    # would make a short run a pure random search, so it scales with the trials here. A
    # continued run draws from a new seed: the same one would repeat the earlier draws
    sampler = optuna.samplers.TPESampler(seed=seed + before, constant_liar=parallel,
                                         n_startup_trials=max(3, min(10, (before + trials) // 3)))
    st = optuna.create_study(direction="maximize", sampler=sampler, storage=store,
                             study_name=name or None, load_if_exists=True)
    if not before:
        last = days[min(day_, len(days)) - 1]
        for k, v in {"stamp": stamp, "stock": group(labs),
                     "stocks": [lab.names["stock"] for lab in labs],
                     "objective": "daily Sharpe ratio (profit per session, marked to market)",
                     "training": [str(days[0].date()), str(last.date())],
                     "validation": [str(cut.date()), str(days[-1].date())]
                     if validate and cut is not None else None,
                     "minimum trades per session": min_per_day,
                     "space": {k: list(v) for k, v in space.items()}, "start": base}.items():
            st.set_user_attr(k, v)
        for x in [base, *warm]:
            x = HP.project(x, space, rules)
            st.enqueue_trial({k: x[k] for k in free}, skip_if_exists=True)
    st.optimize(evaluate, n_trials=trials)
    log = pd.DataFrame([{"trial": t.number, "score": t.value, **t.user_attrs, **t.params}
                        for t in st.trials])
    done = st.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    sessions = min(day_, len(days))
    log["luck"] = [luck(k, sessions) for k in log["score"].notna().cumsum()]   # after each trial
    if not done:                                         # nothing traded enough: the start
        return {"best": base, "score": -1e9, "best trial": None, "trials": log,
                "sessions": sessions, "luck": float("nan"), "start": base, "name": name}
    b = max(done, key=lambda t: t.value)
    return {"best": HP.project({**base, **b.params}, space, rules), "score": float(b.value),
            "best trial": b.number, "trials": log, "sessions": sessions,
            "luck": luck(len(done), sessions), "start": base, "name": name}


def continue_run(name: str, trials: int, *, worker: int = 0, storage=None,
                 training: tuple = ("2025-01-01", "2026-06-30"),
                 validation: tuple = ("2026-07-01", "2026-10-16"),
                 daily_from: str = "2024-01-01", market: str = "SPY") -> dict:
    """Add ``trials`` trials to a run of the journal from a terminal, as the tuning notebook
    would: its stocks, search space, first trial and minimum trades come from the run, the labs
    from :func:`labs` with the notebook's windows (a run made with other data or code is refused,
    as in :func:`optimize`). Several at once run in parallel, one per ``worker`` (its own seed):

        uv run python strategies/classic_signals/tuning.py continue "<run name>" 1000 0 &
        uv run python strategies/classic_signals/tuning.py continue "<run name>" 1000 1 &
    """
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    storage = JOURNAL if storage is None else storage
    rules = COMBINE.load_rules()
    cfg = run_settings(storage, name, rules)
    stocks = optuna.load_study(study_name=name, storage=journal(storage)).user_attrs["stocks"]
    settings = load_settings()
    labs_, _ = labs(make_engine(settings), stocks, training=training, validation=validation,
                    sectors=DATA.SECTOR_FUND, rules=rules, costs=SIM.costs(settings),
                    market=market, daily_from=daily_from)
    return optimize(labs_, validation[0], space=cfg["space"], start=cfg["start"],
                    trials=trials, min_per_day=cfg["min_per_day"], seed=1000 * (worker + 1),
                    validate=True, storage=storage, name=name, parallel=True)


def run_settings(storage, name: str, rules: dict) -> dict:
    """What a run in the journal was made with, to continue it: its search space (cleaned),
    first trial and minimum trades per session."""
    st = optuna.load_study(study_name=name, storage=journal(storage))
    a = st.user_attrs
    return {"space": HP.clean_space(a["space"], rules), "start": a["start"],
            "min_per_day": a["minimum trades per session"], "training": a.get("training"),
            "validation": a.get("validation"), "trials": len(st.trials)}


def runs(storage, stock: str | None = None) -> list[str]:
    """The runs in a journal (of one stock or group, :func:`group`), oldest first."""
    if not Path(storage).exists():
        return []
    store = journal(storage)
    names = optuna.get_all_study_names(store)
    if stock is None:
        return names
    return [n for n in names
            if optuna.load_study(study_name=n, storage=store).user_attrs.get("stock") == stock]


def walk(labs, *, first: int, refit: int, space: dict, start: dict, tune: bool = True,
         trials: int = 30, min_per_day: float = 0.5, seed: int = 0,
         last: int | None = None) -> dict:
    """:func:`optimize` every ``refit`` sessions from session ``first`` up to session ``last``
    (None: the end; sessions of the first stock's calendar), each block of ``refit`` sessions
    traded with the winner chosen on the sessions before it, on one stock or on several (one
    setting for all); ``tune=False``: the starting values (moved into the space) throughout,
    the control. The space must fix the candle size: a walk trades one.

    Returns:
        ``settings``: one row per tuning (its first traded session and day, the training score,
        and the winner's value of every hyperparameter the space leaves free); ``trades``: the
        stitched trades of every stock and traded session; ``position``: +1 / -1 / 0 at every
        traded candle (every stock's, one after the other).
    """
    labs = _labs(labs)
    if HP.is_free(space["candle_size"]):
        raise ValueError("a walk forward trades one candle size: fix it (one walk per size)")
    base = HP.project(start, space, labs[0].rules)
    free = [n for n in HP.table(labs[0].rules) if HP.is_free(space[n])]
    _, days = labs[0].sessions(base["candle_size"])
    end = len(days) if last is None else min(last, len(days))
    best = dict(base)
    rows, blocks = [], []
    for s0 in range(first, end, refit):
        res = None
        if tune:
            res = optimize(labs, days[s0], space=space, start=base, trials=trials,
                           min_per_day=min_per_day, seed=seed + s0, warm=(best,))
            best = res["best"]
        rows.append({"session": s0, "day": days[s0],
                     "score": np.nan if res is None else res["score"],
                     **({k: best[k] for k in free} if tune else {})})
        blocks.append((days[s0], dict(best)))
    stop = days[end] if end < len(days) else None
    sims = [_stitch(lab, base, blocks, stop) for lab in labs]
    return {"settings": pd.DataFrame(rows),
            "trades": pd.concat([x["trades"] for x in sims], ignore_index=True),
            "position": pd.concat([x["position"] for x in sims])}


def _stitch(lab: Lab, base: dict, blocks: list, stop) -> dict:
    """One stock's trades when every block (its first day, its settings) trades with its own
    settings, up to the day ``stop`` (None: the end), simulated in one go: a trade keeps the exit
    it opened with."""
    first = _day(lab, base, blocks[0][0])
    end = None if stop is None else _day(lab, base, stop)
    ctx0 = lab.context(base)
    n = len(ctx0["candles"])
    signal, flat = np.zeros(n, np.int8), np.zeros(n, bool)
    mult, maxs, dist = np.full(n, np.nan), np.zeros(n, np.int64), np.zeros(n)
    kind = np.full(n, "moves", dtype=object)
    up, down = np.full(n, np.nan), np.full(n, np.nan)
    move = ctx0["move"].to_numpy(float).copy()
    cond = ctx0["trend"]["stock"].to_numpy(float).copy()
    for i, (d0, s) in enumerate(blocks):
        nxt = _day(lab, s, blocks[i + 1][0]) if i + 1 < len(blocks) else end
        a, b = lab.rows(s, _day(lab, s, d0), nxt)
        ctx = lab.context(s)
        dec = lab.decide(s, a, b)
        signal[a:b], flat[a:b] = dec["signal"], dec["flat"]
        mult[a:b], maxs[a:b] = s["stop_moves"], s["max_sessions"]
        kind[a:b], dist[a:b] = s["stop_kind"], s["stop_days"]
        move[a:b], cond[a:b] = ctx["move"].iloc[a:b], ctx["trend"]["stock"].iloc[a:b]
        tgt = lab.targets(s)
        if tgt is not None:
            up[a:b], down[a:b] = tgt["long"].iloc[a:b], tgt["short"].iloc[a:b]
    a0, b0 = lab.rows(base, first, end)
    idx = ctx0["candles"].index[a0:b0]
    part = _cut(ctx0, slice(a0, b0))
    stitched = {**part, "move": pd.Series(move[a0:b0], index=idx),
                "trend": {**part["trend"], "stock": pd.Series(cond[a0:b0], index=idx)}}
    per = lambda x: pd.Series(x[a0:b0], index=idx)  # noqa: E731
    return SIM.run(stitched, pd.DataFrame({"signal": signal[a0:b0], "flat": flat[a0:b0]},
                                          index=idx),
                   stop_moves=per(mult), stop_kind=per(kind), stop_days=per(dist),
                   max_sessions=per(maxs),
                   targets=pd.DataFrame({"long": up[a0:b0], "short": down[a0:b0]}, index=idx),
                   **lab.costs)


def _spread(row: dict, points: int, current) -> list:
    """Up to ``points`` values across a hyperparameter's searched range (on its grid), and the
    current value."""
    if "choices" in row:
        return list(row["choices"])
    low, high, step = row["search"]
    grid = np.unique([_on_grid(row, v) for v in np.linspace(low, high, points)])
    return sorted({*grid.tolist(), current})


def effects(labs, settings: dict, *, names=None, first: int = 0, last=None,
            min_per_day: float = 0.5, points: int = 5) -> pd.DataFrame:
    """How each hyperparameter moves the score: one at a time across its range (``points``
    values, or every choice), everything else as in ``settings``, scored on sessions
    ``first`` .. ``last - 1`` (session numbers, or ``last`` a date) of one stock or several
    (their daily profits added up, as :func:`judge`).

    Returns:
        One row per hyperparameter and value: ``hyperparameter``, ``value``, ``as set`` (the
        value of ``settings``), ``score``, ``trades`` and ``P&L $``.
    """
    labs = _labs(labs)
    rows = HP.table(labs[0].rules)
    out = []
    for name in names or list(rows):
        for v in _spread(rows[name], points, settings[name]):
            s, series, n, ok = {**settings, name: v}, [], 0, True
            for lab in labs:
                t, daily = lab.daily(s, first, None if last is None else _day(lab, s, last))
                ok &= score(daily, len(t), min_per_day=min_per_day) > -1e8
                series.append(daily)
                n += len(t)
            together = pd.concat(series, axis=1).fillna(0.0).sum(axis=1)
            out.append({"hyperparameter": name, "value": v, "as set": v == settings[name],
                        "score": score(together, n, min_per_day=0.0) if ok else -1e9,
                        "trades": n, "P&L $": float(together.sum())})
    return pd.DataFrame(out)


def effect_summary(effects_: pd.DataFrame, settings: dict, rules: dict) -> pd.DataFrame:
    """One row per hyperparameter, the strongest first: the lowest and highest score among the
    values that traded enough, the difference (its **effect**), how many values traded too
    little to be scored, the value that scored best, and why one with no effect has none (it is
    not in use: its rule is off, or another stop or target kind is chosen; or no trade
    changed at all)."""
    rows = HP.table(rules)
    out = []
    for name, g in effects_.groupby("hyperparameter", sort=False):
        same = g["trades"].nunique() == 1 and g["P&L $"].round(6).nunique() == 1
        scored = g[g["score"] > -1e8]
        why = ("" if not same else "none: no trade changed" if HP.active(rows[name], settings)
               else f"none: not in use ({HP.needs_words(rows[name])})")
        out.append({"hyperparameter": name, "group": rows[name]["group"],
                    "value as set": settings[name],
                    "lowest score": scored["score"].min() if len(scored) else np.nan,
                    "highest score": scored["score"].max() if len(scored) else np.nan,
                    "effect": 0.0 if same else (scored["score"].max() - scored["score"].min()
                                                if len(scored) > 1 else np.nan),
                    "values trading too little": f"{len(g) - len(scored)} of {len(g)}",
                    "best value": scored.loc[scored["score"].idxmax(), "value"]
                    if len(scored) else None, "no effect?": why})
    return pd.DataFrame(out).sort_values("effect", ascending=False, kind="stable",
                                         na_position="last").reset_index(drop=True)


def _fingerprint(labs, config: dict) -> str:
    """A key that changes whenever anything a walk or a run reads changes: every stock's data,
    the costs, the configuration and the code it runs (not the notebooks or the charts)."""
    parts = [[lab.names, len(lab.minutes), str(lab.minutes.index[0]),
              str(lab.minutes.index[-1]), round(float(lab.minutes["close"].sum()), 4),
              {s: round(float(d["close"].sum()), 4) for s, d in lab.days.items()},
              sorted(map(str, lab.reports.get("reaction day", pd.Series(dtype=str)))),
              sorted(lab.costs.items()),
              None if lab.options is None or not len(lab.options)
              else round(float(lab.options.select_dtypes("number").sum().sum()), 4)]
             for lab in _labs(labs)]
    parts.append(sorted((k, repr(v)) for k, v in config.items()))
    files = [HERE / f"{n}.py" for n in ("indicators", "combine", "simulate", "data",
                                         "pipeline", "hyperparams", "tuning")]
    code = b"".join(p.read_bytes() for p in [*files, *sorted(HERE.glob("rules/*.py"))])
    return hashlib.sha1(repr(parts).encode() + code).hexdigest()[:16]


def _path(labs, name: str, config: dict, cache: Path) -> Path:
    """Where a walk is kept."""
    return Path(cache) / f"{group(labs)}-{name}-{_fingerprint(labs, config)}.pkl"


def walk_cached(labs, name: str, config: dict, *, cache: Path) -> dict:
    """:func:`walk` with ``config``, kept on disk under ``cache`` (a repeat is instant)."""
    path = _path(labs, name, config, cache)
    if path.exists():
        return pickle.loads(path.read_bytes())
    out = walk(labs, **config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def is_cached(labs, name: str, config: dict, *, cache: Path) -> bool:
    """Whether :func:`walk_cached` has this walk on disk already."""
    return _path(labs, name, config, cache).exists()


def run_job(path: Path) -> None:
    """Run the walk a job file describes (:func:`run_walks` writes them), kept on disk."""
    job = pickle.loads(Path(path).read_bytes())
    rules = COMBINE.load_rules()
    labs = []
    for file in job["data"]:
        d = pickle.loads(Path(file).read_bytes())
        labs.append(Lab(d["names"], d["minutes"], d["days"], d["reports"], rules, d["costs"],
                        options=d.get("options")))
    walk_cached(labs, job["name"], job["config"], cache=Path(job["cache"]))


def run_walks(jobs: list, *, cache: Path, workers: int = 1, progress=None) -> dict:
    """Every walk of ``jobs`` (``(labs, name, config)``: one lab or several, ``name`` labels
    the walk, ``config`` is :func:`walk`'s keyword arguments), kept on disk under ``cache``. The
    missing ones run here one after the other, or with ``workers`` > 1 in that many separate
    Python processes at once (each reads its stocks from files and its rules from theirs).

    Returns:
        (stock or group, name) -> the walk.
    """
    todo = [j for j in jobs if not is_cached(*j, cache=cache)]
    if workers > 1 and len(todo) > 1:
        tmp = Path(cache) / "jobs"
        tmp.mkdir(parents=True, exist_ok=True)
        data, queue = {}, []
        for k, (labs, name, config) in enumerate(todo):
            for lab in _labs(labs):
                stock = lab.names["stock"]
                if stock not in data:
                    data[stock] = tmp / f"data-{stock}.pkl"
                    data[stock].write_bytes(pickle.dumps(
                        {"names": lab.names, "minutes": lab.minutes, "days": lab.days,
                         "reports": lab.reports, "costs": lab.costs,
                         "options": lab.options}))
            job = tmp / f"job-{k}.pkl"
            job.write_bytes(pickle.dumps({
                "data": [str(data[lab.names["stock"]]) for lab in _labs(labs)], "name": name,
                "config": config, "cache": str(cache)}))
            queue.append(job)
        running, failed, done = [], [], 0
        while queue or running:
            while queue and len(running) < workers:
                job = queue.pop(0)
                log = job.with_suffix(".log").open("w")
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(job)],
                                        stdout=log, stderr=subprocess.STDOUT)
                running.append((job, proc, log))
            time.sleep(0.5)
            for item in [r for r in running if r[1].poll() is not None]:
                running.remove(item)
                item[2].close()
                if item[1].returncode:
                    failed.append(item[0].with_suffix(".log").read_text()[-3000:])
                done += 1
                if progress:
                    progress(done, len(todo))
        if failed:
            raise RuntimeError(f"{len(failed)} walk(s) failed; the first one's log:\n{failed[0]}")
        shutil.rmtree(tmp, ignore_errors=True)
    elif todo:
        for done, job in enumerate(todo, start=1):
            walk_cached(*job, cache=cache)
            if progress:
                progress(done, len(todo))
    return {(group(labs), name): walk_cached(labs, name, config, cache=cache)
            for labs, name, config in jobs}


if __name__ == "__main__":
    if sys.argv[1] == "continue":     # python tuning.py continue "<run name>" <trials> [worker]
        out = continue_run(sys.argv[2], int(sys.argv[3]),
                           worker=int(sys.argv[4]) if len(sys.argv) > 4 else 0)
        print(f"{sys.argv[2]}: {len(out['trials'])} trials, best {out['score']:.3f} at trial "
              f"{out['best trial']}, luck line {out['luck']:.3f}", flush=True)
    else:                             # one walk in its own process: python tuning.py <job file>
        run_job(Path(sys.argv[1]))
