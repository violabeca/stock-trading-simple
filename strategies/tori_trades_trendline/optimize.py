"""Tune every knob of Tori Trades' trendline bounce on many stocks at once.

The user (2026-10-20): tune all hyperparameters, fitted on several stocks at the same time to
reduce overfitting; report which ones have an edge and actually influence the outcome; try
different optimization strategies and objectives; run as many trials as still improve things.

Fixed before the first trial:

- **TRAIN**: 11 of the study's 21 stocks (every other one of ``study.STOCKS``, from NOW) over
  2026-03-30 to 2026-09-29. The optimizers only ever see TRAIN.
- **Held out**: OTHER STOCKS, the other 10 over the same months; EARLIER, all 21 over
  2025-09-29 to 2026-03-27, the six months before.
- **Space**: 13 knobs (:data:`SPACE`). The line knobs take a few values each, so a trial can
  reuse lines an earlier trial drew.
- **No trade floor** (the user, 2026-10-20: "if there is no signal to open a trade, then there
  should be no trade"). A setting that finds no signal makes no trade and scores 0, so a
  setting that rarely trades can win; every pick is reported with its trade count. (The first
  run pruned trials below 10 trades per stock; its random search was carried over, rescored.)
- **Objectives** (:data:`OBJECTIVES`), all computed from the per-stock results every trial keeps.
- **Studies** (:data:`STUDIES`): random search (every trial also scored on OTHER STOCKS, so each
  knob's importance can be checked out of sample), TPE on each objective (warm-started with the
  random trials), CMA-ES on the return, NSGA-II on return and drawdown. A study runs in batches of
  :data:`BATCH` trials and stops once its best TRAIN value has not improved by the objective's
  tolerance for two batches (at least 3 batches, at most 16; the random search runs 7).
- **Added the same day** (the user asked for three more objectives): ``lcb``, the lower
  confidence bound of the net return per trade (mean minus two standard errors, pooled over the
  TRAIN trades; fewer than two trades is no evidence and scores worst), which rewards evidence
  without a trade floor; ``alpha``, the return minus beta x the stock's own buy-and-hold return
  (beta from daily returns), which stops market exposure from counting as skill; and
  ``worst_third``, the mean return of the worst third of the TRAIN stocks (robustness; the
  median stock and the share of stocks positive are reported beside it). Each gets a TPE study;
  ``lcb`` and ``alpha`` start from their own random batch, because the earlier trials did not
  keep the numbers they need.
- **Judgement**: the 5 best TRAIN trials of every study, and the defaults, are replayed on both
  held-out sets. A knob *matters* when its fANOVA importance on the random trials is high on TRAIN
  and on OTHER STOCKS alike; a setting *has an edge* only when it makes money on both held-out
  sets (and beats buying and holding when it only buys).

Usage::

    uv run python strategies/tori_trades_trendline/optimize.py prepare
    uv run --with cmaes python strategies/tori_trades_trendline/optimize.py run [--only ...]
    uv run python strategies/tori_trades_trendline/optimize.py validate
    uv run python strategies/tori_trades_trendline/optimize.py report

The CMA-ES study needs the ``cmaes`` package, which is not a project dependency: ``--with
cmaes`` adds it for that run only. Everything goes to ``var/tori_trades_trendline/optimize/``
(git-ignored): ``data/`` (the
candles), ``studies.log`` (the Optuna journal), ``validation.pkl``, ``progress.log`` and
``report.txt``.
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
OUT = HERE.parents[1] / "var" / "tori_trades_trendline" / "optimize"
_spec = importlib.util.spec_from_file_location("tori_study_for_optimize", HERE / "study.py")
study_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study_module)
bounce = study_module.bounce
td = bounce.td

STOCKS = study_module.STOCKS
TRAIN, OTHER = STOCKS[::2], STOCKS[1::2]
#: The traded windows: start (first session) and end (last day of data).
WINDOWS = {"late": ("2026-03-30", "2026-09-29"), "early": ("2025-09-29", "2026-03-27")}
#: Every knob of the replay and the values it may take.
SPACE = {
    "tf": Choice(["15m", "1H"]), "log": Choice([False, True]),
    "touch": Choice([0.1, 0.25, 0.5, 1.0]), "pierce": Choice([0.0, 0.1, 0.25]),
    "confirm": Choice([1, 2, 3, 5]), "screen": Choice([80, 150, 300]),
    "refresh": Choice([True, False]), "sides": Choice(["long", "short", "both"]),
    "band": Range(0.0, 2.0), "room": Range(0.1, 3.0, log=True), "give": Range(0.0, 2.0),
    "trend": Choice([0, 20, 50, 100, 200]), "min_span": Range(0.0, 40.0)}
#: Her rules as the notebooks ship them.
DEFAULTS = {"tf": "15m", "log": False, "touch": 0.25, "pierce": 0.1, "confirm": 2,
            "screen": 150, "refresh": True, "sides": "both", "band": 0.25, "room": 0.5,
            "give": 0.0, "trend": 0, "min_span": 0.0}
BATCH, WORKERS, MIN_BATCHES, MAX_BATCHES = 28, 7, 3, 16


def _mean(group: dict, key: str) -> float:
    """The mean of one per-stock number over a group's stocks."""
    return float(np.mean([s[key] for s in group["stocks"].values()]))


def _consistency(group: dict) -> float:
    """t of the per-stock compounded returns: their mean over its standard error."""
    v = np.array([s["log"] for s in group["stocks"].values()])
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if v.std(ddof=1) > 0 else 0.0


def _lcb(group: dict) -> float:
    """The lower confidence bound of the net return per trade, pooled over the group's trades:
    mean minus two standard errors, in bp. Without two trades there is no evidence at all:
    -1000 bp, the worst score."""
    n = sum(s["trades"] for s in group["stocks"].values())
    if n < 2:
        return -1000.0
    total = sum(s["trade_sum"] for s in group["stocks"].values())
    mean = total / n
    var = max((sum(s["trade_sq"] for s in group["stocks"].values()) - n * mean ** 2) / (n - 1),
              0.0)
    return float(1e4 * (mean - 2.0 * np.sqrt(var / n)))


def _worst_third(group: dict) -> float:
    """The mean compounded return of the worst third of the group's stocks."""
    v = np.sort([s["log"] for s in group["stocks"].values()])
    return float(v[:int(np.ceil(len(v) / 3))].mean())


#: name -> (score of a group's results, tolerance a batch must beat to count as progress).
OBJECTIVES = {
    "return": (lambda g: _mean(g, "log"), 0.002),
    "excess": (lambda g: _mean(g, "log") - _mean(g, "bh"), 0.002),
    "consistency": (_consistency, 0.05),
    "sharpe": (lambda g: g["sharpe"], 0.05),
    "per_trade": (lambda g: 1e4 * sum(s["trade_sum"] for s in g["stocks"].values())
                  / max(sum(s["trades"] for s in g["stocks"].values()), 1), 0.5),
    "drawdown": (lambda g: _mean(g, "dd"), 0.002),
    # added 2026-10-20 (the user: try objectives 1-3): evidence, alpha, robustness
    "lcb": (_lcb, 0.5),
    "alpha": (lambda g: float(np.mean([s["log"] - s["beta"] * s["bh"]
                                       for s in g["stocks"].values()])), 0.002),
    "worst_third": (_worst_third, 0.002),
    "median": (lambda g: float(np.median([s["log"] for s in g["stocks"].values()])), 0.002),
    "share_pos": (lambda g: float(np.mean([s["log"] > 0 for s in g["stocks"].values()])),
                  0.02)}
#: name -> (sampler factory, objectives, warm start from the random trials).
STUDIES = {
    "random": (lambda: optuna.samplers.RandomSampler(seed=0), ["return"], False),
    "tpe-return": (lambda: _tpe(1), ["return"], True),
    "tpe-excess": (lambda: _tpe(2), ["excess"], True),
    "tpe-consistency": (lambda: _tpe(3), ["consistency"], True),
    "tpe-sharpe": (lambda: _tpe(4), ["sharpe"], True),
    "tpe-per_trade": (lambda: _tpe(5), ["per_trade"], True),
    "cmaes-return": (lambda: optuna.samplers.CmaEsSampler(
        seed=6, n_startup_trials=BATCH, warn_independent_sampling=False,
        independent_sampler=optuna.samplers.RandomSampler(seed=7)), ["return"], False),
    "nsga2-return-drawdown": (lambda: optuna.samplers.NSGAIISampler(
        population_size=BATCH, seed=8), ["return", "drawdown"], False),
    # added 2026-10-20; lcb and alpha read numbers the random trials did not keep, so they
    # start from their own random batch
    "tpe-lcb": (lambda: _tpe(9), ["lcb"], False),
    "tpe-alpha": (lambda: _tpe(10), ["alpha"], False),
    "tpe-worst_third": (lambda: _tpe(11), ["worst_third"], True)}


def _tpe(seed: int) -> optuna.samplers.TPESampler:
    """TPE for parallel workers: multivariate, pending trials scored as liars."""
    return optuna.samplers.TPESampler(seed=seed, multivariate=True, constant_liar=True,
                                      n_startup_trials=BATCH)


# ---- worker side: candles, chartbooks and drawn lines stay in memory between trials -------

_DATA: dict = {}
_BOOKS: dict = {}
_LINES: OrderedDict = OrderedDict()
_COSTS = None


def _start_worker() -> None:
    """Each worker reads the cost model once and keeps Optuna quiet."""
    global _COSTS
    from trading_simple.config import load_settings
    _COSTS = load_settings().costs


def _window(symbol: str, window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A stock's candles up to the end of ``window``."""
    if (symbol, window) not in _DATA:
        daily, minutes = pd.read_pickle(OUT / "data" / f"{symbol}.pkl")
        end = pd.Timestamp(WINDOWS[window][1])
        _DATA[symbol, window] = (daily[daily.index <= end],
                                 minutes[minutes.index.tz_localize(None).normalize() <= end])
    return _DATA[symbol, window]


def _book(symbol: str, window: str, tf: str) -> bounce.Chartbook:
    """The stock's charts down to ``tf`` over ``window``, built once."""
    if (symbol, window, tf) not in _BOOKS:
        daily, minutes = _window(symbol, window)
        order = list(td.TIMEFRAMES)
        _BOOKS[symbol, window, tf] = bounce.Chartbook(daily, minutes, order[:order.index(tf) + 1],
                                                      td.clock(daily))
    return _BOOKS[symbol, window, tf]


def _lines(symbol: str, window: str, p: dict) -> dict:
    """The lines drawn so far for this stock, window and line knobs (the 60 most recent sets)."""
    key = (symbol, window, p["tf"], p["log"], p["touch"], p["pierce"], p["confirm"], p["screen"])
    if key in _LINES:
        _LINES.move_to_end(key)
    else:
        _LINES[key] = {}
        while len(_LINES) > 60:
            _LINES.popitem(last=False)
    return _LINES[key]


def evaluate(params: dict, groups: dict[str, tuple[list[str], str]],
             shift: float = 0.0) -> dict:
    """Replay one setting on groups of stocks.

    Args:
        params: A value for every knob of :data:`SPACE`.
        groups: ``{name: (stocks, window)}``.
        shift: Placebo: every line moved this many typical candles toward price.

    Returns:
        ``{name: {"stocks": {symbol: numbers}, "sharpe": float}}``. Per stock: ``log``
        (log(1 + return)), ``bh`` (the same for buying and holding), ``trades``, ``long``
        (of them), ``trade_sum`` and ``trade_sq`` (the sum of the trades' returns and of their
        squares), ``dd`` (max drawdown) and ``beta`` (of the strategy's daily returns on the
        stock's); ``sharpe`` is the annualized daily Sharpe ratio of the equal-weight book of the
        stocks.
    """
    out = {}
    for name, (stocks, window) in groups.items():
        per, days = {}, []
        for symbol in stocks:
            daily, minutes = _window(symbol, window)
            r = bounce.replay(daily, minutes, start=WINDOWS[window][0], costs=_COSTS,
                              book=_book(symbol, window, params["tf"]),
                              cache=_lines(symbol, window, params), shift=shift, **params)
            t, s = r["trades"], r["summary"]
            closes = r["equity"].groupby(r["equity"].index.normalize()).last()
            moves = closes[["strategy", "buy and hold"]].pct_change().dropna()
            var = float(moves["buy and hold"].var())
            days.append(moves["strategy"].rename(symbol))
            per[symbol] = {
                "log": float(np.log1p(s["strategy return"])),
                "bh": float(np.log1p(s["buy and hold return"])), "trades": len(t),
                "long": int((t["side"] == "long").sum()) if len(t) else 0,
                "trade_sum": float(t["return"].sum()) if len(t) else 0.0,
                "trade_sq": float((t["return"] ** 2).sum()) if len(t) else 0.0,
                "dd": float(s["strategy max drawdown"]),
                "beta": float(moves["strategy"].cov(moves["buy and hold"]) / var) if var > 0
                else 0.0}
        daily_book = pd.concat(days, axis=1).fillna(0.0).mean(axis=1)
        sd = float(daily_book.std())
        out[name] = {"stocks": per,
                     "sharpe": float(daily_book.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0}
    return out


def _load_stock(symbol: str) -> str:
    """Read one stock's candles from the database and keep them up to the late window's end."""
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    daily, minutes = td.load(make_engine(load_settings()), symbol)
    end = pd.Timestamp(WINDOWS["late"][1])
    pd.to_pickle((daily[daily.index <= end],
                  minutes[minutes.index.tz_localize(None).normalize() <= end]),
                 OUT / "data" / f"{symbol}.pkl")
    return symbol


# ---- driver side ---------------------------------------------------------------------------

def _log(text: str) -> None:
    """A progress line, to the screen and to ``progress.log``."""
    line = f"{time.strftime('%H:%M:%S')} {text}"
    print(line, flush=True)
    with open(OUT / "progress.log", "a") as f:
        f.write(line + "\n")


def _storage() -> optuna.storages.JournalStorage:
    """The Optuna journal every study lives in."""
    from optuna.storages.journal import JournalFileBackend
    return optuna.storages.JournalStorage(JournalFileBackend(str(OUT / "studies.log")))


def _values(result: dict, objectives: list[str]) -> list[float]:
    """The trial's TRAIN scores (no trade floor: a setting without trades scores 0)."""
    return [OBJECTIVES[o][0](result["train"]) for o in objectives]


def _progress(study: optuna.Study) -> float | None:
    """The best TRAIN value so far (the best first objective on the Pareto front)."""
    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not done:
        return None
    if len(study.directions) == 1:
        return study.best_value
    return max(t.values[0] for t in study.best_trials)


def _warm(study: optuna.Study, objectives: list[str], storage) -> None:
    """Copy the random search's trials into ``study``, scored on its own objectives."""
    source = optuna.load_study(study_name="random", storage=storage)
    for t in source.trials:
        if "train" not in t.user_attrs:
            continue
        study.add_trial(optuna.trial.create_trial(
            params=t.params, distributions=SPACE,
            user_attrs={"train": t.user_attrs["train"], "warm": True},
            values=_values({"train": t.user_attrs["train"]}, objectives)))


def run_study(name: str, pool: ProcessPoolExecutor, storage) -> None:
    """Run one study in batches until it stops improving (see the module docstring)."""
    factory, objectives, warm = STUDIES[name]
    study = optuna.create_study(study_name=name, storage=storage, sampler=factory(),
                                directions=["maximize"] * len(objectives), load_if_exists=True)
    if warm and not study.trials:
        _warm(study, objectives, storage)
    groups = {"train": (TRAIN, "late")}
    if name == "random":
        groups["other"] = (OTHER, "late")
    best, stale, began = _progress(study), 0, time.time()
    _log(f"{name}: start, {len(study.trials)} trials so far, best TRAIN {best}")
    fixed = name == "random"
    for batch in range(1, (7 if fixed else MAX_BATCHES) + 1):
        running, asked = {}, 0
        while asked < BATCH or running:
            while asked < BATCH and len(running) < WORKERS:
                trial = study.ask(SPACE)
                running[pool.submit(evaluate, trial.params, groups)] = trial
                asked += 1
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                trial, result = running.pop(future), future.result()
                for key, value in result.items():
                    trial.set_user_attr(key, value)
                study.tell(trial, _values(result, objectives))
        now = _progress(study)
        gain = (now - best) if (now is not None and best is not None) else np.inf
        stale = 0 if gain > OBJECTIVES[objectives[0]][1] else stale + 1
        best = now if best is None or (now is not None and now > best) else best
        _log(f"{name}: batch {batch}, {len(study.trials)} trials, best TRAIN {now:.4f} "
             f"(gain {gain:+.4f}), {time.time() - began:.0f} s")
        if not fixed and batch >= MIN_BATCHES and stale >= 2:
            _log(f"{name}: stopped, no gain above "
                 f"{OBJECTIVES[objectives[0]][1]} for 2 batches")
            break


def _top(storage, n: int = 5) -> dict[str, dict]:
    """The ``n`` best TRAIN trials each study found itself (by its first objective; the
    Pareto front for a two-objective study), with the defaults."""
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


def validate(pool: ProcessPoolExecutor, storage) -> None:
    """Replay the best settings on TRAIN and both held-out sets, then the three best (by the
    held-out return) with every line moved one typical candle toward and away from price."""
    picks = _top(storage)
    groups = {"train": (TRAIN, "late"), "other": (OTHER, "late"), "early": (STOCKS, "early")}
    rows, began = [], time.time()
    futures = {pool.submit(evaluate, params, groups): key for key, params in picks.items()}
    for future in futures:
        rows.append({"setting": futures[future], "shift": 0.0, **future.result()})
    _log(f"validate: {len(rows)} settings in {time.time() - began:.0f} s")
    table = pd.DataFrame(rows)
    held = [(_mean(r["other"], "log") + _mean(r["early"], "log")) / 2 for _, r in table.iterrows()]
    best = table.assign(held=held).sort_values("held", ascending=False)["setting"].head(3)
    futures = {pool.submit(evaluate, picks[key], {"other": groups["other"],
                                                  "early": groups["early"]}, shift): (key, shift)
               for key in best for shift in (-1.0, 1.0)}
    for future in futures:
        key, shift = futures[future]
        rows.append({"setting": key, "shift": shift, **future.result()})
    pd.to_pickle({"picks": picks, "rows": rows}, OUT / "validation.pkl")
    _log(f"validate: placebo done, {time.time() - began:.0f} s")


def report(storage) -> str:
    """The optimization's tables, as text (also saved to ``report.txt``)."""
    from optuna.importance import FanovaImportanceEvaluator, get_param_importances

    out = []

    def show(title, frame):
        out.append(f"\n## {title}\n\n{frame.to_string()}\n")

    summary = []
    for name in STUDIES:
        try:
            study = optuna.load_study(study_name=name, storage=storage)
        except KeyError:
            continue
        own = [t for t in study.trials if not t.user_attrs.get("warm")]
        summary.append({"study": name, "objectives": ", ".join(STUDIES[name][1]),
                        "own trials": len(own),
                        "pruned": sum(t.state == TrialState.PRUNED for t in own),
                        "warm-start trials": len(study.trials) - len(own),
                        "best TRAIN": _progress(study)})
    show("Studies", pd.DataFrame(summary).set_index("study").round(4))

    random = optuna.load_study(study_name="random", storage=storage)
    trials = [t for t in random.trials if "other" in t.user_attrs]
    params = pd.DataFrame([t.params for t in trials])
    for objective in ("return", "excess", "per_trade", "sharpe", "worst_third"):
        score = OBJECTIVES[objective][0]
        train = np.array([score(t.user_attrs["train"]) for t in trials])
        other = np.array([score(t.user_attrs["other"]) for t in trials])
        rho = pd.Series(train).corr(pd.Series(other), method="spearman")
        importance = {}
        for label, values in (("TRAIN", train), ("OTHER", other)):
            s = optuna.create_study(direction="maximize")
            s.add_trials([optuna.trial.create_trial(params=t.params, distributions=SPACE,
                                                    value=float(v))
                          for t, v in zip(trials, values, strict=True)])
            importance[label] = get_param_importances(
                s, evaluator=FanovaImportanceEvaluator(seed=0), normalize=True)
        show(f"Random search ({len(trials)} trials): fANOVA importance for "
             f"{objective} (TRAIN vs OTHER STOCKS rank correlation over the trials: {rho:+.2f})",
             pd.DataFrame(importance).sort_values("TRAIN", ascending=False).round(3))
        effects = []
        for knob, dist in SPACE.items():
            bins = (params[knob].astype(str) if isinstance(dist, Choice)
                    else pd.qcut(params[knob], 4, duplicates="drop").astype(str))
            g = pd.DataFrame({"knob": knob, "value": bins, "TRAIN": train, "OTHER": other})
            effects.append(g.groupby(["knob", "value"]).agg(
                trials=("TRAIN", "size"), TRAIN=("TRAIN", "mean"), OTHER=("OTHER", "mean")))
        show(f"Random search: mean {objective} by knob value (TRAIN and OTHER STOCKS)",
             pd.concat(effects).round(4))

    stored = pd.read_pickle(OUT / "validation.pkl")
    picks, rows = stored["picks"], stored["rows"]
    table = []
    for r in rows:
        line = {"setting": r["setting"], "shift": r["shift"]}
        for group in ("train", "other", "early"):
            if group not in r:
                continue
            g = r[group]
            line |= {f"{group} return": _mean(g, "log"),
                     f"{group} excess": _mean(g, "log") - _mean(g, "bh"),
                     f"{group} alpha": OBJECTIVES["alpha"][0](g),
                     f"{group} worst third": _worst_third(g),
                     f"{group} median": OBJECTIVES["median"][0](g),
                     f"{group} stocks>0": sum(s["log"] > 0 for s in g["stocks"].values()),
                     f"{group} trades": _mean(g, "trades"),
                     f"{group} bp/trade": OBJECTIVES["per_trade"][0](g),
                     f"{group} lcb bp": _lcb(g), f"{group} sharpe": g["sharpe"]}
        table.append(line)
    table = pd.DataFrame(table)
    plain = table[table["shift"] == 0].drop(columns="shift").set_index("setting")
    for group, label in (("train", "TRAIN"), ("other", "OTHER STOCKS (same months)"),
                         ("early", "EARLIER (all 21, the six months before)")):
        cols = [c for c in plain.columns if c.startswith(f"{group} ")]
        show(f"Validation on {label}: the best TRAIN settings of every study; return = mean "
             "log(1 + r) per stock, excess = minus buying and holding, alpha = minus beta x "
             "buying and holding, lcb = mean - 2 se of the return per trade",
             plain[cols].rename(columns=lambda c: c.split(" ", 1)[1]).round(4))
    show("Validation: their knobs", pd.DataFrame(picks).T.loc[
        table.loc[table["shift"] == 0, "setting"]])
    show("Placebo: the three best held-out settings with every line moved one typical candle "
         "toward (+1) and away from (-1) price",
         table[table["setting"].isin(table.loc[table["shift"] != 0, "setting"])]
         .set_index(["setting", "shift"]).dropna(axis=1, how="all").round(4))
    text = "".join(out)
    (OUT / "report.txt").write_text(text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=["prepare", "run", "validate", "report"])
    parser.add_argument("--only", default=",".join(STUDIES),
                        help="studies to run, comma-separated (default: all, in order)")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    if args.what == "report":
        print(report(_storage()))
    elif args.what == "prepare":
        with ProcessPoolExecutor(WORKERS) as pool:
            for symbol in pool.map(_load_stock, STOCKS):
                _log(f"prepare: {symbol}")
    else:
        with ProcessPoolExecutor(WORKERS, initializer=_start_worker) as pool:
            if args.what == "run":
                for name in args.only.split(","):
                    run_study(name, pool, _storage())
            else:
                validate(pool, _storage())
