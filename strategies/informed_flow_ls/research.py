"""informed_flow_ls research helpers, loaded by path by the two marimo notebooks (``rs``).

``strategy.py`` never imports this file. It holds everything the notebooks share:

* the point-in-time universe (:func:`members`, :func:`pit_history`, :func:`member_mask`) and the
  panel loader (:func:`load`): informed_flow's rolling top 500 (v36-v41), with the membership
  carried into the strategy as a ``member`` feature so a training backtest never trades a name
  before the date it joined the universe (the forward-selection defect v36 found);
* the statistics (:func:`nw_t`, :func:`rank_ic`, :func:`residual_forward`, :func:`alpha_of`,
  :func:`bootstrap`, :func:`paired`, :func:`stitched`, :func:`pbo`);
* the nested walk-forward (:func:`tune`, :func:`period`): every 6 months an Optuna study runs on
  that period's past only and is scored on its WORST stretch, not its total (:func:`robust`);
  the pick then trades the next 6 months unseen, beside the untuned config and the controls;
* the luck band (:func:`shuffled_returns`) and the verdict (:func:`verdict`).

Nothing here changes a backtest: every book is run by the engine (``bt._run_scored``).
"""

from __future__ import annotations

import itertools
import os
import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sqlalchemy import text

from trading_simple import backtest as bt
from trading_simple import features
from trading_simple import optimize as opt
from trading_simple.data import load_panel

HERE = Path(__file__).resolve().parent
STUDIES = HERE / "studies"
SCORE_FROM = "2020-07-01"
WF_START = "2022-01-01"
WF_MONTHS = 6
ROLL = {"top_n": 500, "months": 6, "min_dollar_volume": 50e6, "min_price": 5.0, "min_days": 100}
#: Sessions left out between a training window's last scored bar and its test period.
PURGE = 21
#: Contiguous stretches a training window is cut into for the robust objective.
CV_BLOCKS = 4
#: Blocks of the combinatorially symmetric cross-validation behind :func:`pbo`.
PBO_BLOCKS = 8
N_BOOT, BOOT_BLOCK = 2000, 21
#: The pre-registered verdict (IDEAS.md, "Pre-registration v1"). Hard gates and warnings.
GATES = {"min_prob_alpha": 0.90, "luck_percentile": 95, "max_median_pbo": 0.5,
         "max_abs_beta": 0.3, "min_positive_share": 0.5, "late_fill_bps": 50,
         "cost_multiple": 2.0, "borrow_multiple": 2.0, "max_drawdown": -0.25}


# --- universe and panels ---------------------------------------------------------------------

_MEMBERS: dict = {}


def members(engine, te0) -> list[str]:
    """The point-in-time top ``ROLL["top_n"]`` equities at ``te0``, ranked only on the
    ``ROLL["months"]`` before it (median dollar volume, price and session-count floors)."""
    key = str(pd.Timestamp(te0).date())
    store = STUDIES / "members.pkl"                  # the ranking query takes ~5 s per date
    if not _MEMBERS and store.exists():
        _MEMBERS.update(pickle.loads(store.read_bytes()))
    if key not in _MEMBERS:
        start = pd.Timestamp(te0) - pd.DateOffset(months=ROLL["months"])
        ranked = pd.read_sql(text("""
            SELECT p.symbol,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY p.close * p.volume) AS dollar_volume,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY p.close) AS price,
                   count(*) AS days
            FROM prices_ohlcv p JOIN securities s ON s.symbol = p.symbol
            WHERE p.bar = '1d' AND p.ts >= :a AND p.ts < :b AND s.security_type = 'equity'
            GROUP BY p.symbol"""), engine, params={"a": start, "b": pd.Timestamp(te0)}
        ).set_index("symbol")
        pool = ranked[(ranked["days"] >= ROLL["min_days"])
                      & (ranked["dollar_volume"] >= ROLL["min_dollar_volume"])
                      & (ranked["price"] >= ROLL["min_price"])]
        _MEMBERS[key] = sorted(pool.nlargest(ROLL["top_n"], "dollar_volume").index)
        STUDIES.mkdir(exist_ok=True)
        store.write_bytes(pickle.dumps(dict(_MEMBERS)))
    return _MEMBERS[key]


def pit_history(engine, te0) -> dict:
    """Every membership formed from :data:`SCORE_FROM` to ``te0`` (every ``ROLL["months"]``):
    ``{formation date: members}`` — the universe a training window may use, date by date."""
    origins = pd.date_range(SCORE_FROM, te0, freq=f"{ROLL['months']}MS")
    return {o: members(engine, o) for o in origins}


def member_mask(index: pd.DatetimeIndex, columns, history: dict) -> pd.DataFrame:
    """1.0 where a name was a member ON that date (the latest formation at or before it; the
    first formation also covers the warm-up before it), else 0.0."""
    mask = pd.DataFrame(0.0, index=index, columns=columns)
    first = min(history)
    for o, names in sorted(history.items()):
        rows = index >= pd.Timestamp(o, tz="UTC") if o != first else slice(None)
        mask.loc[rows] = np.asarray(pd.Index(columns).isin(names), dtype=float)
    return mask


def load(engine, settings, strat, end, symbols, history: dict | None = None):
    """A daily panel from ``data.start`` to ``end`` with the config's features attached, plus
    the point-in-time ``member`` feature when ``history`` is given."""
    d = strat.data
    p = load_panel(engine, symbols, d["bar"], d["start"], end,
                   session=d.get("session", "regular"), exchanges=settings.exchanges)
    features.attach_features(p, engine, d["features"], settings=settings)
    if history is not None:
        p.features["member"] = member_mask(p.close.index, p.close.columns, history)
    attach_si_shares(p, engine)
    fees = borrow_fees(p.close.index, p.close.columns)
    if fees.notna().any().any():
        p.features["borrow_fee"] = fees
    return p


_SI_SHARES_SQL = """
    SELECT symbol, (known_at AT TIME ZONE 'UTC')::date AS d, settlement_date,
           short_interest AS value
    FROM short_interest
    WHERE symbol = ANY(:symbols) AND known_at <= :as_of
"""


def attach_si_shares(panel, engine) -> None:
    """Attach FINRA's shares sold short as ``panel.features["si_shares"]`` (2026-10-22).

    The strategy's crowding measure (short interest / shares outstanding: the 30% shorting cap and
    the borrow tiers) used to be days-to-cover x the panel's volume / SEC shares — and the stored
    volume is split-adjusted BACKWARD, so before a forward split it read the future split factor
    (NVDA: ~22% instead of 1.2% before its 2024 10:1 split; RESEARCH_LOG §42). FINRA's raw shares
    short and SEC's share count are both as of their own date. Same query, identity screens,
    as-of lag and 45-day max age as the days-to-cover feature (``features._SHORT_INTEREST_SQL``).

    Args:
        panel: A daily panel; its ``features`` dict is updated in place.
        engine: The shared-database engine (one read-only query).
    """
    symbols = list(panel.close.columns)
    names = symbols + sorted({old for sym in symbols
                              for old, *_ in features._SI_PREDECESSOR.get(sym, [])})
    with engine.connect() as conn:
        rows = pd.read_sql(text(_SI_SHARES_SQL), conn, params={
            "symbols": names, "as_of": pd.Timestamp(panel.close.index[-1]).to_pydatetime()})
    rows = features._screen_short_interest(rows)
    grid = features._to_grid(rows, panel.close.index, fill="ffill",
                             max_age_days=features._SI_MAX_AGE_DAYS)
    panel.features["si_shares"] = grid.reindex(columns=panel.close.columns)


#: Where ``scripts/ibkr_shortstock.py`` keeps its dated copies of IBKR's short-stock file.
IBKR_DIR = HERE.parents[1] / "var" / "ibkr"
#: A saved IBKR fee is used for this many days after its snapshot, then the tiers again.
IBKR_MAX_AGE_DAYS = 5


def borrow_fees(index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """Interactive Brokers' yearly borrow fee per name and bar, from the saved snapshots.

    Each ``usa-YYYYMMDD.txt`` copy is IBKR's FEERATE (percent a year) as of that morning, before
    the open: it serves that bar and the next :data:`IBKR_MAX_AGE_DAYS` days. The snapshots only
    exist from 2026-10-21, so a historical panel gets all NaN and the strategy's tiers stand.

    Args:
        index: The panel's bars.
        columns: The panel's symbols.

    Returns:
        A bars x symbols frame of yearly rates (0.004 = 0.4%), NaN where no snapshot applies.
    """
    out = pd.DataFrame(np.nan, index=index, columns=columns)
    for path in sorted(IBKR_DIR.glob("usa-*.txt")):
        day = pd.Timestamp(path.stem.split("-")[1], tz="UTC")
        raw = pd.read_csv(path, sep="|", skiprows=1, dtype=str)
        raw.columns = [c.lstrip("#") for c in raw.columns]
        raw = raw[~raw["SYM"].fillna("#").str.startswith("#")].drop_duplicates("SYM")
        fee = pd.to_numeric(raw.set_index("SYM")["FEERATE"], errors="coerce") / 100
        rows = (index >= day) & (index <= day + pd.Timedelta(days=IBKR_MAX_AGE_DAYS))
        out.loc[rows] = fee.reindex(columns).to_numpy()[None, :].repeat(rows.sum(), axis=0)
    return out


def cached_load(engine, settings, strat, end, symbols, history: dict | None = None):
    """:func:`load`, kept on disk (``studies/panels/``) keyed by everything that shapes the
    panel — the window, the names, the features and the memberships — so a walk-forward rerun
    or a variant reads its panels in seconds instead of minutes."""
    import hashlib

    d = strat.data
    key = repr((str(end), tuple(symbols), tuple(d["features"]), d["start"], d["bar"],
                sorted((str(k), tuple(v)) for k, v in (history or {}).items())))
    path = STUDIES / "panels" / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.pkl"
    if path.exists():
        panel = pickle.loads(path.read_bytes())
        attach_si_shares(panel, engine)          # always fresh: never a stale pickled copy
        return panel
    panel = load(engine, settings, strat, end, symbols, history)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")          # parallel workers: write, then rename
    tmp.write_bytes(pickle.dumps(panel))
    os.replace(tmp, path)
    return panel


def folds(last) -> list[tuple[str, str, str]]:
    """The walk-forward periods: ``(train end, test start, test end)`` every
    :data:`WF_MONTHS` from :data:`WF_START` to ``last``."""
    origins = pd.date_range(WF_START, last, freq=f"{WF_MONTHS}MS")
    out = []
    for o in origins:
        te1 = min(o + pd.DateOffset(months=WF_MONTHS) - pd.Timedelta(days=1), pd.Timestamp(last))
        out.append((str((o - pd.Timedelta(days=1)).date()), str(o.date()), str(te1.date())))
    return out


# --- statistics --------------------------------------------------------------------------------

def nw_t(x, lag: int) -> float:
    """Newey-West (Bartlett) t of a series' mean: overlapping windows autocorrelate."""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    var = e @ e / n
    for j in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - j / (lag + 1.0)) * (e[j:] @ e[:-j]) / n
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else np.nan


def rank_ic(score: pd.DataFrame, rel: pd.DataFrame) -> pd.Series:
    """Daily Spearman rank correlation of ``score`` with ``rel`` over jointly present names."""
    m = score.notna() & rel.notna()
    return score.where(m).rank(axis=1).corrwith(rel.where(m).rank(axis=1), axis=1).dropna()


def trailing_beta(close: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Each name's beta to the equal-weight panel over the trailing ``window`` bars."""
    r = close.pct_change(fill_method=None)
    m = r.mean(axis=1)
    return r.rolling(window, min_periods=window // 2).cov(m).div(
        m.rolling(window, min_periods=window // 2).var(), axis=0)


def residual_forward(close: pd.DataFrame, h: int) -> pd.DataFrame:
    """The ``h``-bar return a signal at bar t can earn — from t+1's close to t+1+h's (the trade
    fills at t+1's open; skipping that first bar is the conservative proxy) — minus trailing
    beta × the market's same return: the selection part only."""
    fwd = close.shift(-h - 1) / close.shift(-1) - 1.0
    return fwd - trailing_beta(close).mul(fwd.mean(axis=1), axis=0)


def alpha_of(r, bench, years):
    """Geometric CAGR, buy-and-hold CAGR, beta and alpha (= CAGR − beta × B&H CAGR) of aligned
    daily return rows (1-d or 2-d: one row per resample)."""
    r, bench = np.atleast_2d(r), np.atleast_2d(bench)
    cagr = np.expm1(np.log1p(r).sum(axis=1) / years)
    bench_cagr = np.expm1(np.log1p(bench).sum(axis=1) / years)
    rc, bc = r - r.mean(axis=1, keepdims=True), bench - bench.mean(axis=1, keepdims=True)
    var = (bc * bc).sum(axis=1)
    beta = np.divide((rc * bc).sum(axis=1), var, out=np.zeros_like(var), where=var > 0)
    return cagr, bench_cagr, beta, cagr - beta * bench_cagr


def block_index(size: int, n: int = N_BOOT, block: int = BOOT_BLOCK, seed: int = 7):
    """Stationary block-bootstrap resamples (Politis-Romano) of positions 0..size-1."""
    rng = np.random.default_rng(seed)
    pos = np.arange(size)
    new = rng.random((n, size)) < 1.0 / block
    new[:, 0] = True
    start = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    origin = np.take_along_axis(np.where(new, rng.integers(0, size, (n, size)), 0), start,
                                axis=1)
    return (origin + pos - start) % size


def bootstrap(r, bench, years, seed: int = 7) -> dict:
    """P(alpha > 0) and the alpha's 5th / 95th percentiles under a stationary block bootstrap
    (the book and buy-and-hold resampled on the same blocks)."""
    r, bench = np.asarray(r, float), np.asarray(bench, float)
    idx = block_index(len(r), seed=seed)
    alpha = alpha_of(r[idx], bench[idx], years)[3]
    return {"P(alpha > 0)": float((alpha > 0).mean()), "alpha p5": float(np.percentile(alpha, 5)),
            "alpha p95": float(np.percentile(alpha, 95))}


def paired(r, base, bench, years, seed: int = 11) -> dict:
    """Alpha of ``r`` minus alpha of ``base`` on the SAME days and the same bootstrap blocks."""
    r, base, bench = (np.asarray(x, float) for x in (r, base, bench))
    idx = block_index(len(r), seed=seed)
    diff = alpha_of(r[idx], bench[idx], years)[3] - alpha_of(base[idx], bench[idx], years)[3]
    point = alpha_of(r, bench, years)[3][0] - alpha_of(base, bench, years)[3][0]
    return {"delta alpha": float(point), "P(delta > 0)": float((diff > 0).mean())}


def stitched(pairs) -> dict:
    """One continuous out-of-sample record from per-period ``(book, buy-and-hold)`` daily
    return Series: CAGR, beta, alpha, the bootstrap, max drawdown, the daily series."""
    r = pd.concat([x for x, _ in pairs])
    b = pd.concat([y for _, y in pairs]).reindex(r.index).fillna(0.0)
    years = bt._years(r.index)
    cagr, bench_cagr, beta, alpha = (float(x[0]) for x in alpha_of(r.to_numpy(), b.to_numpy(),
                                                                   years))
    curve = (1 + r).cumprod()
    return {"years": years, "cagr": cagr, "buy & hold": bench_cagr, "beta": beta,
            "alpha": alpha, "volatility": float(r.std() * np.sqrt(len(r) / years)),
            "max drawdown": float((curve / curve.cummax() - 1).min()),
            **bootstrap(r.to_numpy(), b.to_numpy(), years), "returns": r, "bench": b}


def pbo(returns: pd.DataFrame, blocks: int = PBO_BLOCKS) -> float:
    """Probability of backtest overfitting (Bailey, Borwein, López de Prado & Zhu 2017).

    The trials' daily returns are cut into ``blocks`` contiguous blocks; for every way of
    choosing half of them as "in sample", the trial with the best in-sample Sharpe ratio is
    ranked among all trials on the other half. PBO is the share of splits in which that pick
    lands in the bottom half out of sample: 0 = the best stays good, 0.5 = picking is a coin
    flip, near 1 = the search finds the stretch, not the strategy.

    Args:
        returns: Bars x trials daily returns over one training window.
        blocks: An even number of blocks.

    Returns:
        The PBO (NaN with fewer than 3 trials).
    """
    if returns.shape[1] < 3:
        return np.nan
    parts = np.array_split(returns.to_numpy(dtype=float), blocks)
    below = []
    for ins in itertools.combinations(range(blocks), blocks // 2):
        a = np.vstack([parts[i] for i in ins])
        b = np.vstack([parts[i] for i in range(blocks) if i not in ins])
        sharpe = lambda x: x.mean(0) / np.where(x.std(0) > 0, x.std(0), np.nan)  # noqa: E731
        best = np.nanargmax(sharpe(a))
        oos = pd.Series(sharpe(b)).rank(pct=True).to_numpy()
        below.append(oos[best] <= 0.5)
    return float(np.mean(below))


# --- one book ---------------------------------------------------------------------------------

def book(strat, panel, params: dict, settings, start, end=None) -> dict:
    """Run one parameter set warm (signals on the whole panel, book from ``start``) and return
    its daily returns, buy-and-hold's, the engine metrics and the portfolio."""
    pf, w, fees, slip = bt._run_scored(strat, panel, params, settings, start)
    stats = bt.portfolio_stats(pf)
    _, bench = bt.benchmark(w, fees=fees, slippage=slip, fill=strat.fill)
    r = bt.equity_curve(pf)["value"].pct_change().fillna(0.0)
    if end is not None:
        end = pd.Timestamp(end)
        cut = r.index <= (end if end.tzinfo else end.tz_localize("UTC"))
        r, bench = r[cut], bench.reindex(r.index)[cut]
    return {"returns": r, "bench": bench.reindex(r.index).fillna(0.0), "stats": stats, "pf": pf}


def robust(returns: pd.Series, bench: pd.Series, blocks: int = CV_BLOCKS) -> float:
    """The inner objective: the mean of the training window's per-stretch alphas minus half
    their spread. A parameter set that earns everything in one stretch scores below one that
    earns a little in each — the optimizer is pushed toward what held up, not what peaked."""
    alphas = []
    for idx in np.array_split(np.arange(len(returns)), blocks):
        r, b = returns.iloc[idx], bench.iloc[idx]
        alphas.append(float(alpha_of(r.to_numpy(), b.to_numpy(), bt._years(r.index))[3][0]))
    return float(np.mean(alphas) - 0.5 * np.std(alphas))


# --- the nested walk-forward -----------------------------------------------------------------

def tune(strat, train, settings, name: str, *, n_trials: int, seed: int, score_end,
         late_bps: float = 0.0, pick: str = "best", guard: float = 0.0) -> dict:
    """One period's inner study: TPE over the config's ``search:`` space on the training panel,
    every trial scored warm from :data:`SCORE_FROM` to ``score_end`` by :func:`robust`.

    Resumable: the study lives in ``studies/<name>.log`` (Optuna journal) and every trial's
    daily returns in ``studies/<name>.returns.pkl`` (for :func:`pbo`). Trials trading less than
    ``optuna.min_trades_per_year`` are pruned.

    ``late_bps`` > 0 scores every trial with that much extra slippage a leg (a study that
    prefers settings whose edge survives worse fills). ``pick="plateau"`` takes the trial
    whose score averaged with its 4 nearest neighbours (in the search space scaled to 0-1) is
    best, instead of the single best trial. ``guard`` > 0 runs the config's own values as the
    first trial and keeps them unless the pick beats them by ``guard`` (robust alpha).

    Returns:
        ``{"best": params, "score": robust score, "trials": frame, "pbo": float}``.
    """
    STUDIES.mkdir(exist_ok=True)
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(
        str(STUDIES / f"{name}.log")))
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(20, n_trials // 3))
    study = optuna.create_study(study_name=name, storage=storage, direction="maximize",
                                sampler=sampler, load_if_exists=True)
    space = strat.optuna["space"]
    stamp = {"space": repr(space), "objective": f"robust{CV_BLOCKS}", "score_end": str(score_end)}
    if late_bps:
        stamp["late_bps"] = late_bps
    if guard:
        stamp["guard"] = guard
    scoring = with_costs(settings, 1.0, late_bps) if late_bps else settings
    if study.user_attrs and study.user_attrs.get("stamp") != stamp:
        raise ValueError(f"study {name} was run with another space/objective/window")
    study.set_user_attr("stamp", stamp)
    store = STUDIES / f"{name}.returns.pkl"
    returns = pickle.loads(store.read_bytes()) if store.exists() else {}
    floor = strat.optuna.get("min_trades_per_year", 0)

    def objective(trial):
        params = opt.suggest_params(trial, space)
        if not strat.valid({**strat.params, **params}):
            raise optuna.TrialPruned()
        b = book(strat, train, params, scoring, SCORE_FROM, end=score_end)
        if b["stats"]["trades_per_year"] < floor:
            raise optuna.TrialPruned()
        returns[trial.number] = b["returns"]
        store.write_bytes(pickle.dumps(returns))
        return robust(b["returns"], b["bench"])

    if guard and not study.trials:
        study.enqueue_trial({k: strat.params[k] for k in space})
    left = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if left > 0:
        study.optimize(objective, n_trials=left)
    done = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    done = done[done["state"] == "COMPLETE"]
    best = study.best_trial
    chosen, score = dict(best.params), best.value
    if pick == "plateau" and len(done) > 5:
        chosen, score = plateau(done, space)
    if guard:
        first = study.trials[0]
        if first.value is not None and score - first.value < guard:
            chosen, score = dict(first.params), first.value
    matrix = pd.DataFrame({k: v for k, v in returns.items() if k in set(done["number"])})
    return {"best": chosen, "score": score, "trials": done,
            "pbo": pbo(matrix.dropna()) if len(matrix) else np.nan}


def plateau(done: pd.DataFrame, space: dict, k: int = 5) -> tuple[dict, float]:
    """The trial whose score averaged with its ``k - 1`` nearest neighbours (the search space
    scaled to 0-1 per knob) is best: a setting whose surroundings also work.

    Args:
        done: Completed trials (``value`` and ``params_<knob>`` columns).
        space: The search space (``type``, ``low``, ``high`` per knob).
        k: Neighbourhood size, the trial itself included.

    Returns:
        ``(params, smoothed score)``.
    """
    cols = [f"params_{name}" for name in space]
    unit = np.column_stack([(done[c] - space[n]["low"]) / (space[n]["high"] - space[n]["low"])
                            for c, n in zip(cols, space, strict=True)])
    dist = np.linalg.norm(unit[:, None, :] - unit[None, :, :], axis=2)
    near = np.argsort(dist, axis=1, kind="stable")[:, :k]
    smooth = done["value"].to_numpy()[near].mean(axis=1)
    row = done.iloc[int(np.argmax(smooth))]
    params = {n: (int(row[f"params_{n}"]) if space[n]["type"] == "int"
                  else float(row[f"params_{n}"])) for n in space}
    return params, float(smooth.max())


def period(engine, settings, strat, fold, *, n_trials: int, arms: dict, seed: int = 42,
           late_bps: float = 0.0, pick: str = "best", guard: float = 0.0) -> dict:
    """One walk-forward period; results cached in ``studies/wf-<study>-<test start>.pkl``
    (an arm added later is run alone and added to the cache).

    Trains on every past membership point in time (``member`` feature), scored to
    :data:`PURGE` sessions before the test period; tunes (:func:`tune`); then trades the test
    period on its own load of ``members(test start)``, warm, for every arm.

    Args:
        arms: ``{label: (base, overrides, settings or None)}`` — ``base`` is ``"tuned"`` (this
            period's pick) or ``"config"`` (the shipped values); ``overrides`` go on top.

    Returns:
        ``{"fold", "tuned": params, "score", "pbo", "returns": {arm: (book, bench)},
        "stats": {arm: engine metrics}, "names": members}``.
    """
    tr1, te0, te1 = fold
    cache = STUDIES / f"wf-{strat.optuna['study']}-{te0}.pkl"
    out = pickle.loads(cache.read_bytes()) if cache.exists() else None
    if out is None:
        history = pit_history(engine, te0)
        train = cached_load(engine, settings, strat, tr1,
                            sorted(set().union(*history.values())), history)
        score_end = train.close.index[-1 - PURGE]
        t = tune(strat, train, settings, f"{strat.optuna['study']}-wf-{te0}",
                 n_trials=n_trials, seed=seed, score_end=score_end, late_bps=late_bps,
                 pick=pick, guard=guard)
        out = {"fold": fold, "tuned": t["best"], "score": t["score"], "pbo": t["pbo"],
               "trials": t["trials"], "returns": {}, "stats": {},
               "names": members(engine, te0)}
    todo = {k: v for k, v in arms.items() if k not in out["returns"]}
    if todo:
        window = cached_load(engine, settings, strat, te1, out["names"])
        for label, (base, overrides, cfg) in todo.items():
            params = {**(out["tuned"] if base == "tuned" else {}), **overrides}
            b = book(strat, window, params, cfg or settings, te0)
            out["returns"][label] = (b["returns"], b["bench"])
            out["stats"][label] = b["stats"]
        STUDIES.mkdir(exist_ok=True)
        cache.write_bytes(pickle.dumps(out))
    return out


def compose_params(mod, params: dict) -> dict:
    """The keyword arguments ``strategy.compose`` takes, read from a full parameter dict."""
    import inspect

    names = inspect.signature(mod["compose"]).parameters
    return {k: params[k] for k in names if k != "panel" and k in params}


def shuffled_returns(strat, settings, engine, fold, params: dict, n: int,
                     seed: int = 23) -> list[pd.Series]:
    """The luck band for one period: ``n`` books that keep the strategy's every rule but hold
    RANDOM names — each a fixed relabelling of the composite's columns within dollar-volume
    tenths (ranked on the 126 sessions before the test period), so turnover, holding times and
    liquidity match and only the choice of names is random (never per-bar random ranks: they
    churn the book daily and make the test unfailable, informed_flow v25).

    Returns:
        ``n`` daily return Series over the test period.
    """
    import hashlib

    tr1, te0, te1 = fold
    key = hashlib.sha1(repr(sorted({**strat.params, **params}.items())).encode()).hexdigest()[:8]
    cache = STUDIES / f"luck-{strat.optuna['study']}-{te0}-{n}-{seed}-{key}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    window = cached_load(engine, settings, strat, te1, members(engine, te0))
    mod = strat.signals.__globals__
    p = {**strat.params, **params}
    parts = mod["compose"](window, **compose_params(mod, p))
    before = window.close.index < pd.Timestamp(te0, tz="UTC")
    dollars = (window.close * window.volume)[before].tail(126).median()
    tenth = np.ceil(dollars.rank(pct=True) * 10).fillna(0)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        cols = window.close.columns.to_numpy().copy()
        for g in np.unique(tenth):
            inside = np.flatnonzero(tenth.to_numpy() == g)
            cols[inside] = cols[rng.permutation(inside)]
        score = parts["score"][cols].set_axis(window.close.columns, axis=1)
        score = score.where(parts["eligible"]).rank(axis=1, pct=True)
        longs, shorts, _ = mod["_select"](score, window.close, parts["shortable"], q_in=p["q_in"],
                                          q_out=p["q_out"],
                                          slots=(p["long_slots"], p["short_slots"]),
                                          max_hold=p["max_hold"], short_stop=p["short_stop"],
                                          sides=p["sides"])
        share = p.get("long_share", 0.5)
        w = (bt.next_bar(longs).astype(float) * (p["gross"] * share / p["long_slots"])
             - bt.next_bar(shorts).astype(float) * (p["gross"] * (1 - share) / p["short_slots"]))
        sig = {"weights": w, "borrow": parts["borrow"], "retrim_k": p["retrim_k"]}
        win, sig = bt._score_window(window, sig, te0)
        fees, slip = bt.strategy_costs(strat, win, settings)
        pf = bt.run(win, sig, fees=fees, slippage=slip, init_cash=settings.backtest.init_cash,
                    fill=strat.fill)
        out.append(bt.equity_curve(pf)["value"].pct_change().fillna(0.0))
    cache.write_bytes(pickle.dumps(out))
    return out


def verdict(primary: dict, untuned: dict, luck: np.ndarray, pbos: list[float],
            stress: dict, period_alphas: list[float], controls: dict) -> tuple[pd.DataFrame, str]:
    """The pre-registered verdict (IDEAS.md): hard gates, then warnings.

    Args:
        primary: :func:`stitched` of the tuned arm (the procedure the user chose).
        untuned: :func:`stitched` of the config arm, same days.
        luck: Stitched alphas of the shuffled books.
        pbos: Each period's :func:`pbo`.
        stress: ``{label: stitched}`` of the primary under 2x costs, 2x borrow, late fills.
        period_alphas: The primary's alpha in each period.
        controls: ``{label: stitched}`` — long side alone + SPY hedge, no option sleeve.

    Returns:
        ``(table, "PASS" | "FAIL")`` — the table has gate, value, bar, passed, kind.
    """
    g = GATES
    pair = paired(primary["returns"], untuned["returns"], primary["bench"], primary["years"])
    rows = [
        ("walk-forward alpha > 0", primary["alpha"], "> 0", primary["alpha"] > 0, "hard"),
        ("bootstrap P(alpha > 0)", primary["P(alpha > 0)"], f">= {g['min_prob_alpha']}",
         primary["P(alpha > 0)"] >= g["min_prob_alpha"], "hard"),
        ("beats random-name books (percentile)", float((luck < primary["alpha"]).mean() * 100),
         f">= {g['luck_percentile']}",
         (luck < primary["alpha"]).mean() * 100 >= g["luck_percentile"], "hard"),
        ("tuning beats the untuned config (delta alpha)", pair["delta alpha"], ">= 0",
         pair["delta alpha"] >= 0, "hard"),
        ("median probability of overfitting", float(np.nanmedian(pbos)),
         f"<= {g['max_median_pbo']}", float(np.nanmedian(pbos)) <= g["max_median_pbo"], "hard"),
        ("|beta| (market-neutral)", abs(primary["beta"]), f"<= {g['max_abs_beta']}",
         abs(primary["beta"]) <= g["max_abs_beta"], "hard"),
        ("periods with alpha > 0", float(np.mean(np.array(period_alphas) > 0)),
         f">= {g['min_positive_share']}",
         np.mean(np.array(period_alphas) > 0) >= g["min_positive_share"], "hard"),
        *[(f"alpha > 0 with {k}", v["alpha"], "> 0", v["alpha"] > 0, "hard")
          for k, v in stress.items()],
        ("max drawdown", primary["max drawdown"], f">= {g['max_drawdown']}",
         primary["max drawdown"] >= g["max_drawdown"], "warning"),
        *[(f"beats control: {k}", primary["alpha"] - v["alpha"], "> 0",
           primary["alpha"] > v["alpha"], "warning") for k, v in controls.items()],
    ]
    table = pd.DataFrame(rows, columns=["gate", "value", "bar", "passed", "kind"])
    passed = table.loc[table["kind"] == "hard", "passed"].all()
    return table, "PASS" if passed else "FAIL"


def with_costs(settings, multiple: float = 1.0, late_bps: float = 0.0):
    """Settings with every cost scaled by ``multiple`` and ``late_bps`` added to each leg's
    slippage (orders filled worse than the open)."""
    c = settings.costs
    costs = c.model_copy(update={"commission_per_share": c.commission_per_share * multiple,
                                 "min_commission": c.min_commission * multiple,
                                 "spread_bps": c.spread_bps * multiple,
                                 "slippage_bps": c.slippage_bps * multiple + late_bps})
    return settings.model_copy(update={"costs": costs})
