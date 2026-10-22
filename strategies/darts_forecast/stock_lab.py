"""One stock under the microscope (``darts_forecast_stock_marimo.py``): its data and split, NEW
inputs defined in the notebook beside ``tsmixer_lab``'s ten, a look-ahead check and an
information check for each, and one configuration trained over several seeds, so that a change
of inputs can be told apart from seed noise.

New helpers live here, never in ``tsmixer_lab.py``: that file's source hash is part of every
per-stock study's name, so one edit there orphans the studies.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    """A helper module of this folder, loaded by path once (shared with the other notebooks)."""
    key = f"darts_{name}"
    path = HERE / f"{name}.py"
    mod = sys.modules.get(key)
    if mod is not None and Path(getattr(mod, "__file__", "")) == path:
        return mod
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


lab = _sibling("tsmixer_lab")
ds, ex = lab.ds, lab.ex
_BASE_EXAMPLES = lab.examples
LOG_DIR = lab.BASKET_DIR / "stock"  # one experiment log per stock (git-ignored)


# ---------------------------------------------------------------------------------------------
# The data.
# ---------------------------------------------------------------------------------------------
def load(settings, engine, symbol: str, *, train, validation, test, decision: int) -> dict:
    """One stock's 1-minute bars from TRAIN's first day to TEST's last (dividend-adjusted), every
    ``tsmixer_lab`` input built from them, the arrays new inputs are computed from, and the split
    (the same as ``darts_forecast_tsmixer_marimo.py``'s).

    Args:
        settings: Repo settings.
        engine: Database engine.
        symbol: The stock.
        train: TRAIN's (first, last) day.
        validation: VALIDATION's (first, last) day.
        test: TEST's (first, last) day.
        decision: The decision minute (minutes after 09:30).

    Returns:
        ``F`` (``tsmixer_lab.build``), ``raw`` (:func:`raw_arrays`), ``SP`` (each part's first
        and last session), ``fit`` / ``held`` / ``val`` / ``test`` (session indices: learn,
        hold out for early stopping and β, score, read once).
    """
    panel = ds.rs.load(settings, engine, start=train[0], end=test[1], options=False,
                       symbols=[symbol], adjust=True)
    if symbol not in panel.symbols:
        raise ValueError(f"no 1-minute data for {symbol}")
    F = lab.build(panel, engine, basket_from=train[0], basket_to=test[1], options=False)
    dates = np.array(F["dates"], dtype=object)
    sp = {"sessions": list(dates)}
    for part, (a, b) in {"train": train, "validation": validation, "test": test}.items():
        inside = [d for d in dates if a <= d <= b]
        if not inside:
            raise ValueError(f"{symbol}: no stored session in {part.upper()} {a} → {b}")
        sp[part] = (inside[0], inside[-1])
    fit, held, val = ds.split_days(F["pp"], sp, "validation", train[0])
    later = np.flatnonzero(F["pp"]["usable"] & (dates >= sp["test"][0]) & (dates <= sp["test"][1]))
    return {"F": F, "raw": raw_arrays(panel, F, decision), "SP": sp, "fit": fit, "held": held,
            "val": val, "test": later}


def raw_arrays(panel, F: dict, decision: int) -> dict:
    """What a new input is computed from, for the panel's (one) stock.

    Returns:
        Sessions x 390 minutes: ``open``, ``high``, ``low``, ``close`` (NaN in a minute without a
        trade), ``volume`` (0 without a trade), ``y`` (each minute's move in typical moves: the
        model's own path), ``market`` / ``sector`` (the baskets' minute moves in their typical
        moves); per session: ``day_close`` (the last trade); and ``dates``, ``decision``.
    """
    p = ds.full_clock(panel)
    n = len(F["dates"])
    traded = F["pp"]["volume"][..., 0] > 0
    out = {k: np.where(traded, np.ascontiguousarray(getattr(p, a).to_numpy(float)[:, 0])
                       .reshape(n, lab.SESSION), np.nan)
           for k, a in (("open", "open_"), ("high", "high"), ("low", "low"), ("close", "close"))}
    return {**out, "volume": F["pp"]["volume"][..., 0].astype(float),
            "y": F["y"][..., 0].astype(float), "market": F["bar"]["market"][..., 0].astype(float),
            "sector": F["bar"]["sector"][..., 0].astype(float),
            "day_close": F["pp"]["close"][:, 0].astype(float), "dates": F["dates"],
            "decision": int(decision)}


# ---------------------------------------------------------------------------------------------
# New inputs: how to write one, and their checks.
# ---------------------------------------------------------------------------------------------
def registry():
    """An empty set of new inputs and the decorator that adds one.

    Returns:
        ``(found, add)``: ``@add("day" | "bar", what=..., why=...)`` above a function of ``raw``
        (:func:`raw_arrays`) puts it in ``found`` under the function's name. A "day" input
        returns one value per session, a "bar" input one per session and minute.
    """
    found = {}

    def add(level: str, *, what: str, why: str = ""):
        if level not in ("day", "bar"):
            raise ValueError(f"level must be 'day' or 'bar', got {level!r}")

        def keep(fn):
            found[fn.__name__] = {"level": level, "fn": fn, "what": what, "why": why}
            return fn

        return keep

    return found, add


def typical(x: np.ndarray) -> np.ndarray:
    """A per-minute array (sessions x 390) in its typical size: divided by the root-mean-square of
    the same clock minute over the previous 20 sessions (earlier sessions only)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return x / lab._typical(np.asarray(x, float)[..., None])[..., 0]


def causal_z(x: np.ndarray, days: int = 60) -> np.ndarray:
    """A per-session series against the mean and standard deviation of its previous ``days``
    values (earlier sessions only)."""
    s = pd.Series(np.asarray(x, float))
    past = s.rolling(days, min_periods=days // 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((s - past.mean().shift(1)) / past.std().shift(1)).to_numpy()


def compute(new: dict, raw: dict) -> dict:
    """Every new input's values, shape-checked.

    Returns:
        ``{name: {"level", "values", "what", "why"}}``.
    """
    n = len(raw["dates"])
    out = {}
    for name, v in new.items():
        values = np.asarray(v["fn"](raw), float)
        want = (n,) if v["level"] == "day" else (n, lab.SESSION)
        if values.shape != want:
            raise ValueError(f"{name} returns shape {values.shape}; a {v['level']} input needs "
                             f"{want}")
        out[name] = {**v, "values": values}
    return out


def leak_check(fn, level: str, raw: dict, sessions) -> float:
    """The largest change in what the model would read when everything after the decision is
    hidden: for each of ``sessions``, the input is computed again from ``raw`` cut at that
    session's decision minute (its later minutes, its own last trade and every later session
    NaN) and compared with the full computation on that session (a day input: its value; a
    minute input: every minute before the decision). 0 = no look-ahead; a NaN on one side only
    counts as a difference of 1e9."""
    full = np.asarray(fn(raw), float)
    D = raw["decision"]
    worst = 0.0
    for s in sessions:
        cut = dict(raw)
        for k, v in raw.items():
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                v = v.copy()
                if v.ndim == 2:
                    v[s + 1:] = np.nan
                    v[s, D:] = np.nan
                else:
                    v[s:] = np.nan
                cut[k] = v
        part = np.asarray(fn(cut), float)
        a, b = (full[s], part[s]) if level == "day" else (full[s, :D], part[s, :D])
        worst = max(worst, float(np.max(np.abs(np.nan_to_num(a, nan=1e9)
                                               - np.nan_to_num(b, nan=1e9)))))
    return worst


def at_decision(values: np.ndarray, level: str, d: np.ndarray, decision: int) -> np.ndarray:
    """An input's value on sessions ``d`` as the model last sees it: a day input's value, a
    minute input's value in the last minute before the decision."""
    return values[d] if level == "day" else values[d, decision - 1]


# ---------------------------------------------------------------------------------------------
# tsmixer_lab with the new inputs.
# ---------------------------------------------------------------------------------------------
class Inputs:
    """``tsmixer_lab``'s inputs plus the new ones (:func:`compute`'s output), in the shape
    ``tsmixer_importance`` expects of ``lab`` (``FEATURES``, ``examples``, ``end_moves``), and
    able to train and predict with them. A day input becomes one more static column (clipped at
    ±``DAY_CLIP``), a minute input one more past covariate (clipped at ±``daystart.CLIP``); NaN
    is 0. Without new inputs in ``feats`` everything is exactly ``tsmixer_lab``'s."""

    def __init__(self, new: dict | None = None):
        self.new = {}
        for name, v in (new or {}).items():
            x = np.asarray(v["values"], float)
            day = v["level"] == "day"
            if x.ndim == (1 if day else 2):  # one stock: add its axis
                x = x[..., None]
            lim = lab.DAY_CLIP if day else ds.CLIP
            self.new[name] = {**v, "values": np.clip(np.nan_to_num(x), -lim, lim)}
        self.FEATURES = {**lab.FEATURES, **{
            n: {"level": v["level"], "width": 1, "source": "this notebook", "what": v["what"],
                "why": v["why"]} for n, v in self.new.items()}}

    def key(self, feats) -> tuple:
        """What identifies the inputs of ``feats``: their names and the new ones' values."""
        return tuple(sorted(feats)) + tuple(
            hashlib.sha1(self.new[f]["values"].tobytes()).hexdigest()[:12]
            for f in sorted(feats) if f in self.new)

    def tag(self, feats) -> str:
        """A short fingerprint of :meth:`key`: changes when a new input's values change."""
        return hashlib.sha1(repr(self.key(feats)).encode()).hexdigest()[:8]

    def end_moves(self, F: dict, cell, d, k) -> np.ndarray:
        """``tsmixer_lab.end_moves``."""
        return lab.end_moves(F, cell, d, k)

    def examples(self, F: dict, cell, days, feats, *, prev_days: int = 1, full: bool) -> tuple:
        """``tsmixer_lab.examples`` with the new inputs of ``feats`` added: minute inputs after
        the past covariates, day inputs after the static columns."""
        ser, cov, d, k = _BASE_EXAMPLES(F, cell, days, feats, prev_days=prev_days, full=full)
        bars = [f for f, v in self.new.items() if f in feats and v["level"] == "bar"]
        days_in = [f for f, v in self.new.items() if f in feats and v["level"] == "day"]
        if not ser or not (bars or days_in):
            return ser, cov, d, k
        from darts import TimeSeries

        _, off, L, H = cell
        n = L + H if full else L
        if days_in:
            first = sum(v["width"] for f, v in lab.FEATURES.items()
                        if f in feats and v["level"] == "day")
            vals = np.stack([self.new[f]["values"] for f in days_in], axis=-1)
            ser = [s.with_static_covariates(s.static_covariates.assign(**{
                f"day_{first + j}": vals[di, ki, j] for j in range(len(days_in))}))
                for s, di, ki in zip(ser, d, k, strict=True)]
        if bars:
            add = np.stack([self.new[f]["values"] for f in bars], axis=-1).astype(np.float32)
            idx = ser[0].time_index
            cov = [TimeSeries.from_times_and_values(
                idx, add[di, off:off + n, ki] if c is None else
                np.concatenate([c.values(), add[di, off:off + n, ki]], axis=1),
                columns=bars if c is None else [*c.columns, *bars])
                for c, di, ki in zip(cov or [None] * len(ser), d, k, strict=True)]
        return ser, cov, d, k

    @contextlib.contextmanager
    def _own_examples(self):
        """``tsmixer_lab.train`` / ``predict`` look ``examples`` up in their module at call
        time: point it here for the duration of one call, so the training is exactly theirs."""
        lab.examples = self.examples
        try:
            yield
        finally:
            lab.examples = _BASE_EXAMPLES

    def train(self, *args, **kwargs) -> tuple:
        """``tsmixer_lab.train`` with these inputs."""
        with self._own_examples():
            return lab.train(*args, **kwargs)

    def predict(self, *args, **kwargs) -> tuple:
        """``tsmixer_lab.predict`` with these inputs."""
        with self._own_examples():
            return lab.predict(*args, **kwargs)


# ---------------------------------------------------------------------------------------------
# Several seeds, compared.
# ---------------------------------------------------------------------------------------------
def fit_seeds(inputs: Inputs, data: dict, cell, feats, hp: dict, *, prev_days: int, seeds,
              epochs: int, patience: int, draws: int, cache: dict, on_fit=None) -> dict:
    """One configuration trained once per seed on TRAIN, scaled on the held-out days and scored
    on VALIDATION; each seed's fit is kept in ``cache`` (same inputs and values, configuration
    and seed: not trained again).

    Args:
        inputs: The inputs (:class:`Inputs`).
        data: :func:`load`'s output.
        cell: (1, offset, input, output) minutes.
        feats: The optional inputs read (``tsmixer_lab``'s and new ones).
        hp: TSMixer's knobs (as ``tsmixer_lab.train``'s ``hp``, without the patience).
        prev_days: Sessions behind the "prev" input.
        seeds: The random seeds.
        epochs: At most this many passes.
        patience: Early stopping's patience.
        draws: Random outcomes of the luck check.
        cache: Fits kept between calls.
        on_fit: Called after every seed.

    Returns:
        ``seeds`` (one row per seed: ``end_corr``, ``luck_z``, ``hit_rate``, ``beta``,
        ``passes``), ``pe`` (seeds x scored days: predicted end moves in units of the seed's
        own spread on the held-out days, so every seed weighs alike; β can be 0 and would drop a
        seed), ``ae``, ``d``, ``k``, ``ens_pe`` (the seeds' average) and ``ensemble`` (its
        ``tsmixer_lab.grade`` scores).
    """
    F = data["F"]
    kw = {"prev_days": prev_days}
    for seed in seeds:
        key = (inputs.key(feats), tuple(cell), prev_days, json.dumps(hp, sort_keys=True), epochs,
               patience, seed)
        if key not in cache:
            model, curve = inputs.train(F, cell, data["fit"], data["held"], feats,
                                        {**hp, "patience": patience}, epochs=epochs, seed=seed,
                                        **kw)
            ph, dh, kh = inputs.predict(model, F, cell, data["held"], feats, **kw)
            beta = ex.calibration(ph.sum(1)[:, None], lab.end_moves(F, cell, dh, kh)[:, None])
            raw, d, k = inputs.predict(model, F, cell, data["val"], feats, **kw)
            ae = lab.end_moves(F, cell, d, k)
            scores, _, _ = lab.grade(raw.sum(1), ae, d, k, F, beta=beta, draws=draws)
            spread = float(np.std(ph.sum(1))) or 1.0
            cache[key] = {"pe": raw.sum(1) / spread, "ae": ae, "d": d, "k": k, "row": {
                "seed": seed, **{s: scores[s] for s in ("end_corr", "luck_z", "hit_rate")},
                "beta": beta, "passes": len(curve.rows)}}
        if on_fit is not None:
            on_fit()
    runs = [cache[(inputs.key(feats), tuple(cell), prev_days, json.dumps(hp, sort_keys=True),
                   epochs, patience, s)] for s in seeds]
    first = runs[0]
    ens_pe = np.mean([r["pe"] for r in runs], axis=0)
    ensemble, _, _ = lab.grade(ens_pe, first["ae"], first["d"], first["k"], F, beta=1.0,
                               draws=draws)
    return {"seeds": pd.DataFrame([r["row"] for r in runs]),
            "pe": np.stack([r["pe"] for r in runs]),
            "ae": first["ae"], "d": first["d"], "k": first["k"], "ens_pe": ens_pe,
            "ensemble": ensemble}


def _corr(p: np.ndarray, a: np.ndarray) -> float:
    """Pearson correlation; 0 when either side does not move."""
    return float(np.corrcoef(p, a)[0, 1]) if len(p) > 2 and np.std(p) > 0 and np.std(a) > 0 \
        else 0.0


def compare(results: dict, base: str) -> pd.DataFrame:
    """One row per arm of :func:`fit_seeds` results: the seeds' mean end correlation and SD, the
    change against arm ``base`` paired by seed (mean, and t = mean / its standard error), the
    seed-averaged model's end correlation and luck z, and its change on the first and the second
    half of the scored days (a change that holds in both halves is less likely a fluke)."""
    b = results[base]
    sb = b["seeds"].set_index("seed")["end_corr"]
    first = b["d"] <= np.median(b["d"])
    rows = []
    for arm, r in results.items():
        if not np.array_equal(r["d"], b["d"]):
            raise ValueError(f"{arm} is scored on other days than {base}")
        s = r["seeds"].set_index("seed")["end_corr"]
        diff = (s - sb).dropna()
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
        rows.append({
            "arm": arm, "seeds": len(s), "mean end corr": s.mean(), "sd over seeds": s.std(ddof=1),
            "change (paired)": diff.mean(), "t of the change": diff.mean() / se if se else np.nan,
            "averaged model's end corr": r["ensemble"]["end_corr"],
            "its luck z": r["ensemble"]["luck_z"],
            "change, 1st half": _corr(r["ens_pe"][first], r["ae"][first])
            - _corr(b["ens_pe"][first], b["ae"][first]),
            "change, 2nd half": _corr(r["ens_pe"][~first], r["ae"][~first])
            - _corr(b["ens_pe"][~first], b["ae"][~first])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# What carries a score.
# ---------------------------------------------------------------------------------------------
def robustness(pe: np.ndarray, ae: np.ndarray, *, top: int = 5) -> pd.DataFrame:
    """How much of an end-move correlation a few days carry: the correlation itself, the rank
    (Spearman) correlation, the correlation without the 1 / 3 / ``top`` biggest actual moves,
    the share of the covariance from the ``top`` days that add the most, and the hit rate."""
    from scipy.stats import spearmanr

    big = np.argsort(-np.abs(ae))
    c = (pe - pe.mean()) * (ae - ae.mean())
    rows = [("end correlation (the score)", _corr(pe, ae)),
            ("rank correlation (Spearman: big days count like small ones)",
             float(spearmanr(pe, ae)[0]))]
    for n in sorted({1, 3, top}):
        keep = np.sort(big[n:])
        rows.append((f"end correlation without the {n} biggest move(s)", _corr(pe[keep], ae[keep])))
    rows += [(f"share of the score from the {top} days that add the most",
              float(np.sort(c)[::-1][:top].sum() / c.sum()) if c.sum() > 0 else np.nan),
             ("hit rate (up / down right)", float(np.mean(np.sign(pe) == np.sign(ae))))]
    return pd.DataFrame(rows, columns=["check", "value"])


def contributions(pe: np.ndarray, ae: np.ndarray) -> np.ndarray:
    """Each day's share of the end correlation's covariance (they sum to 1; negative: the day
    pulls the score down)."""
    c = (pe - pe.mean()) * (ae - ae.mean())
    return c / c.sum() if c.sum() != 0 else np.zeros_like(c)


# ---------------------------------------------------------------------------------------------
# The score in prices and dollars (pictures, not a backtest).
# ---------------------------------------------------------------------------------------------
def day_paths(F: dict, cell, d: np.ndarray, raw_paths: np.ndarray, beta: float) -> dict:
    """The price paths of scored days in % from the price at the decision (the last trade by
    then).

    Args:
        F: ``tsmixer_lab.build``'s inputs (one stock).
        cell: (1, offset, input, output) minutes.
        d: The sessions.
        raw_paths: The model's predicted minute moves (sessions x output minutes, typical
            moves), as ``tsmixer_lab.predict`` returns them.
        beta: The size scale from the held-out days.

    Returns:
        ``minutes`` (the actual path's minutes from the decision: 1 - input .. output),
        ``actual`` (sessions x those minutes, the last trade at each minute's end), ``ahead``
        (0 .. output), ``predicted`` (the predicted path × β, each minute's move times its
        typical size, sessions x ahead) and ``raw`` (the same without β).
    """
    _, off, L, H = cell
    D = off + L
    cf = F["pp"]["cf"][..., 0]
    u = F["u"][d, D:D + H, 0]

    def path(scale: float) -> np.ndarray:
        return 100 * np.c_[np.zeros(len(d)), np.exp(np.cumsum(scale * raw_paths * u, 1)) - 1]

    return {"minutes": np.arange(1 - L, H + 1),
            "actual": 100 * (cf[d, off:D + H] / cf[d, D - 1][:, None] - 1),
            "ahead": np.arange(H + 1), "predicted": path(beta), "raw": path(1.0)}


def real_moves(F: dict, cell, d: np.ndarray) -> np.ndarray:
    """The stock's actual move from the decision to the horizon's end on sessions ``d``, as a
    fraction: the last trade by the end ÷ the last trade by the decision − 1."""
    _, off, L, H = cell
    cf = F["pp"]["cf"][..., 0]
    return cf[d, off + L + H - 1] / cf[d, off + L - 1] - 1


def traders(pe: np.ndarray, moves: np.ndarray, *, notional: float, cost: float) -> pd.DataFrame:
    """Two traders acting on every prediction: at the decision they buy (or sell short) at the
    last trade, at the horizon's end they close. "sized" holds ``notional`` × the prediction ÷
    its average size on these days (capped at 3×; a scale read on the same days: fine for a
    picture, not for a backtest), "direction" ±``notional`` by the prediction's sign.

    Args:
        pe: Predicted end moves, one per day.
        moves: Actual moves over the horizon (fractions), one per day.
        notional: The typical position in dollars.
        cost: A round trip's cost as a fraction of the position.

    Returns:
        Per day: ``<trader> position`` ($), ``<trader> gross`` and ``<trader> net`` P&L ($).
    """
    size = pe / np.mean(np.abs(pe)) if np.any(pe) else np.zeros_like(pe)
    out = {}
    for who, w in {"sized": notional * np.clip(size, -3, 3),
                   "direction": notional * np.sign(pe)}.items():
        out |= {f"{who} position": w, f"{who} gross": w * moves,
                f"{who} net": w * moves - cost * np.abs(w)}
    return pd.DataFrame(out)


def log(stock: str, table: pd.DataFrame, **context) -> pd.DataFrame:
    """Append ``table``'s rows (with ``context`` columns and the time) to the stock's experiment
    log, ``var/tsmixer_lab/stock/<stock>-experiments.csv``.

    Returns:
        The whole log, newest first.
    """
    path = LOG_DIR / f"{stock}-experiments.csv"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = table.assign(time=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                        **{k: json.dumps(v) if not isinstance(v, str) else v
                           for k, v in context.items()})
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)
    return read_log(stock)


def read_log(stock: str) -> pd.DataFrame:
    """The stock's experiment log (:func:`log`), newest first; empty without one."""
    path = LOG_DIR / f"{stock}-experiments.csv"
    return pd.read_csv(path).iloc[::-1].reset_index(drop=True) if path.exists() \
        else pd.DataFrame()
