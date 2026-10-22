"""Movers (``darts_forecast_movers_marimo.py``): can the move of a stock over a fixed afternoon
window be predicted well enough to pick the day's best calls?

One example is one STOCK-DAY: the minutes before the decision (13:30) are the input, the move from
the fill (the first trade after the decision) to the horizon's end (15:30) is the outcome, in
PERCENT, labelled **down / flat / up** (flat: under ``theta`` percent either way). One TSMixer
reads the input and outputs the three class probabilities; every stock-day counts once in its
training loss, whatever the size of the move.

What counts is the CALL, not the stock-day: with ``CAPITAL`` and ``POSITION`` the book holds
``K`` positions, so each day the model's ``K`` most likely movers are called, each long (if "up"
is likelier than "down") or short. :func:`ladder` scores those calls, every call one vote, against
rules that need no model: random picks, and the ``K`` stocks with the biggest trailing average
move (a stock's activity is cheap to know and triples the hit rate), called on the PRIOR side
(:func:`prior_side`: the side the training days' moves leaned to, which is what a model that knows
nothing about the direction calls; in a rising market it beats a coin without knowing anything).

It builds on ``tsmixer_lab`` (passed in as ``lab``: its minute arrays, inputs and Darts
plumbing) and edits nothing there: that file's source is part of every TSMixer study's name.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VAR = HERE.parents[1] / "var" / "movers"      # the universe screen and every trial's probabilities
CLASSES = ("down", "flat", "up")              # class 0, 1, 2
ACT_DAYS = 20                                 # sessions behind a stock's trailing activity
_CE: list = []                                # the loss class, built on first use (torch is slow)


# ---------------------------------------------------------------------------------------------
# The universe: stocks with 1-minute bars since the start, screened on TRAIN only.
# ---------------------------------------------------------------------------------------------
def universe(engine, *, start: str, train_end: str, decision: int, horizon: int, theta: float,
             n: int, min_share: float = 0.97, min_dollars: float = 1e6,
             refresh: bool = False) -> pd.DataFrame:
    """The ``n`` most active liquid stocks that have 1-minute bars since ``start``.

    Only the TRAIN window is read (activity measured on VALIDATION or TEST days would pick the
    stocks that happened to move there): per stock the share of its sessions with a move of at
    least ``theta`` percent between the decision and the horizon's end, its median move, price and
    median dollars traded in that window. Stocks with fewer than ``min_share`` of the longest
    history, or under ``min_dollars`` traded in the window, are dropped. Saved in :data:`VAR`.

    Args:
        engine: Database engine.
        start: The first day every stock must have 1-minute bars (the first two weeks are checked).
        train_end: The last day read.
        decision: Minutes after 09:30 of the decision.
        horizon: Minutes from the decision to the end of the window.
        theta: The size of a "move", in percent.
        n: How many stocks to keep.
        min_share: Fewest sessions, as a share of the longest history.
        min_dollars: Fewest dollars traded in the window on a median day.
        refresh: Query again.

    Returns:
        One row per stock, most active first: ``days``, ``big`` (share of sessions with a move
        of at least ``theta``), ``median_move`` (percent), ``price``, ``dollars_m`` (millions).
    """
    from sqlalchemy import text

    key = hashlib.sha1(json.dumps([start, train_end, decision, horizon, theta, n, min_share,
                                   min_dollars]).encode()).hexdigest()[:10]
    path = VAR / f"universe-{key}.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    t0, t1 = _clock(decision), _clock(decision + horizon - 1)
    ny = "ts AT TIME ZONE 'America/New_York'"
    first = (pd.Timestamp(start) + pd.Timedelta(days=14)).date().isoformat()
    with engine.connect() as c:
        early = pd.read_sql(text("SELECT DISTINCT symbol FROM prices_ohlcv WHERE bar = '1m' "
                                 "AND ts >= :a AND ts < :b"), c, params={"a": start, "b": first})
        rows = pd.read_sql(text(
            f"SELECT symbol, ({ny})::date AS d, "
            f"max(open) FILTER (WHERE ({ny})::time = CAST(:t0 AS time)) AS o, "
            f"max(close) FILTER (WHERE ({ny})::time = CAST(:t1 AS time)) AS c, "
            "sum(volume * close) AS dollars FROM prices_ohlcv WHERE bar = '1m' AND rth "
            "AND symbol = ANY(CAST(:s AS text[])) AND ts >= :a AND ts < :b "
            f"AND ({ny})::time BETWEEN CAST(:t0 AS time) AND CAST(:t1 AS time) "
            "GROUP BY symbol, d"), c, params={
                "s": sorted(early.symbol), "a": start, "t0": t0, "t1": t1,
                "b": (pd.Timestamp(train_end) + pd.Timedelta(days=1)).date().isoformat()})
    rows = rows.dropna(subset=["o", "c"])
    rows["a"] = (rows["c"] / rows["o"] - 1).abs() * 100
    g = rows.groupby("symbol")
    out = pd.DataFrame({"days": g.size(), "big": g["a"].apply(lambda x: float((x >= theta).mean())),
                        "median_move": g["a"].median(), "price": g["o"].median(),
                        "dollars_m": g["dollars"].median() / 1e6})
    enough = out["days"] >= min_share * out["days"].max()
    out = out[enough & (out["dollars_m"] >= min_dollars / 1e6)]
    out = out.sort_values("big", ascending=False).head(n)
    VAR.mkdir(parents=True, exist_ok=True)
    out.to_pickle(path)
    return out


def _clock(minutes: int) -> str:
    """The New York clock time ``minutes`` after the 09:30 open, "HH:MM"."""
    return f"{(570 + minutes) // 60:02d}:{(570 + minutes) % 60:02d}"


# ---------------------------------------------------------------------------------------------
# The outcome and the activity.
# ---------------------------------------------------------------------------------------------
def with_opens(F: dict, opens: pd.DataFrame) -> dict:
    """``F`` with ``open``: the first trade of every minute (sessions x 390 x stocks, NaN for a
    minute without a trade: IBKR's volume-0 placeholders copy the last close), which
    :func:`moves` fills at. ``lab.build`` keeps no open prices, and ``tsmixer_lab.py`` must not
    change (its source is in every TSMixer study's name).

    Args:
        F: ``lab.build``'s inputs.
        opens: The panel's opens on the full clock (``daystart.full_clock(panel).open_``).

    Returns:
        ``F`` itself, with ``open`` added.
    """
    vol = F["pp"]["volume"]
    op = np.ascontiguousarray(opens[F["names"]].to_numpy(float)).reshape(vol.shape)
    F["open"] = np.where(vol > 0, op, np.nan)
    return F


def moves(F: dict, cell) -> np.ndarray:
    """The move from the FILL (the open of the first traded minute from the decision on, inside
    the horizon) to the last trade of the horizon, in percent (sessions x stocks); NaN on a
    half-day or without a trade in the horizon. The decision minute's last trade is not the
    entry: it sits on the bid or the ask, and a call cannot trade at it.

    Args:
        F: ``lab.build``'s inputs, with :func:`with_opens`'s ``open``.
        cell: (1, offset, input, output) minutes.

    Returns:
        The percent moves.
    """
    _, off, L, H = cell
    D = off + L
    traded = np.isfinite(F["open"][:, D:D + H, :])
    first = D + traded.argmax(axis=1)
    fill = np.take_along_axis(F["open"], first[:, None, :], axis=1)[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        mv = (F["pp"]["cf"][:, D + H - 1, :] / np.where(traded.any(axis=1), fill, np.nan) - 1) * 100
    return np.where(F["pp"]["usable"][:, None], mv, np.nan)


def labels(mv: np.ndarray, theta: float) -> np.ndarray:
    """The class of every move: 0 down (at most ``-theta`` percent), 2 up (at least ``theta``),
    else 1 flat; -1 where the move is unknown."""
    cls = np.where(mv <= -theta, 0, np.where(mv >= theta, 2, 1))
    return np.where(np.isfinite(mv), cls, -1)


def activity(F: dict, cell, theta: float) -> dict:
    """What each stock's previous :data:`ACT_DAYS` sessions say about its move in this window:
    ``mean`` (average absolute move, percent), ``big`` (share of sessions with a move of at least
    ``theta``) and ``log_mean`` (the mean on a log scale around 0.5 percent, clipped to ±3). The
    window is the same clock window every session; the session itself is never included.

    Args:
        F: ``lab.build``'s inputs.
        cell: (1, offset, input, output) minutes.
        theta: The size of a "move", in percent.

    Returns:
        ``{name: sessions x stocks}``, NaN until half the days exist.
    """
    a = pd.DataFrame(np.abs(moves(F, cell)))
    roll = lambda x: x.rolling(ACT_DAYS, min_periods=ACT_DAYS // 2).mean().shift(1).to_numpy()  # noqa: E731
    mean = roll(a)
    big = roll((a >= theta).astype(float).where(a.notna()))
    with np.errstate(invalid="ignore", divide="ignore"):
        log_mean = np.clip(np.log(np.maximum(mean, 0.05) / 0.5), -3, 3)
    return {"mean": mean, "big": big, "log_mean": np.where(np.isfinite(mean), log_mean, np.nan)}


def targets(F: dict, cell, d: np.ndarray, k: np.ndarray, theta: float) -> tuple:
    """The class and the trailing activity of the stock-days ``(d, k)``."""
    return labels(moves(F, cell), theta)[d, k], activity(F, cell, theta)["mean"][d, k]


def prior_side(F: dict, cell, days, theta: float) -> int:
    """The side a model that knows nothing about the direction calls: 2 (long) if at least as
    many stock-days of ``days`` (the days it learns from) moved up as down, else 0 (short).

    Calls are graded against this side, not against a coin: in a rising market "always long" is
    right on more than half of the moves (55% in H1 2026) without knowing anything.
    """
    cls = labels(moves(F, cell), theta)[np.asarray(days, int)]
    return 2 if (cls == 2).sum() >= (cls == 0).sum() else 0


# ---------------------------------------------------------------------------------------------
# The examples, training and prediction.
# ---------------------------------------------------------------------------------------------
def examples(lab, F: dict, cell, days, feats, *, prev_days: int = 1, stock_id: bool = True,
             theta: float = 1.0, full: bool) -> tuple:
    """The Darts series of every valid stock-day of ``days``. The TARGET carries only the label:
    three columns (down, flat, up), zero except a one-hot class on the last step (``full``: the
    input and the output, exactly L + H minutes; else the input only). The path and the BAR
    inputs of ``feats`` are past covariates over the input minutes (and ONLY those). The statics:
    the stock's one-hot (``stock_id``), the DAY inputs of ``feats`` and the trailing activity.

    Returns:
        ``(series, covariates, day per series, stock per series)``.
    """
    from darts import TimeSeries

    _, off, L, H = cell
    cls = labels(moves(F, cell), theta)
    act = activity(F, cell, theta)
    day = np.concatenate([lab.day_inputs(F, cell, feats, prev_days),
                          np.nan_to_num(act["log_mean"])[..., None],
                          np.nan_to_num(act["big"])[..., None]], axis=-1)
    bars = ["path", *[f for f, v in lab.FEATURES.items()
                      if f in feats and v["level"] == "bar" and f != "path"]]
    ok = lab.valid(F, cell) & (cls >= 0) & np.isfinite(act["log_mean"])
    names = F["names"]
    eye = np.eye(len(names)) if stock_id else None
    cols = [f"is_{x}" for x in names] if stock_id else []
    cols += [f"day_{i}" for i in range(day.shape[-1])]
    steps = L + H if full else L
    idx, cov_idx = pd.RangeIndex(steps), pd.RangeIndex(L)
    ser, covs, dd, kk = [], [], [], []
    for d in days:
        for k in range(len(names)):
            if not ok[d, k]:
                continue
            y = np.zeros((steps, 3), np.float32)
            if full:
                y[-1, cls[d, k]] = 1.0
            static = pd.DataFrame([np.r_[eye[k] if stock_id else [], day[d, k]]], columns=cols)
            ser.append(TimeSeries.from_times_and_values(idx, y, columns=list(CLASSES))
                       .with_static_covariates(static))
            path = np.stack([F["bar"][f][d, off:off + L, k] for f in bars], axis=-1)
            covs.append(TimeSeries.from_times_and_values(
                cov_idx, np.nan_to_num(path).astype(np.float32), columns=bars))
            dd.append(d)
            kk.append(k)
    return ser, covs, np.array(dd, int), np.array(kk, int)


def _ce():
    """The class-weighted cross-entropy (built on first use: torch imports slowly). Darts calls it
    with (batch, 1, 3) logits and one-hot labels; every stock-day weighs the weight of its class
    (1, ``flat_weight``, 1), so a big move counts exactly as much as a small one."""
    if not _CE:
        import torch

        class WeightedCE(torch.nn.Module):
            def __init__(self, flat_weight: float = 1.0):
                super().__init__()
                self.reduction, self.flat_weight = "mean", float(flat_weight)

            def forward(self, pred, target):
                w = pred.new_tensor([1.0, self.flat_weight, 1.0])
                loss = -(target * torch.log_softmax(pred, dim=-1)).sum(-1) * (target * w).sum(-1)
                return loss.mean() if self.reduction == "mean" else loss

        _CE.append(WeightedCE)
    return _CE[0]


def train(lab, F: dict, cell, fit_days, held_days, feats, hp: dict, *, prev_days: int = 1,
          stock_id: bool = True, theta: float = 1.0, epochs: int = 60, seed: int = 0) -> tuple:
    """ONE TSMixer for every stock, fit on the valid stock-days of ``fit_days``; ``held_days``
    stop the training early and the best pass's weights are kept. The model predicts one step, the
    horizon's last minute (``output_chunk_shift``), as three logits.

    Args:
        lab: The ``tsmixer_lab`` module.
        F: ``lab.build``'s inputs.
        cell: (1, offset, input, output) minutes.
        fit_days: Session indices to learn on.
        held_days: Session indices held out (early stopping).
        feats: The optional inputs read.
        hp: TSMixer's knobs on top of ``explorer.make_path_model``'s, plus "lr", "weight_decay"
            (AdamW), "flat_weight" (the loss weight of the flat class) and "patience".
        prev_days: Sessions behind the "prev" input.
        stock_id: Give the model the stock's one-hot.
        theta: The size of a "move", in percent.
        epochs: At most this many passes.
        seed: Random seed.

    Returns:
        ``(model, info)``; ``info``: ``val_loss`` (kept), ``prior_loss`` (the same loss of the
        best constant: the fit days' class shares times the class weights, normalized; the model
        must beat it), ``epochs``.
    """
    import torch

    _, _, L, H = cell
    kw = dict(prev_days=prev_days, stock_id=stock_id, theta=theta, full=True)
    ser, cov, _, _ = examples(lab, F, cell, fit_days, feats, **kw)
    vser, vcov, _, _ = examples(lab, F, cell, held_days, feats, **kw)
    if not ser or not vser:
        raise ValueError("no valid stock-day to learn on or to hold out")
    base = lab.ex.make_path_model("tsmixer", L, 1, epochs=epochs, n_samples=len(ser), seed=seed)
    hp = dict(hp)
    optim = {k: hp.pop(k) for k in ("lr", "weight_decay") if k in hp}
    hp["optimizer_kwargs"] = {"lr": lab.ex.SETTINGS["lr"], **optim}
    hp["optimizer_cls"] = torch.optim.AdamW
    flat_weight, patience = hp.pop("flat_weight", 1.0), hp.pop("patience", None)
    hp["loss_fn"] = _ce()(flat_weight)
    hp["output_chunk_shift"] = H - 1
    model = type(base)(**{**base.model_params, **hp})
    for c in model.trainer_params["callbacks"]:  # the early stopping's patience
        if patience is not None and hasattr(c, "patience"):
            c.patience = patience
    model.fit(ser, past_covariates=cov, val_series=vser, val_past_covariates=vcov, verbose=False)
    lab.ex.SM["_quiet"]()
    keep = next(c for c in model.trainer_params["callbacks"] if hasattr(c, "state"))
    if keep.state is not None:  # back to the weights of the best pass
        model.model.load_state_dict(keep.state)
    w = np.array([1.0, flat_weight, 1.0])
    share = np.mean([s.values()[-1] for s in ser], axis=0) * w  # the best constant under the
    share /= share.sum()                                         # weighted loss: share x weight
    held = np.stack([s.values()[-1] for s in vser])
    prior = float(np.mean(-(held * np.log(np.maximum(share, 1e-9))).sum(1) * (held * w).sum(1)))
    return model, {"val_loss": float(keep.best), "prior_loss": prior,
                   "epochs": int(getattr(model, "epochs_trained", epochs))}


def predict_proba(lab, model, F: dict, cell, days, feats, *, prev_days: int = 1,
                  stock_id: bool = True, theta: float = 1.0) -> tuple:
    """The model's class probabilities of every valid stock-day of ``days``, read from that day's
    input minutes only.

    Returns:
        ``(probabilities (examples x 3: down, flat, up), day, stock)``.
    """
    ser, cov, d, k = examples(lab, F, cell, days, feats, prev_days=prev_days, stock_id=stock_id,
                              theta=theta, full=False)
    if not ser:
        return np.zeros((0, 3), np.float32), d, k
    out = model.predict(n=1, series=ser, past_covariates=cov, batch_size=1024, verbose=False)
    z = np.stack([o.values()[0] for o in out]).astype(np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32), d, k


# ---------------------------------------------------------------------------------------------
# The calls and the ladder.
# ---------------------------------------------------------------------------------------------
def picks(score: np.ndarray, d: np.ndarray, K: int, tie: np.ndarray | None = None) -> np.ndarray:
    """The indices of the ``K`` highest ``score`` of every day (all of them on a day with fewer).
    Equal scores are ordered by ``tie`` (higher first), then by position: without a ``tie`` a
    model whose few distinct values tie many stocks would pick by the stocks' order."""
    keys = [-np.asarray(score, float), np.asarray(d)]
    if tie is not None:
        keys.insert(0, -np.nan_to_num(np.asarray(tie, float), nan=-np.inf))
    order = np.lexsort(keys)
    ds_ = np.asarray(d)[order]
    start = np.r_[0, np.flatnonzero(np.diff(ds_)) + 1]
    rank = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
    return order[rank < K]


def calls(p: np.ndarray, d: np.ndarray, act: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """A model's calls: every day its ``K`` stock-days with the biggest ``p(up) + p(down)`` (equal
    ones: the more active stock first, so a tie falls back on the activity rule), each long (2) if
    ``p(up) >= p(down)``, else short (0).

    Returns:
        ``(indices of the calls, their sides)``.
    """
    p = np.asarray(p, float)
    p_move = np.round(p[:, 0] + p[:, 2], 6)  # float noise must not break a tie
    mp = picks(p_move, d, K, tie=act)
    return mp, np.where(p[mp, 2] >= p[mp, 0], 2, 0)


def _ap_by_day(y: np.ndarray, score: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    """The average precision of ``score`` for ``y`` inside each day, and the share of ``y`` (what
    a blind score gets), both averaged over the days that hold both classes."""
    from sklearn.metrics import average_precision_score

    rows = [(average_precision_score(y[m], score[m]), y[m].mean())
            for m in (d == x for x in np.unique(d)) if 0 < y[m].sum() < m.sum()]
    return tuple(float(v) for v in np.mean(rows, axis=0)) if rows else (np.nan, np.nan)


def _z(x: pd.Series) -> tuple[float, float, float]:
    """The mean of the per-day values, its standard error over days and the ratio (a day is one
    observation: the picks of one day share the market's mood)."""
    x = x.dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float(x.mean()) if len(x) else np.nan, np.nan, 0.0
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    return float(x.mean()), se, float(x.mean() / se)


def ladder(cls: np.ndarray, d: np.ndarray, act: np.ndarray, p: np.ndarray | None = None, *,
           K: int = 5, prior: int) -> dict:
    """The calls' scores, every call one vote, against the rules that need no model.

    A day's calls are :func:`calls`. A call is RIGHT when the stock moved at least ``theta`` the
    called way (class equals the side). The rules pick at random or by ``act`` (the stock's
    trailing average move: the NAIVE rule), with a random side (right on half of the picks that
    moved) or, for the naive picks, on the ``prior`` side (what a model that learned nothing about
    the direction calls). THE score is the call precision above the naive picks on the prior side:
    above the random side, a model would score for leaning the way the market went. Everything is
    the average over days of a day's average over its calls; the z of a difference is over days,
    paired (a day's calls share the market's mood).

    Args:
        cls: Class of every stock-day (0 down, 1 flat, 2 up).
        d: Its day.
        act: Its trailing activity (higher: expected to move more).
        p: Its class probabilities (examples x 3), or None for the rules alone.
        K: Calls per day.
        prior: The prior side (0 short, 2 long): :func:`prior_side` of the days the model learns
            from, never of the graded ones.

    Returns:
        ``days``; ``move_random`` / ``move_naive`` (share of picks that moved), ``call_random`` /
        ``call_naive`` (share of calls right, random side), ``call_prior`` (the naive picks on the
        prior side); with ``p``: ``calls``, ``move_rate``, ``call_prec`` and its ``call_se``,
        ``excess_prior`` / ``excess_z_prior`` (THE score) and ``excess_z_random`` (above random
        picks), ``move_z_naive`` / ``move_z_random`` (the picking alone), ``dir_acc`` (right among
        the picks that moved), ``dir_prior`` (the prior side's share on the same picks) and
        ``side_z`` (the model's sides against the prior side on its own picks: the direction
        alone), and ``ap`` / ``ap_naive`` / ``ap_base`` (average precision of "will it move"
        WITHIN each day, averaged over days: the model, the activity rule, chance. Pooled over all
        stock-days it would also credit knowing which days are busy, which K calls a day cannot
        use).
    """
    cls, d, act = np.asarray(cls), np.asarray(d), np.asarray(act, float)
    days = np.unique(d)
    moved = (cls != 1).astype(float)

    def per_day(idx, x) -> pd.Series:
        return pd.Series(np.asarray(x, float)).groupby(d[idx]).mean().reindex(days)

    rnd = per_day(np.arange(len(cls)), moved)
    nv = picks(np.nan_to_num(act, nan=-1.0), d, K)
    naive, naive_prior = per_day(nv, moved[nv]), per_day(nv, cls[nv] == prior)
    out = {"days": len(days), "move_random": float(rnd.mean()), "move_naive": float(naive.mean()),
           "call_random": float(rnd.mean() / 2), "call_naive": float(naive.mean() / 2),
           "call_prior": float(naive_prior.mean())}
    if p is None:
        return out
    mp, side = calls(p, d, act, K)
    hit, hit_prior = (cls[mp] == side).astype(float), (cls[mp] == prior).astype(float)
    call_d, move_d = per_day(mp, hit), per_day(mp, moved[mp])
    n_moved = moved[mp].sum()
    mean, _, z = _z(call_d - naive_prior)
    out |= {"calls": len(mp), "move_rate": float(move_d.mean()), "call_prec": float(call_d.mean()),
            "call_se": _z(call_d)[1], "excess_prior": mean, "excess_z_prior": z,
            "excess_z_random": _z(call_d - rnd / 2)[2], "move_z_naive": _z(move_d - naive)[2],
            "move_z_random": _z(move_d - rnd)[2],
            "dir_acc": float(hit.sum() / n_moved) if n_moved else np.nan,
            "dir_prior": float(hit_prior.sum() / n_moved) if n_moved else np.nan,
            "side_z": _z(call_d - per_day(mp, hit_prior))[2]}
    p_move = np.asarray(p, float)[:, [0, 2]].sum(axis=1)
    out["ap"], out["ap_base"] = _ap_by_day(moved, p_move, d)
    out["ap_naive"] = _ap_by_day(moved, np.nan_to_num(act), d)[0]
    return out
