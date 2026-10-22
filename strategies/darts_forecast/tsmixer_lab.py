"""TSMixer deep-dive helpers (``darts_forecast_tsmixer_marimo.py``): the ten inputs a model may
read, the market and sector baskets two of them need, the Darts examples of every (day, stock),
TSMixer training and the end-move grade with its luck check.

One example is one STOCK-DAY at 1-minute bars: the minutes from 09:30 + offset to the decision
are the input, the minutes from the decision to the end of the horizon the output. The model
reads the stock's own path plus any of the BAR inputs (one value per input minute: past
covariates) and DAY inputs (one value per stock-day, known at the decision: static inputs, beside
the stock's one-hot). Every input is in the stock's own typical units, measured on EARLIER
sessions only, so calm and wild stocks, the open and midday mix in one model.

It builds on ``daystart.py`` (the minute arrays, bars and typical moves; loaded once as :data:`ds`,
with ``ds.ex`` / ``ds.rs``) and edits nothing there.
"""

from __future__ import annotations

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


ds = _sibling("daystart")
ex, rs = ds.ex, ds.rs
SESSION = ds.SESSION
BASKET_DIR = HERE.parents[1] / "var" / "tsmixer_lab"  # the basket aggregates (git-ignored)
SAME_HOUR_DAYS = 10    # sessions behind the "same hour" input
OPTION_Z_DAYS = 60     # sessions behind each option signal's own scale
OPTION_SIGNALS = ("iv_hv", "skew", "cust_pc")
DAY_CLIP = 4.0         # a day input beyond 4 of its typical units counts as 4

#: The inputs, in their fixed column order: level ("bar": one value per input minute, a past
#: covariate; "day": one value per stock-day, a static input), what it is, why it may carry
#: information (the literature), where it comes from, and its number of columns.
FEATURES = {
    "path": {"level": "bar", "width": 1, "source": "1-minute prices",
             "what": "the stock's own 1-minute moves before the decision, in typical moves "
                     "(the target's own history; always read)",
             "why": "short-lag momentum and reversal; the bid-ask bounce"},
    "volume": {"level": "bar", "width": 1, "source": "1-minute prices",
               "what": "each minute's volume against the stock's usual minute (log)",
               "why": "Gervais, Kaniel & Mingelgrin (2001): unusual volume carries information"},
    "range": {"level": "bar", "width": 1, "source": "1-minute prices",
              "what": "each minute's high-low range against the usual range at that time",
              "why": "volatility clustering; Gao et al. (2018): intraday momentum is stronger "
                     "on volatile days"},
    "flow": {"level": "bar", "width": 1, "source": "1-minute prices",
             "what": "order-flow proxy: where each minute closed inside its high-low range "
                     "(+1 at the high, -1 at the low)",
             "why": "Chordia & Subrahmanyam (2004): order imbalance predicts short-horizon "
                    "returns"},
    "market": {"level": "bar", "width": 1, "source": "1-minute prices (basket)",
               "what": "the market's 1-minute moves: an equal-weight basket of the stocks with "
                       "full 1-minute data, in typical moves",
               "why": "Gao, Han, Li & Zhou (2018): market intraday momentum; a stock's beta"},
    "sector": {"level": "bar", "width": 1, "source": "1-minute prices (basket)",
               "what": "the stock's sector's 1-minute moves: the basket's same-sector stocks, "
                       "the stock itself left out",
               "why": "Hou (2007): industry information diffuses with a lag"},
    "gap": {"level": "day", "width": 1, "source": "1-minute prices",
            "what": "today's first trade against yesterday's last, in typical nights (0 until "
                    "the stock has traded)",
            "why": "Lou, Polk & Skouras (2019): overnight and intraday returns pull apart"},
    "prev": {"level": "day", "width": 1, "source": "1-minute prices",
             "what": "the last `prev_days` sessions' close-to-close move, in typical moves",
             "why": "Jegadeesh (1990), Lehmann (1990): short-term reversal; COLB's 2-day "
                    "reversal in the earlier studies"},
    "same_hour": {"level": "day", "width": 1, "source": "1-minute prices",
                  "what": "the stock's move over the same clock window (decision to end of "
                          f"horizon) on the previous {SAME_HOUR_DAYS} sessions, averaged",
                  "why": "Heston, Korajczyk & Sadka (2010): returns persist at the same time of "
                         "day for weeks"},
    "options": {"level": "day", "width": len(OPTION_SIGNALS), "source": "option files (5 stocks)",
                "what": "from the previous close: implied minus historical volatility, put "
                        "skew, customer put/call volume (each against its own last "
                        f"{OPTION_Z_DAYS} sessions)",
                "why": "Bali & Hovakimian (2009), Xing, Zhang & Zhao (2010), Pan & Poteshman "
                       "(2006): option prices and volume lead the stock"},
}
OPTIONAL = [f for f in FEATURES if f != "path"]  # the inputs that can be switched off


# ---------------------------------------------------------------------------------------------
# The market and sector baskets.
# ---------------------------------------------------------------------------------------------
def basket(engine, start: str, end: str, *, min_share: float = 0.95,
           refresh: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """The market and sector baskets: per regular-session minute and sector, the sum and the
    count of the 1-minute moves (log close / open of minutes with a trade) of every stock with a
    12:00 bar on at least ``min_share`` of the sessions from ``start`` to ``end``. Two SQL
    aggregations (about a minute each), saved in :data:`BASKET_DIR`.

    Args:
        engine: Database engine.
        start: First day.
        end: Last day.
        min_share: The share of sessions a basket stock must have.
        refresh: Query again.

    Returns:
        ``(moves, sectors)``: ``moves`` indexed by (ts, sector) with ``sum`` and ``n``; the
        sector of every basket stock.
    """
    from sqlalchemy import text

    key = hashlib.sha1(json.dumps([start, end, min_share]).encode()).hexdigest()[:10]
    path = BASKET_DIR / f"basket-{start}-{end}-{key}.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    window = {"a": start, "b": str((pd.Timestamp(end) + pd.Timedelta(days=1)).date())}
    with engine.connect() as c:
        days = pd.read_sql(text(
            "SELECT symbol, count(*) AS days FROM prices_ohlcv WHERE bar = '1m' AND rth "
            "AND ts >= :a AND ts < :b AND (ts AT TIME ZONE 'America/New_York')::time = '12:00' "
            "GROUP BY symbol"), c, params=window)
    members = sorted(days.loc[days["days"] >= min_share * days["days"].max(), "symbol"])
    sectors = ex.listing(engine, members)["sector"].reindex(members).fillna("other")
    sectors = sectors.replace("", "other")
    pairs = ", ".join(f"('{s}', '{sectors[s]}')" for s in members if "'" not in sectors[s])
    with engine.connect() as c:
        moves = pd.read_sql(text(
            f"SELECT p.ts, m.sector, sum(ln(p.close / p.open)) AS sum, count(*) AS n "
            f"FROM prices_ohlcv p JOIN (VALUES {pairs}) AS m(symbol, sector) USING (symbol) "
            "WHERE p.bar = '1m' AND p.rth AND p.ts >= :a AND p.ts < :b AND p.volume > 0 "
            "AND p.open > 0 AND p.close > 0 GROUP BY p.ts, m.sector"), c, params=window)
    moves["ts"] = pd.to_datetime(moves["ts"], utc=True)
    out = (moves.set_index(["ts", "sector"]).sort_index(), sectors)
    BASKET_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, path)
    return out


# ---------------------------------------------------------------------------------------------
# The inputs.
# ---------------------------------------------------------------------------------------------
def _typical(x: np.ndarray, floor: float = 0.25) -> np.ndarray:
    """Per clock minute and stock, the root-mean-square of ``x`` (sessions x 390 x stocks) over
    the previous :data:`ds.VOL_DAYS` sessions (earlier sessions only), floored at ``floor`` x the
    stock's average over those sessions."""
    import warnings

    n, m, k = x.shape
    lo = ds.VOL_DAYS // 2
    ms = pd.DataFrame((x ** 2).reshape(n, m * k)).rolling(ds.VOL_DAYS, min_periods=lo).mean()
    ms = ms.shift(1).to_numpy().reshape(n, m, k)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.fmax(np.sqrt(ms), floor * np.sqrt(np.nanmean(ms, axis=1, keepdims=True)))


def _basket_bars(pp: dict, r1: np.ndarray, tag: str) -> np.ndarray:
    """1-minute basket moves (sessions x 390 x columns) in their own typical moves, through
    ``daystart.bars`` (the same rolling scale and clip as the stock's path)."""
    like = {"key": (pp["key"], tag), "r1": r1, "regular": pp["regular"],
            "volume": np.ones_like(r1), "cf": np.ones_like(r1)}
    return ds.bars(like, 1)["y"]


def build(panel, engine, *, basket_from: str, basket_to: str, options: bool = False) -> dict:
    """Every input of every stock-minute and stock-day the panel allows, plus the target.

    Args:
        panel: The 1-minute panel (dividend-adjusted, ``research.load``).
        engine: Database engine (the baskets).
        basket_from: The baskets' first day.
        basket_to: The baskets' last day.
        options: Load the option signals (slow: hundreds of daily files).

    Returns:
        ``pp`` (``daystart.prepared``), ``names``, ``dates``, ``y`` / ``u`` (the target and its
        typical move), ``bar`` (input -> sessions x 390 x stocks), ``gap``, ``prev_move``
        (``{days: sessions x stocks}``), ``options`` (sessions x stocks x 3, or None),
        ``sectors`` (the stocks' sectors), ``in_basket``, ``basket_size`` and ``available``
        (stock x input: the share of the stock's sessions with data on which the input has a
        value).
    """
    pp = ds.prepared(panel)
    p = ds.full_clock(panel)
    n, N = len(pp["dates"]), len(pp["names"])
    shape = (n, SESSION, N)
    arr = lambda k: np.ascontiguousarray(getattr(p, k).to_numpy(float)).reshape(shape)  # noqa: E731
    hi, lo, op, cl = arr("high"), arr("low"), arr("open_"), arr("close")
    traded = pp["volume"] > 0
    b1 = ds.bars(pp, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rng = np.where(traded & (hi > lo), np.log(hi / lo), 0.0)
        flow = np.where(traded & (hi > lo), (2 * cl - hi - lo) / (hi - lo), 0.0)
        own = np.where(traded & (op > 0), np.log(cl / op), 0.0)
    moves, sectors = basket(engine, basket_from, basket_to)
    want = p.close.index
    total = moves.groupby(level="ts")[["sum", "n"]].sum().reindex(want)
    mkt = (total["sum"] / total["n"]).to_numpy().reshape(n, SESSION, 1)
    listing = ex.listing(engine, pp["names"])["sector"].reindex(pp["names"]).fillna("other")
    in_basket = np.array([s in sectors.index for s in pp["names"]], float)
    sec = np.full(shape, np.nan)
    for k, name in enumerate(pp["names"]):
        part = moves.xs(listing[name], level="sector") if listing[name] in set(
            moves.index.get_level_values("sector")) else None
        if part is None:
            continue
        part = part.reindex(want)
        s_k = part["sum"].to_numpy().reshape(n, SESSION) - in_basket[k] * own[..., k]
        n_k = part["n"].to_numpy().reshape(n, SESSION) - in_basket[k] * traded[..., k]
        with np.errstate(invalid="ignore", divide="ignore"):
            sec[..., k] = np.where(n_k > 0, s_k / n_k, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rng_x = np.clip(rng / _typical(rng), 0, 10) - 1
    bar = {"path": b1["y"], "volume": b1["volume"], "range": rng_x, "flow": flow,
           "market": np.repeat(_basket_bars(pp, mkt, "market"), N, axis=2),
           "sector": _basket_bars(pp, sec, "sector")}
    prev = {d: ds.previous(pp, d)["prev_move"] for d in (1, 2, 5)}
    opt = None
    frames = rs.option_frames(pp["dates"], pp["names"]) if options else {}
    if all(f"opt_{s}" in frames for s in OPTION_SIGNALS):
        opt = np.stack([_zscore(frames[f"opt_{s}"].to_numpy(float)) for s in OPTION_SIGNALS], -1)
    data = np.isfinite(b1["u"]).any(axis=1)          # (session, stock): the stock has data
    with np.errstate(invalid="ignore"):
        has = {"market": np.repeat((np.isfinite(mkt[..., 0]).mean(axis=1) >= 0.5)[:, None], N, 1),
               "sector": np.isfinite(sec).mean(axis=1) >= 0.5,
               "options": (np.isfinite(opt).all(axis=-1) if opt is not None
                           else np.zeros((n, N), bool))}
    from_prices = {f: data for f in FEATURES if f not in has}   # the stock's own prices
    available = pd.DataFrame({f: [float(h[data[:, k], k].mean()) if data[:, k].any() else 0.0
                                  for k in range(N)] for f, h in {**from_prices, **has}.items()},
                             index=pp["names"])[list(FEATURES)]
    return {"pp": pp, "names": list(pp["names"]), "dates": np.array([str(d) for d in pp["dates"]]),
            "y": b1["y"], "u": b1["u"], "bar": bar, "gap": pp["gap"], "prev_move": prev,
            "options": opt, "sectors": listing, "in_basket": in_basket, "available": available,
            "basket_size": len(sectors)}


def _zscore(x: np.ndarray) -> np.ndarray:
    """Each column against its own previous :data:`OPTION_Z_DAYS` values (mean, SD; the value
    itself included: all are known at the session), clipped at ±:data:`DAY_CLIP`."""
    f = pd.DataFrame(x)
    r = f.rolling(OPTION_Z_DAYS, min_periods=20)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = ((f - r.mean()) / r.std()).to_numpy()
    return np.clip(z, -DAY_CLIP, DAY_CLIP)


def day_inputs(F: dict, cell, feats, prev_days: int = 1) -> np.ndarray:
    """The DAY inputs of one configuration, (sessions x stocks x columns), in the order of
    :data:`FEATURES`; NaN is 0. The gap is 0 where the stock had not traded before the decision;
    "same hour" is the mean move over the same clock window on the previous sessions."""
    _, off, L, H = cell
    D = off + L
    cols = []
    for f in [f for f in FEATURES if f in feats and FEATURES[f]["level"] == "day"]:
        if f == "gap":
            cols.append(np.where(F["pp"]["first"] < D, F["gap"], 0.0)[..., None])
        elif f == "prev":
            cols.append(F["prev_move"][prev_days][..., None])
        elif f == "same_hour":
            end = np.nan_to_num(F["y"][:, D:D + H]).sum(axis=1) / np.sqrt(H)
            mean = pd.DataFrame(end).rolling(SAME_HOUR_DAYS, min_periods=5).mean().shift(1)
            cols.append(np.clip(mean.to_numpy(), -DAY_CLIP, DAY_CLIP)[..., None])
        elif f == "options":
            if F["options"] is None:
                raise ValueError("no option data for these stocks: switch the input off")
            cols.append(F["options"])
    n, N = F["gap"].shape
    return np.nan_to_num(np.concatenate(cols, axis=-1)) if cols else np.zeros((n, N, 0))


def valid(F: dict, cell, min_traded: float = 0.5) -> np.ndarray:
    """Which (session, stock) make an example of ``cell``: a regular session, a typical scale on
    every minute of the input and the output (the stock's data reach far enough back), and a
    trade in at least ``min_traded`` of the input minutes."""
    _, off, L, H = cell
    D = off + L
    scale = np.isfinite(F["u"][:, off:D + H]).all(axis=1)
    traded = (F["pp"]["volume"][:, off:D] > 0).mean(axis=1) >= min_traded
    return scale & traded & F["pp"]["usable"][:, None]


def examples(F: dict, cell, days, feats, *, prev_days: int = 1, full: bool) -> tuple:
    """The Darts series of every valid (day, stock) of ``days``: the target minutes from
    09:30 + offset (``full``: the input and the output, exactly L + H minutes, one training sample
    each; else the input only), the BAR inputs of ``feats`` over the same minutes (one past
    covariate series with a column each, or None), and the static inputs: the stock's one-hot,
    then the DAY inputs of ``feats``, in one fixed column order.

    Returns:
        ``(series, covariates or None, day per series, stock per series)``.
    """
    from darts import TimeSeries

    _, off, L, H = cell
    n = L + H if full else L
    names = F["names"]
    bars = [f for f in FEATURES if f in feats and FEATURES[f]["level"] == "bar" and f != "path"]
    day = day_inputs(F, cell, feats, prev_days)
    cols = [f"is_{x}" for x in names] + [f"day_{i}" for i in range(day.shape[-1])]
    eye = np.eye(len(names))
    y = np.nan_to_num(F["y"]).astype(np.float32)
    cov = (np.nan_to_num(np.stack([F["bar"][f] for f in bars], axis=-1)).astype(np.float32)
           if bars else None)
    ok = valid(F, cell)
    idx = pd.RangeIndex(0, n)
    ser, covs, dd, kk = [], [], [], []
    for d in days:
        for k in range(len(names)):
            if not ok[d, k]:
                continue
            static = pd.DataFrame([np.r_[eye[k], day[d, k]]], columns=cols)
            ser.append(TimeSeries.from_times_and_values(idx, y[d, off:off + n, k:k + 1],
                                                        columns=["y"])
                       .with_static_covariates(static))
            if cov is not None:
                covs.append(TimeSeries.from_times_and_values(idx, cov[d, off:off + n, k],
                                                             columns=bars))
            dd.append(d)
            kk.append(k)
    return ser, (covs if cov is not None else None), np.array(dd, int), np.array(kk, int)


def end_moves(F: dict, cell, d: np.ndarray, k: np.ndarray) -> np.ndarray:
    """The actual move from the decision to the end of the horizon (typical moves) of every
    (day, stock)."""
    _, off, L, H = cell
    return np.nan_to_num(F["y"][d, off + L:off + L + H, k]).sum(axis=1)


# ---------------------------------------------------------------------------------------------
# Training, prediction, the grade.
# ---------------------------------------------------------------------------------------------
_PARTS: dict = {}


def _torch_parts():
    """The end-move loss and the per-pass recorder (built on first use: torch imports slowly)."""
    if _PARTS:
        return _PARTS["loss"], _PARTS["curve"]
    import pytorch_lightning as pl
    import torch

    class EndpointLoss(torch.nn.Module):
        """The squared error of the SUM of the predicted minute moves over the horizon: only
        the move to the end counts. Darts calls it with (batch, minutes, 1) tensors."""

        def __init__(self):
            super().__init__()
            self.reduction = "mean"  # Darts expects the attribute

        def forward(self, pred, target):
            err = (pred - target).sum(dim=1).pow(2)
            return err.mean() if self.reduction == "mean" else err

    class TrainingCurve(pl.Callback):
        """Every pass: mean training loss, held-out loss, held-out R² against "no change"
        (1 - loss / baseline); ``on_pass(self)`` after each pass."""

        def __init__(self, baseline, on_pass=None):
            self.baseline, self.on_pass, self.rows, self._losses = baseline, on_pass, [], []

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self._losses.append(float(outputs["loss"] if isinstance(outputs, dict) else outputs))

        def on_validation_epoch_end(self, trainer, pl_module):
            val = trainer.callback_metrics.get("val_loss")
            if trainer.sanity_checking or val is None:
                return
            self.rows.append({"pass": len(self.rows) + 1, "train": float(np.mean(self._losses)),
                              "validation": float(val), "r2": 1 - float(val) / self.baseline})
            self._losses = []
            if self.on_pass is not None:
                self.on_pass(self)

    _PARTS.update(loss=EndpointLoss, curve=TrainingCurve)
    return EndpointLoss, TrainingCurve


def _new_model(cell, hp: dict, *, n_samples: int, epochs: int, seed: int):
    """The untrained TSMixer of :func:`train` (``hp`` as there)."""
    import torch

    loss_cls, _ = _torch_parts()
    _, _, L, H = cell
    base = ex.make_path_model("tsmixer", L, H, epochs=epochs, n_samples=n_samples, seed=seed)
    hp = dict(hp)
    optim = {k: hp.pop(k) for k in ("lr", "weight_decay") if k in hp}
    hp["optimizer_kwargs"] = {"lr": ex.SETTINGS["lr"], **optim}
    hp["optimizer_cls"] = torch.optim.AdamW
    if hp.pop("loss", "bar") == "endpoint":
        hp["loss_fn"] = loss_cls()
    patience = hp.pop("patience", None)
    model = type(base)(**{**base.model_params, **hp})
    for c in model.trainer_params["callbacks"]:  # the early stopping's patience
        if patience is not None and hasattr(c, "patience"):
            c.patience = patience
    return model


def save_model(model, path) -> None:
    """A trained model's weights under ``path`` (``.ckpt``), with the two fields Darts reads
    back beside them (dtype, input shapes). Darts' own ``save`` and Lightning's checkpoint both
    pickle the loss, and the end-move loss (a class built inside :func:`_torch_parts`) cannot be
    pickled; predicting never reads the loss."""
    import torch

    m = model.model
    torch.save({"state_dict": m.state_dict(), "model_dtype": m.dtype,
                "train_sample_shape": m.train_sample_shape}, str(path))


def load_model(path, cell, hp: dict):
    """A model saved by :func:`save_model`, ready to :func:`predict`: TSMixer built again from
    its ``cell`` and knobs ``hp`` (as in :func:`train`), then its weights loaded."""
    model = _new_model(cell, hp, n_samples=0, epochs=1, seed=0)
    model.load_weights_from_checkpoint(file_name=str(path), load_encoders=False,
                                       skip_checks=True, map_location="cpu")
    return model


def train(F: dict, cell, fit_days, held_days, feats, hp: dict, *, prev_days: int = 1,
          epochs: int = 60, on_pass=None, seed: int = 0) -> tuple:
    """ONE TSMixer for every stock, fit on the valid stock-days of ``fit_days``; the
    ``held_days`` stop its training early and the best pass's weights are kept.

    Args:
        F: :func:`build`'s inputs.
        cell: (1, offset, input, output) minutes.
        fit_days: Session indices to learn on.
        held_days: Session indices held out (early stopping).
        feats: The inputs read (besides the path).
        hp: TSMixer's knobs on top of ``explorer.make_path_model``'s, plus "lr", "weight_decay"
            (AdamW), "patience" and "loss" ("bar" | "endpoint").
        prev_days: Sessions behind the "prev" input.
        epochs: At most this many passes.
        on_pass: Called with the per-pass recorder after every pass.
        seed: Random seed.

    Returns:
        ``(model, curve)``.
    """
    _, curve_cls = _torch_parts()
    _, off, L, H = cell
    ser, cov, _, _ = examples(F, cell, fit_days, feats, prev_days=prev_days, full=True)
    vser, vcov, _, _ = examples(F, cell, held_days, feats, prev_days=prev_days, full=True)
    if not ser or not vser:
        raise ValueError("no valid stock-day to learn on or to hold out")
    model = _new_model(cell, hp, n_samples=len(ser), epochs=epochs, seed=seed)
    endpoint = hp.get("loss", "bar") == "endpoint"
    y_out = [s.values()[L:, 0] for s in vser]
    no_change = np.mean([y.sum() ** 2 if endpoint else np.mean(y ** 2) for y in y_out])
    curve = curve_cls(float(no_change), on_pass)
    model.trainer_params["callbacks"].append(curve)
    kw = {"past_covariates": cov, "val_past_covariates": vcov} if cov is not None else {}
    model.fit(ser, val_series=vser, verbose=False, **kw)
    ex.SM["_quiet"]()
    keep = next(c for c in model.trainer_params["callbacks"] if hasattr(c, "state"))
    if keep.state is not None:  # back to the weights of the best pass
        model.model.load_state_dict(keep.state)
    return model, curve


def predict(model, F: dict, cell, days, feats, *, prev_days: int = 1) -> tuple:
    """The model's next H minute moves of every valid (day, stock) of ``days``, read from that
    day's input minutes only.

    Returns:
        ``(paths (examples x H), day, stock)``.
    """
    _, _, _, H = cell
    ser, cov, d, k = examples(F, cell, days, feats, prev_days=prev_days, full=False)
    if not ser:
        return np.zeros((0, H), np.float32), d, k
    kw = {"past_covariates": cov} if cov is not None else {}
    out = model.predict(n=H, series=ser, batch_size=1024, verbose=False, **kw)
    return np.stack([p.values()[:, 0] for p in out]).astype(np.float32), d, k


def _corr(p: np.ndarray, a: np.ndarray) -> float:
    """Pearson correlation; 0 when either side does not move."""
    return float(np.corrcoef(p, a)[0, 1]) if len(p) > 2 and np.std(p) > 0 and np.std(a) > 0 \
        else 0.0


def grade(pe: np.ndarray, ae: np.ndarray, d: np.ndarray, k: np.ndarray, F: dict, *,
          beta: float, draws: int = 200) -> tuple[dict, dict, dict]:
    """The end-move grade of predicted ``pe`` vs actual ``ae`` end moves: the correlation (the
    score), hit rate, size error after the scale ``beta``, per stock, per month, and the luck
    check: the same predictions against ``draws`` make-believe outcomes that keep every move's
    size but flip its sign at random, one coin per DAY for all stocks (``luck_z``, ``luck_p``;
    ``hit_p``: the hit rate by coin-tossing).

    Returns:
        ``(scores, per-stock correlation, per-month correlation)``.
    """
    from scipy.stats import binomtest

    hits = np.sign(pe) == np.sign(ae)
    _, by_day = np.unique(d, return_inverse=True)
    coins = np.random.default_rng(0).choice([-1.0, 1.0], size=(draws, by_day.max() + 1))
    luck = np.array([_corr(pe, c[by_day] * ae) for c in coins])
    out = {"end_corr": _corr(pe, ae), "hit_rate": float(hits.mean()),
           "hit_p": float(binomtest(int(hits.sum()), len(hits), 0.5,
                                    alternative="greater").pvalue),
           "mae_skill": float(1 - np.mean(np.abs(beta * pe - ae)) / np.mean(np.abs(ae))),
           "beta": float(beta), "loudness": float(np.sum(pe ** 2) / np.sum(ae ** 2))}
    out |= {"luck_z": float((out["end_corr"] - luck.mean()) / luck.std()) if luck.std() > 0
            else 0.0, "luck_p": float((1 + np.sum(luck >= out["end_corr"])) / (1 + draws))}
    per = {F["names"][s]: _corr(pe[k == s], ae[k == s]) for s in np.unique(k)}
    month = np.array([F["dates"][x][:7] for x in d])
    by_month = {m: _corr(pe[month == m], ae[month == m]) for m in np.unique(month)}
    out |= {"stocks_positive": float(np.mean([v > 0 for v in per.values()])),
            "months_positive": float(np.mean([v > 0 for v in by_month.values()]))}
    return out, per, by_month
