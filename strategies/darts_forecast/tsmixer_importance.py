"""What one trained TSMixer leans on, per input (``darts_forecast_tsmixer_marimo.py``, last
sections): permutation importance and SHAP values.

Both ask the same thing in the same way: what happens when an input is ABSENT. Absent means read
from ANOTHER day of the SAME stock, so the input keeps its level and spread and only its link to
this day's outcome is cut, while the stock and every other input stay. The model is not retrained.

- :func:`importance`: scramble one input (or a group) over the stock-days scored; the drop in
  end-move correlation is how much of the SCORE the input carries.
- :func:`shap_values`: Shapley values of the predicted end move over the inputs (the ``shap``
  library, which ``darts`` already requires): how much each input MOVED each prediction, signed,
  whether or not that helped.

It builds on ``tsmixer_lab`` (passed in as ``lab``) and edits nothing there: that file's source is
part of every study's name, so changing it would start a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def day_columns(lab, feats) -> dict[str, list[str]]:
    """The static ``day_*`` columns of every DAY input read, in the order ``lab.examples`` lays
    them out (the stock's one-hot comes first, then these).

    Args:
        lab: The ``tsmixer_lab`` module.
        feats: The optional inputs read.

    Returns:
        ``{input: [column names]}``; the option input takes one column per option signal.
    """
    out, i = {}, 0
    for f, v in lab.FEATURES.items():
        if f in feats and v["level"] == "day":
            out[f] = [f"day_{j}" for j in range(i, i + v["width"])]
            i += v["width"]
    return out


def players(lab, feats) -> list[str]:
    """The inputs that can be absent, in ``lab.FEATURES``' order: the stock's own path, then every
    optional input read."""
    return ["path", *[f for f in lab.FEATURES if f in feats and f != "path"]]


def pack(lab, ser, cov, feats) -> dict:
    """The examples of ``lab.examples`` as arrays, built once: the path (examples x minutes), the
    BAR inputs (examples x minutes x inputs), the static columns and where every input sits."""
    bars = [f for f, v in lab.FEATURES.items() if f in feats and v["level"] == "bar"
            and f != "path"]
    return {"y": np.stack([s.values()[:, 0] for s in ser]),
            "cov": None if cov is None else np.stack([c.values() for c in cov]),
            "static": np.stack([s.static_covariates.to_numpy(float)[0] for s in ser]),
            "scols": list(ser[0].static_covariates.columns), "bars": bars,
            "days": day_columns(lab, feats), "index": ser[0].time_index}


def compose(pk: dict, names: list[str], src: np.ndarray, host: np.ndarray) -> tuple:
    """One Darts series (and past covariates) per row of ``src``: row ``r`` reads the input
    ``names[g]`` from example ``src[r, g]``, and the stock and everything else from ``host[r]``.

    Args:
        pk: :func:`pack`'s arrays.
        names: The inputs, as :func:`players` lists them.
        src: (rows x inputs) example numbers.
        host: (rows,) the example whose stock the row has (one stock per row).

    Returns:
        ``(series, covariates or None)``.
    """
    from darts import TimeSeries

    g = {n: i for i, n in enumerate(names)}
    y = pk["y"][src[:, g["path"]]]
    static = pk["static"][host].copy()
    for f, cols in pk["days"].items():
        j = [pk["scols"].index(c) for c in cols]
        static[:, j] = pk["static"][src[:, g[f]]][:, j]
    idx = pk["index"]
    ser = [TimeSeries.from_times_and_values(idx, y[r][:, None], columns=["y"])
           .with_static_covariates(pd.DataFrame([static[r]], columns=pk["scols"]))
           for r in range(len(src))]
    if pk["cov"] is None:
        return ser, None
    cov = np.empty((len(src), *pk["cov"].shape[1:]), pk["cov"].dtype)
    for c, f in enumerate(pk["bars"]):
        cov[:, :, c] = pk["cov"][src[:, g[f]], :, c]
    return ser, [TimeSeries.from_times_and_values(idx, cov[r], columns=pk["bars"])
                 for r in range(len(src))]


def _swap(names, group, k, rng) -> np.ndarray:
    """``src`` for :func:`compose`: every example reads the ``group`` inputs of another example of
    its own stock (a random permutation within each stock) and the rest from itself."""
    n = len(k)
    donor = np.arange(n)
    for s in np.unique(k):
        idx = np.flatnonzero(k == s)
        donor[idx] = rng.permutation(idx)
    src = np.tile(np.arange(n)[:, None], (1, len(names)))
    for g, name in enumerate(names):
        if name in group:
            src[:, g] = donor
    return src


def scramble(lab, ser, cov, feats, group, k, rng) -> tuple[list, list | None]:
    """Copies of the examples in which every example reads the ``group`` inputs of another
    example of the same stock.

    Args:
        lab: The ``tsmixer_lab`` module.
        ser: The Darts target series (the path as ``y``, the stock and DAY inputs as static
            covariates), one per example.
        cov: The BAR inputs as past covariates, one per example (None: no BAR input is read).
        feats: The optional inputs read.
        group: The names to scramble together; ``"path"`` is the stock's own path.
        k: The stock of every example.
        rng: A numpy random generator.

    Returns:
        ``(series, covariates or None)``: new lists.
    """
    names = players(lab, feats)
    return compose(pack(lab, ser, cov, feats), names, _swap(names, group, k, rng),
                   np.arange(len(ser)))


def _corr(p: np.ndarray, a: np.ndarray) -> float:
    """Pearson correlation; 0 when either side does not move."""
    return float(np.corrcoef(p, a)[0, 1]) if len(p) > 2 and np.std(p) > 0 and np.std(a) > 0 \
        else 0.0


def _ends(model, ser, cov, H: int) -> np.ndarray:
    """The model's predicted move to the end of the horizon: its next ``H`` minute moves, summed."""
    kw = {"past_covariates": cov} if cov is not None else {}
    out = model.predict(n=H, series=ser, batch_size=1024, verbose=False, **kw)
    return np.stack([p.values()[:, 0] for p in out]).sum(axis=1)


def importance(lab, model, F: dict, cell, days, feats, *, prev_days: int = 1,
               repeats: int = 10, seed: int = 0) -> tuple[float, pd.DataFrame]:
    """The end-move correlation of ``model`` on the valid stock-days of ``days`` and how much of
    it every input carries: the correlation again with that input scrambled (``repeats`` times).

    Args:
        lab: The ``tsmixer_lab`` module.
        model: A trained TSMixer (``lab.train``).
        F: ``lab.build``'s inputs.
        cell: (1, offset, input, output) minutes, the model's own.
        days: Session indices to score (VALIDATION).
        feats: The optional inputs the model reads.
        prev_days: Sessions behind the "prev" input.
        repeats: Scrambles per input.
        seed: Random seed of the scrambles.

    Returns:
        ``(base, table)``: the unscrambled correlation and one row per input (and one for market
        and sector together when both are read, since they share most of their information):
        ``drop`` = base minus the mean scrambled correlation, its ``sd`` over the scrambles, the
        share of scrambles that ``hurt`` (lowered the correlation), best first.
    """
    H = cell[3]
    ser, cov, d, k = lab.examples(F, cell, days, feats, prev_days=prev_days, full=False)
    if len(ser) < 3:
        raise ValueError("fewer than 3 valid stock-days to score")
    ae = lab.end_moves(F, cell, d, k)
    pk, names = pack(lab, ser, cov, feats), players(lab, feats)
    base = _corr(_ends(model, ser, cov, H), ae)
    groups = [(f,) for f in names] + ([("market", "sector")] if {"market", "sector"} <= set(feats)
                                      else [])
    rng = np.random.default_rng(seed)
    rows, me = [], np.arange(len(ser))
    for g in groups:
        c = np.array([_corr(_ends(model, *compose(pk, names, _swap(names, g, k, rng), me), H), ae)
                      for _ in range(repeats)])
        rows.append({"input": " + ".join(g),
                     "level": "per minute" if lab.FEATURES[g[0]]["level"] == "bar" else "per day",
                     "drop": base - c.mean(), "sd": c.std(), "hurt": float(np.mean(c < base)),
                     "scrambled corr": c.mean()})
    return base, pd.DataFrame(rows).sort_values("drop", ascending=False).reset_index(drop=True)


def shap_values(lab, model, F: dict, cell, days, feats, *, prev_days: int = 1,
                explain: int = 30, background: int = 10, permutations: int = 6,
                seed: int = 0) -> dict:
    """Shapley values of the predicted end move over the inputs (the ``shap`` library's
    permutation explainer), per stock: an input is absent when it is read from one of
    ``background`` other days of the SAME stock, and the value of a present input is how far it
    moves this prediction away from the average of those.

    Args:
        lab: The ``tsmixer_lab`` module.
        model: A trained TSMixer (``lab.train``).
        F: ``lab.build``'s inputs.
        cell: (1, offset, input, output) minutes, the model's own.
        days: Session indices to explain (VALIDATION).
        feats: The optional inputs the model reads.
        prev_days: Sessions behind the "prev" input.
        explain: Stock-days explained per stock, spread evenly over the days.
        background: Other days of the stock that stand for "absent".
        permutations: Random input orders per stock-day (more: less noise, slower).
        seed: Random seed.

    Returns:
        A dict: ``names`` (the inputs), ``values`` (stock-days x inputs, in typical moves of the
        predicted end move), ``base`` (the average prediction when every input is absent),
        ``pred`` (the model's own prediction; ``base + values.sum(1)`` equals it up to
        rounding), ``stock`` and ``date`` of every row, ``x`` (the value of every one-number DAY
        input).
    """
    import shap

    H = cell[3]
    ser, cov, d, k = lab.examples(F, cell, days, feats, prev_days=prev_days, full=False)
    pk, names = pack(lab, ser, cov, feats), players(lab, feats)
    P, rng = len(names), np.random.default_rng(seed)

    def f(z: np.ndarray) -> np.ndarray:  # row r: input g read from example z[r, g]
        src = np.rint(z).astype(int)
        return _ends(model, *compose(pk, names, src, src[:, 0]), H)

    picked, vals, base = [], [], []
    for s in np.unique(k):
        idx = np.flatnonzero(k == s)
        if len(idx) < 3:
            continue
        pick = idx[np.linspace(0, len(idx) - 1, min(explain, len(idx))).round().astype(int)]
        bg = rng.choice(idx, min(background, len(idx)), replace=False)
        masker = shap.maskers.Independent(np.repeat(bg[:, None], P, axis=1).astype(float),
                                          max_samples=len(bg))
        out = shap.PermutationExplainer(f, masker, seed=seed)(
            np.repeat(pick[:, None], P, axis=1).astype(float),
            max_evals=2 * P * permutations + 1, silent=True)
        picked.append(pick)
        vals.append(out.values)
        base.append(np.broadcast_to(out.base_values, (len(pick),)))
    if not picked:
        raise ValueError("no stock with 3 valid stock-days to explain")
    pick = np.concatenate(picked)
    same = np.repeat(pick[:, None], P, axis=1)
    scalar = {f_: c[0] for f_, c in pk["days"].items() if len(c) == 1}
    return {"names": names, "values": np.concatenate(vals), "base": np.concatenate(base),
            "pred": f(same.astype(float)), "stock": np.array([F["names"][i] for i in k[pick]]),
            "date": np.array([F["dates"][i] for i in d[pick]]),
            "x": pd.DataFrame({f_: pk["static"][pick, pk["scols"].index(c)]
                               for f_, c in scalar.items()})}
