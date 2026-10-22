"""tsmixer_importance: the scrambles keep every input's values within its stock and cut only the
named input; a model that reads one input loses exactly that one; SHAP values give that input
the whole move and add up to the prediction; Darts accepts the rebuilt series. Synthetic series
only (no DB); skipped when the optional ``darts`` extra is missing.
"""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies

pytest.importorskip("darts")
from darts import TimeSeries  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"test_{name}", strategies.STRATEGIES_DIR / "darts_forecast" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LAB = _load("tsmixer_lab")
IMP = _load("tsmixer_importance")
L, H, PER_STOCK = 12, 4, 30


def examples(seed: int = 0):
    """Two stocks x 30 examples with the real layout of ``lab.examples``: the path as ``y`` with
    the one-hot and the gap as statics, the volume as a past covariate. The outcome is the last
    input minute's volume plus a little noise."""
    rng = np.random.default_rng(seed)
    cols = ["is_A", "is_B", "day_0"]
    ser, cov, k = [], [], []
    for s in (0, 1):
        for _ in range(PER_STOCK):
            idx = pd.RangeIndex(L)
            static = pd.DataFrame([np.r_[np.eye(2)[s], rng.normal()]], columns=cols)
            ser.append(TimeSeries.from_times_and_values(idx, rng.normal(size=(L, 1)),
                                                        columns=["y"])
                       .with_static_covariates(static))
            cov.append(TimeSeries.from_times_and_values(idx, rng.normal(size=(L, 1)),
                                                        columns=["volume"]))
            k.append(s)
    last = np.array([c.values()[-1, 0] for c in cov])
    return ser, cov, np.array(k), last + rng.normal(0, 0.05, len(last))


class ReadsVolume:
    """A stand-in model: predicts the last volume as the end move, reads nothing else."""

    def predict(self, n, series, past_covariates=None, batch_size=None, verbose=False):
        return [TimeSeries.from_times_and_values(
            pd.RangeIndex(n), np.full((n, 1), c.values()[-1, 0] / n)) for c in past_covariates]


def fake_lab(ser, cov, k, ae):
    return SimpleNamespace(
        FEATURES=LAB.FEATURES, end_moves=lambda *a: ae,
        examples=lambda *a, **kw: (ser, cov, np.zeros(len(ser), int), k))


def test_day_columns_follow_the_examples_layout_and_options_take_three():
    got = IMP.day_columns(LAB, ["volume", "options", "gap", "same_hour"])
    assert got == {"gap": ["day_0"], "same_hour": ["day_1"],
                   "options": ["day_2", "day_3", "day_4"]}  # FEATURES' order, not the argument's


@pytest.mark.parametrize("group", [("path",), ("volume",), ("gap",)])
def test_a_scramble_permutes_only_the_named_input_within_each_stock(group):
    ser, cov, k, _ = examples()
    rng = np.random.default_rng(1)
    s2, c2 = IMP.scramble(LAB, ser, cov, ["volume", "gap"], group, k, rng)
    path = lambda a: np.array([x.values()[:, 0] for x in a])  # noqa: E731
    vol = lambda a: np.array([x.values()[:, 0] for x in a])  # noqa: E731
    gap = lambda a: np.array([x.static_covariates["day_0"].iloc[0] for x in a])  # noqa: E731
    hot = lambda a: np.array([x.static_covariates[["is_A", "is_B"]].to_numpy()[0] for x in a])  # noqa: E731
    np.testing.assert_array_equal(hot(s2), hot(ser))  # the stock never changes
    for name, get, before, after in (("path", path, ser, s2), ("volume", vol, cov, c2),
                                     ("gap", gap, ser, s2)):
        moved = not np.array_equal(get(before), get(after))
        assert moved == (name in group), name
        if moved:  # a permutation inside each stock: same values, other days
            for s in (0, 1):
                a, b = np.sort(get(before)[k == s], axis=0), np.sort(get(after)[k == s], axis=0)
                np.testing.assert_allclose(a.ravel(), b.ravel())


def test_a_model_that_reads_only_the_volume_loses_exactly_the_volume():
    ser, cov, k, ae = examples()
    base, table = IMP.importance(fake_lab(ser, cov, k, ae), ReadsVolume(), {}, (1, 0, L, H),
                                 [0], ["volume", "gap"], repeats=5)
    assert base > 0.99
    t = table.set_index("input")
    assert t.index[0] == "volume" and t.loc["volume", "drop"] > 0.8
    assert t.loc["volume", "hurt"] == 1.0
    assert t.loc[["path", "gap"], "drop"].abs().max() == 0.0  # it never looked at them


def test_market_and_sector_are_also_scrambled_together():
    ser, cov, k, ae = examples()
    cov2 = [TimeSeries.from_times_and_values(c.time_index, np.repeat(c.values(), 3, axis=1),
                                             columns=["volume", "market", "sector"]) for c in cov]
    _, table = IMP.importance(fake_lab(ser, cov2, k, ae), ReadsVolume(), {}, (1, 0, L, H), [0],
                              ["volume", "market", "sector"], repeats=2)
    assert "market + sector" in set(table["input"])


def test_darts_accepts_the_scrambled_series_of_a_real_tsmixer():
    pytest.importorskip("torch")
    from darts.models import TSMixerModel

    ser, cov, k, ae = examples()
    full = [TimeSeries.from_times_and_values(pd.RangeIndex(L + H), np.random.default_rng(i).normal(
        size=(L + H, 1)), columns=["y"]).with_static_covariates(s.static_covariates)
        for i, s in enumerate(ser)]
    fcov = [TimeSeries.from_times_and_values(pd.RangeIndex(L + H), np.random.default_rng(i).normal(
        size=(L + H, 1)), columns=["volume"]) for i in range(len(ser))]
    model = TSMixerModel(
        input_chunk_length=L, output_chunk_length=H, hidden_size=8, ff_size=8, num_blocks=1,
        n_epochs=1, batch_size=16, use_static_covariates=True, random_state=0,
        pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False,
                           "enable_model_summary": False, "logger": False,
                           "enable_checkpointing": False})
    model.fit(full, past_covariates=fcov, verbose=False)
    base, table = IMP.importance(fake_lab(ser, cov, k, ae), model, {}, (1, 0, L, H), [0],
                                 ["volume", "gap"], repeats=2)
    assert np.isfinite(base) and set(table["input"]) == {"path", "volume", "gap"}
    assert np.isfinite(table[["drop", "sd", "scrambled corr"]].to_numpy()).all()


def test_compose_reads_every_input_from_its_own_example_and_the_stock_from_the_host():
    ser, cov, k, _ = examples()
    names = IMP.players(LAB, ["volume", "gap"])
    assert names == ["path", "volume", "gap"]
    pk = IMP.pack(LAB, ser, cov, ["volume", "gap"])
    src = np.array([[3, 7, 11], [0, 0, 0]])  # row 0: path from 3, volume from 7, gap from 11
    s2, c2 = IMP.compose(pk, names, src, np.array([40, 0]))  # host 40 is stock 1
    np.testing.assert_allclose(s2[0].values(), ser[3].values())
    np.testing.assert_allclose(c2[0].values(), cov[7].values())
    st = s2[0].static_covariates
    assert st[["is_A", "is_B"]].to_numpy()[0].tolist() == [0.0, 1.0]  # the host's stock
    assert st["day_0"].iloc[0] == ser[11].static_covariates["day_0"].iloc[0]
    for a, b in ((s2[1], ser[0]), (c2[1], cov[0])):  # the identity row changes nothing
        np.testing.assert_allclose(a.values(), b.values())
    assert s2[1].static_covariates.equals(ser[0].static_covariates)


def shap_lab(ser, cov, k):
    lab = fake_lab(ser, cov, k, np.zeros(len(ser)))
    lab.examples = lambda *a, **kw: (ser, cov, np.zeros(len(ser), int), k)
    return lab


def test_shap_gives_a_model_that_reads_only_the_volume_all_of_the_move():
    ser, cov, k, _ = examples()
    F = {"names": ["A", "B"], "dates": ["2026-01-02"]}
    r = IMP.shap_values(shap_lab(ser, cov, k), ReadsVolume(), F, (1, 0, L, H), [0],
                        ["volume", "gap"], explain=8, background=6, permutations=3)
    v = r["values"]
    assert r["names"] == ["path", "volume", "gap"] and v.shape == (16, 3)  # 8 per stock
    np.testing.assert_allclose(v[:, [0, 2]], 0, atol=1e-9)  # it never looked at them
    np.testing.assert_allclose(r["base"] + v.sum(axis=1), r["pred"], atol=1e-5)  # adds up
    np.testing.assert_allclose(v[:, 1], r["pred"] - r["base"], atol=1e-5)
    assert np.abs(v[:, 1]).mean() > 0.3  # the volume really moves the prediction
    assert set(r["stock"]) == {"A", "B"} and list(r["x"].columns) == ["gap"]


def test_shap_values_of_a_real_tsmixer_add_up_to_its_predictions():
    pytest.importorskip("torch")
    from darts.models import TSMixerModel

    ser, cov, k, _ = examples()
    full = [TimeSeries.from_times_and_values(pd.RangeIndex(L + H), np.random.default_rng(i).normal(
        size=(L + H, 1)), columns=["y"]).with_static_covariates(s.static_covariates)
        for i, s in enumerate(ser)]
    fcov = [TimeSeries.from_times_and_values(pd.RangeIndex(L + H), np.random.default_rng(i).normal(
        size=(L + H, 1)), columns=["volume"]) for i in range(len(ser))]
    model = TSMixerModel(
        input_chunk_length=L, output_chunk_length=H, hidden_size=8, ff_size=8, num_blocks=1,
        n_epochs=1, batch_size=16, use_static_covariates=True, random_state=0,
        pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False,
                           "enable_model_summary": False, "logger": False,
                           "enable_checkpointing": False})
    model.fit(full, past_covariates=fcov, verbose=False)
    F = {"names": ["A", "B"], "dates": ["2026-01-02"]}
    r = IMP.shap_values(shap_lab(ser, cov, k), model, F, (1, 0, L, H), [0], ["volume", "gap"],
                        explain=3, background=4, permutations=2)
    np.testing.assert_allclose(r["base"] + r["values"].sum(axis=1), r["pred"], atol=1e-4)
    assert np.isfinite(r["values"]).all() and r["values"].shape == (6, 3)
