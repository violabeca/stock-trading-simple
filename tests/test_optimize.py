"""Optuna harness: space mapping, storage paths, Pareto picking, boundary checks."""

import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")

from trading_simple import optimize as opt  # noqa: E402
from trading_simple import strategies  # noqa: E402


def test_suggest_params_covers_every_space_type():
    space = {
        "window": {"type": "int", "low": 5, "high": 50},
        "log_horizon": {"type": "int", "low": 1, "high": 60, "log": True},
        "threshold": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.25},
        "direction": {"type": "categorical", "choices": ["long", "short", "both"]},
    }
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    caught = {}
    study.optimize(lambda t: caught.update(opt.suggest_params(t, space)) or 0.0, n_trials=1)
    assert 5 <= caught["window"] <= 50
    assert 1 <= caught["log_horizon"] <= 60
    assert caught["threshold"] % 0.25 == pytest.approx(0.0)
    assert caught["direction"] in {"long", "short", "both"}


def test_suggest_params_unknown_type_raises():
    study = optuna.create_study()
    trial = study.ask()
    with pytest.raises(ValueError, match="Unknown space type"):
        opt.suggest_params(trial, {"x": {"type": "gaussian"}})


def test_make_sampler_selection_and_unknown():
    cfg = {"seed": 7}
    assert isinstance(opt.make_sampler(cfg | {"sampler": "nsga2"}),
                      optuna.samplers.NSGAIISampler)
    assert isinstance(opt.make_sampler(cfg | {"sampler": "tpe"}), optuna.samplers.TPESampler)
    with pytest.raises(KeyError, match="Choices"):
        opt.make_sampler({"sampler": "genetic"})


def test_storage_path_lives_in_strategy_folder():
    strat = strategies.load("volume_momentum")
    opt.storage_for(strat, "unit-test-storage")
    path = strat.dir / "studies" / "unit-test-storage.log"
    assert path.exists()
    path.unlink()  # keep the repo clean


def test_progress_callback_prints_one_line_per_trial(capsys):
    """The live log must name the objectives it is actually printing.

    It used to hard-code "sharpe=", so once the study moved to benchmark-relative
    objectives every streamed line labelled an information ratio as a Sharpe — a wrong
    number in front of you for as long as the run lasts.
    """
    study = optuna.create_study(directions=["maximize", "maximize"],
                                sampler=optuna.samplers.RandomSampler(seed=3))
    study.optimize(lambda t: (t.suggest_float("w", 0, 1), 0.2), n_trials=2,
                   callbacks=[opt.progress_callback()])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2
    assert all(all(f"{name}=" in line for name in opt.OBJECTIVES) and "w=" in line
               for line in out)
    assert not any("sharpe=" in line for line in out)   # not an objective any more


def test_study_path_matches_storage_location():
    strat = strategies.load("volume_momentum")
    assert opt.study_path(strat, "s1") == strat.dir / "studies" / "s1.log"


def test_pick_best_weights_change_the_winner():
    study = optuna.create_study(directions=["maximize", "maximize"])
    for sharpe, ret in [(2.0, 0.10), (0.5, 0.90), (1.5, 0.50)]:
        study.add_trial(optuna.trial.create_trial(
            params={}, distributions={}, values=[sharpe, ret]))
    assert opt.pick_best(study, weights=(1.0, 0.0)).values[0] == 2.0   # pure Sharpe
    assert opt.pick_best(study, weights=(0.0, 1.0)).values[1] == 0.90  # pure return
    balanced = opt.pick_best(study, weights=(1.0, 1.0))
    assert balanced.values == [1.5, 0.5]                               # the compromise


def test_boundary_report_flags_edges_log_aware():
    space = {
        "window": {"type": "int", "low": 10, "high": 100},
        "mult": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.25},
        "horizon": {"type": "int", "low": 5, "high": 120, "log": True},
        "direction": {"type": "categorical", "choices": ["long", "short", "both"]},
    }
    best = {"window": 10, "mult": 2.5, "horizon": 110, "direction": "both"}
    report = opt.boundary_report(best, space)
    assert "direction" not in report.index                  # categoricals have no edges
    assert report.loc["window", "at_edge"] == "low"
    assert report.loc["window", "action"] == "widen low"
    assert report.loc["mult", "action"] == "ok"
    # 110 of [5, 120] is >5% from the top linearly but <5% away in log space
    assert report.loc["horizon", "at_edge"] == "high"
    assert opt.boundary_report({}, {}).empty                # no numeric params -> empty report


def test_boundary_report_counts_front_share():
    space = {"w": {"type": "int", "low": 1, "high": 100}}
    front = pd.DataFrame({"w": [1, 2, 99, 100]})
    report = opt.boundary_report({"w": 50}, space, front=front)
    assert report.loc["w", "at_edge"] == ""                 # the pick itself is centered
    assert report.loc["w", "front_at_low"] == 0.5
    assert report.loc["w", "front_at_high"] == 0.5
    assert report.loc["w", "action"] == "widen low & widen high"


SPACE = {
    "window": {"type": "int", "low": 10, "high": 100},
    "horizon": {"type": "int", "low": 5, "high": 120, "log": True},
    "weight": {"type": "float", "low": 0.25, "high": 3.0, "step": 0.25},
    "events": {"type": "int", "low": 1, "high": 3},          # too coarse for ±10% of its span
    "displace": {"type": "categorical", "choices": [True, False]},
}
PICKED = {"window": 50, "horizon": 24, "weight": 1.0, "events": 1, "displace": True}


def test_neighbors_moves_one_coordinate_at_a_time_on_the_searched_scale():
    star = opt.neighbors(PICKED, SPACE)
    center = star[star["perturbed"] == ""]
    assert len(center) == 1 and center["offset"].iloc[0] == 0.0
    for _, row in star[star["perturbed"] != ""].iterrows():
        differs = [k for k in PICKED if row[k] != PICKED[k]]
        assert differs == [row["perturbed"]], "a neighbor must move exactly one knob"

    def values(name):
        return star.loc[star["perturbed"] == name, name].tolist()

    window = values("window")
    assert all(isinstance(v, int) for v in window)            # an int space stays integral
    assert window == sorted(window) and window[0] < 50 < window[-1]
    assert all(SPACE["window"]["low"] <= v <= SPACE["window"]["high"] for v in window)
    # A log range is perturbed multiplicatively, so the up-move is the bigger absolute one.
    horizon = values("horizon")
    assert horizon[-1] - 24 > 24 - horizon[0]
    # 5% and 10% of the weight span both snap onto the same 0.25 grid point: one row, not two.
    assert values("weight") == [0.75, 1.25]
    # 10% of a 1-3 span rounds straight back to the pick. Falling back to the smallest
    # representable step is what stops a coarse knob being reported inert by construction.
    assert values("events") == [2]                            # 0 is out of bounds, dropped
    switch = star[star["perturbed"] == "displace"]
    assert switch["displace"].tolist() == [False] and switch["offset"].isna().all()


def test_neighbors_drops_perturbations_the_strategy_rejects():
    """A neighbor that violates valid() is never backtested — nor silently kept."""
    space = {"q_in": {"type": "float", "low": 0.6, "high": 0.95, "step": 0.05},
             "q_out": {"type": "float", "low": 0.4, "high": 0.9, "step": 0.05}}
    star = opt.neighbors({"q_in": 0.75, "q_out": 0.70}, space,
                         valid=lambda p: p["q_out"] < p["q_in"])
    assert (star["q_out"] < star["q_in"]).all()
    assert 0.75 not in star["q_out"].tolist()   # the step up would tie the entry band


def _star(rows: list[dict], center: float = 0.10) -> pd.DataFrame:
    """A synthetic sensitivity() frame: one center row plus labelled perturbations."""
    base = {"knob": 1.0, "switch": True, "n_trades": 100}
    out = [{**base, "perturbed": "", "offset": 0.0, "excess_cagr": center}]
    out += [{**base, **r} for r in rows]
    return pd.DataFrame(out)


def test_stability_report_verdicts_and_the_action_each_implies():
    def verdict(low, high, **over):
        star = _star([{"perturbed": "knob", "offset": -0.05, "knob": 0.9, "excess_cagr": low},
                      {"perturbed": "knob", "offset": 0.05, "knob": 1.1, "excess_cagr": high}],
                     **over)
        return opt.stability_report(star).loc["knob"]

    assert verdict(0.099, 0.101)["verdict"] == "inert"        # nothing moves -> stop searching it
    assert verdict(0.070, 0.065)["verdict"] == "plateau"      # it costs something, nothing breaks
    assert verdict(0.090, 0.140)["verdict"] == "off-center"   # a neighbor wins in sample
    assert verdict(0.010, 0.010)["verdict"] == "spike"        # both sides collapse
    cliff = verdict(0.090, 0.020)
    assert cliff["verdict"] == "cliff (high)"
    assert cliff["suggest"] == 0.9                            # step AWAY from the edge
    assert "knob=0.9" in cliff["action"]

    # A pick on a searched BOUND has no safe side to step onto: suggesting the picked
    # value back would be a no-op move, and the real fix is a wider bound (§8), not this.
    on_bound = opt.stability_report(_star(
        [{"perturbed": "knob", "offset": 0.05, "knob": 1.1, "excess_cagr": 0.02}])).loc["knob"]
    assert on_bound["verdict"] == "cliff (high)" and on_bound["suggest"] is None
    assert "widen" in on_bound["action"]
    # A negative center is the trap the old 'median > 0.5 * max' rule inverted on: half of
    # a negative best is BIGGER than it, so a failing strategy reported a healthy plateau.
    assert verdict(-0.20, -0.30, center=-0.10)["verdict"] == "n/a"


def test_stability_report_reads_a_categorical_and_counts_thin_neighbors():
    star = _star([{"perturbed": "switch", "offset": float("nan"), "switch": False,
                   "excess_cagr": 0.01, "n_trades": 3}])
    row = opt.stability_report(star, min_trades=30).loc["switch"]
    assert row["verdict"] == "decisive"          # the other choice gives the edge back
    assert row["n_thin"] == 1                    # counted, not hidden: the book starved
    assert opt.stability_report(_star([])).empty


def test_sensitivity_runs_the_star_through_the_engine(panel):
    """End to end: labelled rows in, one backtested row each out, types intact."""
    from trading_simple.config import Settings

    strat = strategies.load("volume_momentum")
    picked = {k: strat.params[k] for k in strat.optuna["space"]}
    star = opt.sensitivity(strat, panel, picked, settings=Settings())
    assert len(star) == len(opt.neighbors(picked, strat.optuna["space"], valid=strat.valid))
    assert {"perturbed", "offset", "excess_cagr", "n_trades"} <= set(star.columns)
    assert star["excess_cagr"].notna().all()
    assert star["fast"].dtype.kind == "i"        # an int knob must not reach signals() as a float
    report = opt.stability_report(star, min_trades=1)
    assert set(report.index) <= set(picked) and "action" in report.columns


def test_objective_prunes_below_min_trades_per_year():
    """A handful of lucky trades must not crown the front: the floor prunes them.

    The floor is a RATE, not an absolute count: an absolute floor would be
    implicitly far stricter on a short backtest window than a long one for the
    identical trading style (this bit donchian_breakout's short window in
    production — see the README/CLAUDE.md "Comparability rules").
    """
    from dataclasses import replace

    from tests.conftest import make_panel
    from trading_simple import backtest as bt
    from trading_simple.config import Settings

    panel = make_panel()   # ~0.23-year window: one trade here is a rate of ~4.4/year

    def one_trade(p, **_):
        entries = pd.DataFrame(False, index=p.close.index, columns=p.close.columns)
        entries.iloc[5, 0] = True
        return {"entries": entries, "exits": bt.exit_after(entries, 10)}

    strat = strategies.load("volume_momentum")

    def run_study(optuna_cfg):
        s = replace(strat, signals=one_trade, valid=lambda p: True, params={},
                    optuna=optuna_cfg)
        study = optuna.create_study(directions=["maximize", "maximize"],
                                    sampler=optuna.samplers.RandomSampler(seed=1))
        study.optimize(lambda t: opt.objective(t, s, panel, Settings()), n_trials=1)
        return study.trials[0].state

    assert (run_study({"space": {}, "min_trades_per_year": 20})
            is optuna.trial.TrialState.PRUNED)          # ~4.4/year < 20/year
    assert run_study({"space": {}}) is optuna.trial.TrialState.COMPLETE


def test_pareto_and_trials_frames():
    first, second = opt.OBJECTIVES          # (info_ratio, excess_return) — benchmark-relative
    study = optuna.create_study(directions=["maximize", "maximize"])
    for a, b in [(1.0, 0.2), (0.2, 0.8), (0.1, 0.1),                   # third is dominated
                 (1.0, 0.2)]:                                          # fourth duplicates first
        study.add_trial(optuna.trial.create_trial(
            params={"w": 1}, distributions={"w": optuna.distributions.IntDistribution(1, 2)},
            values=[a, b]))
    front = opt.pareto_frame(study)
    assert len(front) == 2 and front[first].is_monotonic_decreasing    # dup collapsed
    assert front["trial"].tolist() == [0, 1]                           # earliest trial kept
    everything = opt.trials_frame(study)
    assert len(everything) == 4 and {"trial", "w", first, second} <= set(everything)
