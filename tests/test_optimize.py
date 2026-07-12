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
