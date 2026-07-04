"""Optuna harness: space mapping, storage paths, Pareto picking."""

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


def test_pick_best_weights_change_the_winner():
    study = optuna.create_study(directions=["maximize", "maximize"])
    for sharpe, ret in [(2.0, 0.10), (0.5, 0.90), (1.5, 0.50)]:
        study.add_trial(optuna.trial.create_trial(
            params={}, distributions={}, values=[sharpe, ret]))
    assert opt.pick_best(study, weights=(1.0, 0.0)).values[0] == 2.0   # pure Sharpe
    assert opt.pick_best(study, weights=(0.0, 1.0)).values[1] == 0.90  # pure return
    balanced = opt.pick_best(study, weights=(1.0, 1.0))
    assert balanced.values == [1.5, 0.5]                               # the compromise


def test_pareto_and_trials_frames():
    study = optuna.create_study(directions=["maximize", "maximize"])
    for sharpe, ret in [(1.0, 0.2), (0.2, 0.8), (0.1, 0.1)]:           # third is dominated
        study.add_trial(optuna.trial.create_trial(
            params={"w": 1}, distributions={"w": optuna.distributions.IntDistribution(1, 2)},
            values=[sharpe, ret]))
    front = opt.pareto_frame(study)
    assert len(front) == 2 and front["sharpe"].is_monotonic_decreasing
    everything = opt.trials_frame(study)
    assert len(everything) == 3 and {"trial", "w", "sharpe", "total_return"} <= set(everything)
