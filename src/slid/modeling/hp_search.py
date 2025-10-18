import gc
import torch
import optuna
from optuna import Trial
from typing import Dict, Union


def hp_space(trial: Trial, seed: int = 42) -> Dict[str, Union[float, int, str]]:
    return {
        "body_learning_rate": trial.suggest_float(
            "body_learning_rate", low=1e-6, high=1e-3, log=True
        ),
        "num_epochs": trial.suggest_int("num_epochs", low=1, high=2),
        "batch_size": trial.suggest_categorical("batch_size", choices=[2, 4, 8]),
        "sampling_strategy": trial.suggest_categorical(
            "sampling_strategy", choices=["undersampling", "oversampling"]
        ),
        "end_to_end": True,
        "save_strategy": "epoch",
        "use_amp": True,
        "l2_weight": 0.01,
        "max_length": 512,
        "save_strategy": "no",
        "seed": seed,
        # head parameters
        "max_iter": trial.suggest_int("max_iter", low=50, high=300, step=50),
        "solver": trial.suggest_categorical(
            "solver", ["newton-cg", "lbfgs", "liblinear"]
        ),
    }


def run_hp_search_optuna_updated(trainer, n_trials, direction, **kwargs):
    """Update hp search to add memory management."""

    def _objective(trial):
        trainer.objective = None
        trainer.train(trial=trial)

        # memory management
        del trainer.model
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate if needed
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)

        return trainer.objective

    timeout = kwargs.pop("timeout", None)
    n_jobs = kwargs.pop("n_jobs", 1)
    study = optuna.create_study(direction=direction, **kwargs)

    # memory management : overkill, but also adding gc_after_trial=True in study.optimize()
    study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, gc_after_trial=True)  # type: ignore
    best_trial = study.best_trial
    return BestRun(str(best_trial.number), best_trial.value, best_trial.params, study)  # type: ignore
