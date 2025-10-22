"""Trainer module."""

import gc
import torch
import argparse
import pandas as pd
from functools import partial
from time import perf_counter
from typing import Any, Dict, Optional
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from setfit import SetFitModel, Trainer, sample_dataset

from slid.utils import utils
from slid.preprocessing import log
from slid.modeling import hp_search
from slid.modeling import evaluator


text_column = "log"
label_column = "flaky"
time_column = "created_at"
DEFAUTL_PRETRAINED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_NUM_SHOTS = 12  # 12 shots is recommended from https://arxiv.org/abs/2507.04173
N_TRIALS_HPP_SEARCH = 5


def load_dataframe(input_file: str) -> pd.DataFrame:
    """Read input file and apply log processing."""
    df = pd.read_csv(input_file)
    df[text_column] = df[text_column].astype(str)
    df[text_column] = df[text_column].apply(log.clean)
    df.sort_values(by=time_column, ascending=True, inplace=True)
    return df


def create_dataset_split_dict(
    df: pd.DataFrame,
    valid_size: float = 0.5,
    test_size: float = 0.5,
    random_seed: int = 42,
    shuffle: bool = True,
) -> DatasetDict:
    """Split data into `test_size`% of df in test, `valid`% of (1-test_size)% of df in valid, and the remaining in train.

    Return a DatasetDict[str, Dataset] of data splits.
    """
    # split data into 25% train, 25% valid, 50% test
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        shuffle=shuffle,
        stratify=df[label_column],
    )
    train, valid = train_test_split(
        train,
        test_size=valid_size,
        random_state=random_seed,
        shuffle=shuffle,
        stratify=train[label_column],
    )

    # create dataset dict
    dataset = DatasetDict()
    dataset["train"] = Dataset.from_pandas(train)
    dataset["valid"] = Dataset.from_pandas(valid)
    dataset["test"] = Dataset.from_pandas(test)

    # free memory
    del df, train, valid, test

    return dataset


def model_init(
    params: Optional[Dict[str, Any]] = None, model_name: str = DEFAUTL_PRETRAINED_MODEL
) -> SetFitModel:
    """
    Initialize a SetFit model with optional hyperparameter overrides from a trial or param dict.
    """
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    params = {
        "trust_remote_code": True,
        # "local_files_only": True,
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        },
    }
    # memory management
    gc.collect()
    torch.cuda.empty_cache()

    return SetFitModel.from_pretrained(model_name, **params)


def create_model(
    df: pd.DataFrame,
    project: str,
    num_shots: int,
    base_model: str,
    output_dir: str,
    seed: int,
) -> Trainer:
    """Created model."""
    dataset = create_dataset_split_dict(df, random_seed=seed)

    # sample n_shots from training dataset
    train_dataset = sample_dataset(
        dataset=dataset["train"],
        label_column=label_column,
        num_samples=num_shots,
        seed=seed,
    )
    valid_dataset = dataset["valid"]
    test_dataset = dataset["test"]

    # create trainer
    trainer = Trainer(
        model_init=partial(model_init, model_name=base_model),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        metric="f1",
        metric_kwargs={"average": "binary"},
        column_mapping={
            text_column: "text",
            label_column: "label",
        },
    )

    # hyperparameters tuning
    trainer.run_hp_search_optuna = hp_search.run_hp_search_optuna_updated  # type: ignore
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=partial(hp_search.hp_space, seed=seed),  # type: ignore
        n_trials=N_TRIALS_HPP_SEARCH,
    )

    # training with best hyperparameters
    trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    start_time = perf_counter()
    trainer.train()
    training_time = perf_counter() - start_time

    # prediction on unseen data
    X_test, y_test = test_dataset[text_column], test_dataset[label_column]
    y_pred = trainer.model.predict(X_test)

    # evaluation on in-project test data
    result = evaluator.compute_metrics(y_pred, y_test)
    result["random_seed"] = seed
    result["num_shots"] = num_shots
    result["training_time"] = training_time

    utils.to_csv([result], f"{output_dir}/{project}.csv")
    return trainer


def main(
    project: str,
    input_dataset: str,
    num_shots: int,
    base_model_name: str,
    output_dir: str,
    seed: int,
):
    """Train a model and save in output directory."""
    df = load_dataframe(input_file=input_dataset)
    trainer = create_model(df, project, num_shots, base_model_name, output_dir, seed)
    
    # Export setfit model
    model = trainer.model
    model.push_to_hub(
        f"ahenrij/slid-{project}",
        commit_message=f"Fine-tuned on {base_model_name} with {num_shots} shots.",
        private=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project")
    parser.add_argument(
        "-i",
        "--input-dataset",
        help="Input failed job dataset as .csv including columns of input `log`, `flaky` and `created_at`.",
    )
    parser.add_argument("-n", "--num-shots", default=DEFAULT_NUM_SHOTS)
    parser.add_argument("-s", "--seed", default=42)
    parser.add_argument("-o", "--output-dir")
    parser.add_argument(
        "-m",
        "--model-name",
        default=DEFAUTL_PRETRAINED_MODEL,
        help="Pretrained small language model.",
    )
    args = parser.parse_args()
    main(
        project=args.project,
        input_dataset=args.input_dataset,
        num_shots=int(args.num_shots),
        base_model_name=args.model_name,
        output_dir=args.output_dir,
        seed=int(args.seed),
    )
