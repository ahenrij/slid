import argparse
from pathlib import Path
from time import perf_counter

import evaluate
import numpy as np
import torch
from setfit import SetFitModel
from tqdm.auto import tqdm

from src.slid import utils
from src.slid.modeling.run import create_dataset_split_dict, load_dataframe

metric = evaluate.load("accuracy")


class PerformanceBenchmark:
    def __init__(self, model, dataset, optim_type):
        self.model = model
        self.dataset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self):
        preds = self.model.predict(self.dataset["text"])
        labels = self.dataset["label"]
        accuracy = metric.compute(predictions=preds, references=labels)
        print(f"Accuracy on test set - {accuracy['accuracy']:.3f}")
        return accuracy

    def compute_size(self):
        state_dict = self.model.model_body.state_dict()
        tmp_path = Path("model.pt")
        torch.save(state_dict, tmp_path)
        # Calculate size in megabytes
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        # Delete temporary file
        tmp_path.unlink()
        print(f"Model size (MB) - {size_mb:.2f}")
        return {"size_mb": size_mb}

    def time_model(self, query="that loves its characters and communicates something rather beautiful about human nature"):
        latencies = []
        # Warmup
        for _ in range(10):
            _ = self.model([query])
        # Timed run
        for _ in range(100):
            start_time = perf_counter()
            _ = self.model([query])
            latency = perf_counter() - start_time
            latencies.append(latency)
        # Compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        print(rf"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.compute_accuracy())
        metrics[self.optim_type].update(self.time_model())
        return metrics


def main(
    project: str,
    input_dataset: str,
    output_dir: str,
    seed: int,
):
    df = load_dataframe(input_file=input_dataset)

    dataset = create_dataset_split_dict(df, random_seed=seed)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    small_model = SetFitModel.from_pretrained(f"ahenrij/slid-{project}")
    pb = PerformanceBenchmark(model=small_model, dataset=test_dataset, optim_type="veloren (PyTorch)")
    perf_metrics = pb.run_benchmark()
    utils.to_csv([perf_metrics], f"{output_dir}/performance.csv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project")
    parser.add_argument(
        "-i",
        "--input-dataset",
        help="Input failed job dataset as .csv including columns of input `log`, `flaky` and `created_at`.",
    )
    parser.add_argument("-s", "--seed", default=42)
    parser.add_argument("-o", "--output-dir")
    args = parser.parse_args()
    main(
        project=args.project,
        input_dataset=args.input_dataset,
        num_shots=int(args.num_shots),
        base_model_name=args.model_name,
        output_dir=args.output_dir,
        seed=int(args.seed),
    )
