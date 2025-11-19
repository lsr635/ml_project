#!/usr/bin/env python
"""Utilities to parse training logs and generate comparison plots."""

import argparse
import glob
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_log(log_path: str) -> List[Dict[str, float]]:
    """Parse a training log and return epoch level metrics."""
    epoch_data: List[Dict[str, float]] = []
    current_epoch: Dict[str, float] = {}

    epoch_pattern = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
    metric_pattern = re.compile(
        r"Training Loss:\s*([0-9.]+)\s*[|]\s*Training Accuracy:\s*([0-9.]+)\s*\n"
        r"Validation Loss:\s*([0-9.]+)\s*[|]\s*Validation Accuracy:\s*([0-9.]+)"
    )

    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    epochs = epoch_pattern.findall(text)
    metrics = metric_pattern.findall(text)

    if not epochs or not metrics:
        print(f"Warning: no epoch metrics found in {log_path}")
        return epoch_data

    if len(epochs) != len(metrics):
        print(
            f"Warning: mismatch between epochs ({len(epochs)}) "
            f"and metric blocks ({len(metrics)}) in {log_path}"
        )

    for (epoch_idx, epoch_total), metric_values in zip(epochs, metrics):
        train_loss, train_acc, val_loss, val_acc = map(float, metric_values)
        epoch_data.append(
            {
                "epoch": int(epoch_idx),
                "epochs_total": int(epoch_total),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    return epoch_data


def ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_metric(
    all_runs: Dict[str, List[Dict[str, float]]],
    metric_key: str,
    ylabel: str,
    filename: str,
    output_dir: str,
) -> None:
    ensure_output_dir(output_dir)
    plt.figure(figsize=(10, 6))

    for run_name, data in sorted(all_runs.items()):
        epochs = [item["epoch"] for item in data]
        metric_values = [item[metric_key] for item in data]
        plt.plot(epochs, metric_values, marker="o", linewidth=1.5, label=run_name)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} per Epoch")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved {output_path}")


def plot_final_metric(
    all_runs: Dict[str, List[Dict[str, float]]],
    metric_key: str,
    ylabel: str,
    filename: str,
    output_dir: str,
) -> None:
    ensure_output_dir(output_dir)
    labels = []
    values = []

    for run_name, data in sorted(all_runs.items()):
        if not data:
            continue
        labels.append(run_name)
        values.append(data[-1][metric_key])

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color="steelblue")
    plt.ylabel(ylabel)
    plt.title(f"Final {ylabel} per Experiment")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}",
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved {output_path}")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics from training logs")
    parser.add_argument(
        "--logs",
        nargs="*",
        help="Specific log files to parse (overrides --glob)",
    )
    parser.add_argument(
        "--glob",
        default=os.path.join("logs", "exp*.log"),
        help="Glob pattern to find log files when --logs is not provided",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("logs", "figures"),
        help="Directory to save generated plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli()

    if args.logs:
        log_files = sorted(args.logs)
    else:
        log_files = sorted(glob.glob(args.glob))

    if not log_files:
        raise FileNotFoundError("No log files provided or found via glob pattern")

    all_runs: Dict[str, List[Dict[str, float]]] = {}
    for log_path in log_files:
        run_name = os.path.splitext(os.path.basename(log_path))[0]
        epoch_data = parse_log(log_path)
        if epoch_data:
            all_runs[run_name] = epoch_data
            print(f"Parsed {len(epoch_data)} epochs from {log_path}")
        else:
            print(f"Skipped {log_path}, no epoch data parsed")

    if not all_runs:
        raise RuntimeError("No runs with metrics were parsed")

    # Line plots per epoch
    plot_metric(
        all_runs,
        "train_loss",
        "Training Loss",
        "training_loss_per_epoch.png",
        args.output_dir,
    )
    plot_metric(
        all_runs,
        "val_loss",
        "Validation Loss",
        "validation_loss_per_epoch.png",
        args.output_dir,
    )
    plot_metric(
        all_runs,
        "train_acc",
        "Training Accuracy",
        "training_accuracy_per_epoch.png",
        args.output_dir,
    )
    plot_metric(
        all_runs,
        "val_acc",
        "Validation Accuracy",
        "validation_accuracy_per_epoch.png",
        args.output_dir,
    )

    # Bar charts for final metrics
    plot_final_metric(
        all_runs,
        "val_acc",
        "Validation Accuracy",
        "final_validation_accuracy.png",
        args.output_dir,
    )
    plot_final_metric(
        all_runs,
        "val_loss",
        "Validation Loss",
        "final_validation_loss.png",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
