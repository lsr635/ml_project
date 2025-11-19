#!/usr/bin/env python
"""Collect test-set metrics from experiment logs into a single table."""

import argparse
import glob
import os
import re
from typing import Dict, List

import pandas as pd

REPORT_PATTERN = re.compile(
    r"Test Set Performance:\s*=+\s*(.*?)\n\nConfusion Matrix:",
    re.DOTALL,
)


def parse_report(report_text: str) -> Dict[str, float]:
    """Parse the classification report block into a flat metrics dict."""
    metrics: Dict[str, float] = {}
    lines = [line.strip() for line in report_text.strip().splitlines() if line.strip()]

    # Skip header row if present
    if lines and lines[0].startswith("precision"):
        lines = lines[1:]

    for line in lines:
        parts = line.split()
        if not parts:
            continue

        label_tokens: List[str]

        # accuracy line has only accuracy + support
        if parts[0].lower() == "accuracy":
            if len(parts) >= 3:
                metrics["accuracy"] = float(parts[1])
                metrics["accuracy_support"] = float(parts[2])
            continue

        if len(parts) < 5:
            continue

        if len(parts) > 5:
            label_tokens = parts[:-4]
            metric_tokens = parts[-4:]
        else:
            label_tokens = [parts[0]]
            metric_tokens = parts[1:5]

        label = "_".join(token.lower() for token in label_tokens)

        try:
            precision, recall, f1, support = map(float, metric_tokens)
        except ValueError:
            continue

        metrics[f"{label}_precision"] = precision
        metrics[f"{label}_recall"] = recall
        metrics[f"{label}_f1"] = f1
        metrics[f"{label}_support"] = support

    return metrics


def collect_metrics(log_paths: List[str]) -> pd.DataFrame:
    rows = []

    for log_path in log_paths:
        with open(log_path, "r", encoding="utf-8") as f:
            text = f.read()

        match = REPORT_PATTERN.search(text)
        if not match:
            print(f"Warning: no test report found in {log_path}")
            continue

        report_text = match.group(1)
        metrics = parse_report(report_text)
        if not metrics:
            print(f"Warning: unable to parse metrics in {log_path}")
            continue

        metrics["experiment"] = os.path.splitext(os.path.basename(log_path))[0]
        rows.append(metrics)

    if not rows:
        raise RuntimeError("No metrics parsed from the provided logs")

    df = pd.DataFrame(rows)
    df.sort_values("experiment", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect test metrics from logs")
    parser.add_argument(
        "--logs",
        nargs="*",
        help="Explicit list of log files to parse",
    )
    parser.add_argument(
        "--glob",
        default=os.path.join("logs", "exp*.log"),
        help="Glob pattern used when --logs is not given",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("logs", "test_metrics_summary.csv"),
        help="Path to save the aggregated metrics table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.logs:
        log_paths = sorted(args.logs)
    else:
        log_paths = sorted(glob.glob(args.glob))

    if not log_paths:
        raise FileNotFoundError("No log files provided or found via glob pattern")

    df = collect_metrics(log_paths)
    df.to_csv(args.output, index=False)
    print(f"Saved metrics table to {args.output}")
    print("\nPreview:\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
