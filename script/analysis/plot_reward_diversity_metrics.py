#!/usr/bin/env python3

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt


REWARD_METRICS = [
    ("grpo_reward", "GRPO reward"),
    ("grpo_terminal_reward", "Terminal reward"),
    ("grpo_format_reward", "Format reward"),
    ("grpo_reward_gain", "Reward gain"),
]

DIVERSITY_METRICS = [
    ("grpo_reward_std", "Reward std"),
    ("grpo_better_branch_rate", "Better-than-base rate"),
    ("grpo_branch_entropy", "Branch entropy"),
]
REQUIRED_KEYS = {"epoch"} | {key for key, _ in REWARD_METRICS + DIVERSITY_METRICS}


def parse_training_log(log_path: Path) -> list[dict]:
    records: list[dict] = []
    for raw_line in log_path.read_text().splitlines():
        line = raw_line.strip()
        if not line.startswith("{") or "'epoch'" not in line:
            continue
        try:
            record = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if isinstance(record, dict) and REQUIRED_KEYS.issubset(record):
            records.append(record)
    return records


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 2:
        return values[:]

    averaged: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        segment = values[start : idx + 1]
        averaged.append(sum(segment) / len(segment))
    return averaged


def plot_metrics(records: list[dict], output_path: Path, smooth_window: int) -> None:
    if not records:
        raise ValueError("No training metric records were found in the log.")

    epochs = [float(record["epoch"]) for record in records]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for axis, metric_group, title in [
        (axes[0], REWARD_METRICS, "Reward Metrics"),
        (axes[1], DIVERSITY_METRICS, "Diversity Metrics"),
    ]:
        for key, label in metric_group:
            values = [float(record[key]) for record in records]
            smoothed = moving_average(values, smooth_window)
            axis.plot(epochs, values, alpha=0.25, linewidth=1.2)
            axis.plot(epochs, smoothed, label=label, linewidth=2.0)

        axis.set_title(title)
        axis.set_ylabel("Value")
        axis.legend(loc="best")

    axes[1].set_xlabel("Epoch")
    fig.suptitle(f"Reward and Diversity Metrics\n{output_path.stem}", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward and diversity metrics from an SDAR training log.")
    parser.add_argument("log_path", type=Path, help="Path to the training log file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <log_dir>/<log_stem>_reward_diversity.png",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average window for the main plotted curves.",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = args.log_path.with_name(f"{args.log_path.stem}_reward_diversity.png")

    records = parse_training_log(args.log_path)
    plot_metrics(records, output_path, max(1, args.smooth_window))
    print(f"Saved plot to: {output_path}")
    print(f"Parsed records: {len(records)}")


if __name__ == "__main__":
    main()
