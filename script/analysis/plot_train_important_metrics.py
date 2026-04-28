#!/usr/bin/env python3

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_GROUPS = [
    (
        "Core Losses",
        [
            ("loss", "Total loss"),
            ("weighted_diffusion_loss", "Weighted diffusion loss"),
            ("remask_loss", "Remask loss"),
        ],
    ),
    (
        "Remask Quality",
        [
            ("remask_f1", "Remask F1"),
            ("remask_precision", "Remask precision"),
            ("remask_recall", "Remask recall"),
        ],
    ),
    (
        "GRPO Rewards",
        [
            ("grpo_reward", "GRPO reward"),
            ("grpo_terminal_reward", "Terminal reward"),
            ("grpo_reward_gain", "Reward gain"),
        ],
    ),
    (
        "Branch Policy",
        [
            ("grpo_branch_entropy", "Branch entropy"),
            ("grpo_better_branch_rate", "Better-than-base rate"),
            ("grpo_reward_std", "Reward std"),
        ],
    ),
]

REQUIRED_KEYS = {"epoch"} | {
    key for _, metric_group in METRIC_GROUPS for key, _ in metric_group
}


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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for axis, (title, metric_group) in zip(axes, METRIC_GROUPS):
        for key, label in metric_group:
            values = [float(record[key]) for record in records]
            smoothed = moving_average(values, smooth_window)
            axis.plot(epochs, values, alpha=0.2, linewidth=1.0)
            axis.plot(epochs, smoothed, label=label, linewidth=2.0)

        axis.set_title(title)
        axis.set_ylabel("Value")
        axis.legend(loc="best")

    for axis in axes[2:]:
        axis.set_xlabel("Epoch")

    fig.suptitle(f"Important Training Metrics\n{output_path.stem}", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot important SDAR training metrics from a training log."
    )
    parser.add_argument("log_path", type=Path, help="Path to the training log file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <log_dir>/<log_stem>_important_metrics.png",
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
        output_path = args.log_path.with_name(f"{args.log_path.stem}_important_metrics.png")

    records = parse_training_log(args.log_path)
    plot_metrics(records, output_path, max(1, args.smooth_window))
    print(f"Saved plot to: {output_path}")
    print(f"Parsed records: {len(records)}")


if __name__ == "__main__":
    main()
