#!/usr/bin/env python3

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt


LINE_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]

METRIC_YLABELS = {
    "grpo_remask_rate": "Remask rate",
    "grpo_branch_remask_span": "Rate span",
    "grpo_better_branch_rate": "Better-than-base rate",
    "grpo_terminal_reward": "Terminal reward",
    "grpo_format_reward": "Format reward",
    "grpo_reward": "Avg reward",
    "grpo_reward_gain": "Reward gain",
    "grpo_reward_std": "Reward std",
    "grpo_branch_entropy": "Entropy",
    "grpo_branch_count": "Branch count",
    "grpo_candidate_block_count": "Candidate blocks",
    "grpo_rollout_depth": "Rollout depth",
    "remask_pred_rate": "Pred rate",
    "remask_precision": "Precision",
    "remask_recall": "Recall",
    "remask_f1": "F1",
}


METRIC_GROUPS = [
    (
        "behavior",
        "Actual Remask Behavior",
        [
            ("grpo_remask_rate", "rollout remask rate"),
            ("grpo_branch_remask_span", "branch remask span"),
            ("grpo_branch_entropy", "branch entropy"),
        ],
    ),
    (
        "coverage",
        "Branch / Candidate Coverage",
        [
            ("grpo_branch_count", "branch count"),
            ("grpo_candidate_block_count", "candidate block count"),
            ("grpo_rollout_depth", "rollout depth"),
        ],
    ),
    (
        "supervision",
        "Remask Head Supervision",
        [
            ("remask_pred_rate", "pred rate"),
            ("remask_precision", "precision"),
            ("remask_recall", "recall"),
            ("remask_f1", "f1"),
        ],
    ),
    (
        "utility",
        "Remask Branch Utility",
        [
            ("grpo_reward_gain", "reward gain"),
            ("grpo_better_branch_rate", "better-than-base rate"),
            ("grpo_reward_std", "reward std"),
        ],
    ),
    (
        "rate_semantic",
        "Rate Metrics",
        [
            ("grpo_remask_rate", "rollout remask rate"),
            ("grpo_branch_remask_span", "branch remask span"),
            ("grpo_better_branch_rate", "better-than-base rate"),
        ],
    ),
    (
        "reward_scale",
        "Reward Scale",
        [
            ("grpo_reward_gain", "reward gain"),
            ("grpo_reward_std", "reward std"),
        ],
    ),
    (
        "terminal_only",
        "Terminal Reward",
        [
            ("grpo_terminal_reward", "terminal reward"),
        ],
    ),
    (
        "entropy_only",
        "Branch Entropy",
        [
            ("grpo_branch_entropy", "branch entropy"),
        ],
    ),
]


SCHEDULE_LINES = [
    (0.301 * 3.0, "thr=0.50"),
    (0.331 * 3.0, "remask loss=0.02"),
    (0.672 * 3.0, "remask loss=0"),
    (0.701 * 3.0, "thr=0.65"),
]


def color_for(index: int) -> str:
    return LINE_COLORS[index % len(LINE_COLORS)]


def ylabel_for(metric_key: str) -> str:
    return METRIC_YLABELS.get(metric_key, "Value")


def flatten_metric_specs(metric_groups):
    metric_specs = []
    for _, group_title, metric_group in metric_groups:
        for key, label in metric_group:
            metric_specs.append((group_title, key, label))
    return metric_specs


def plot_metric_axis(
    axis,
    records: list[dict],
    group_title: str,
    key: str,
    label: str,
    smooth_window: int,
    show_schedule: bool,
    color_index: int,
) -> bool:
    xs, ys = series_for(records, key)
    if ys:
        smoothed = moving_average(ys, smooth_window)
        color = color_for(color_index)
        axis.plot(xs, ys, alpha=0.22, linewidth=1.0, color=color)
        axis.plot(xs, smoothed, linewidth=2.0, color=color, label=label)
        axis.legend(loc="best", fontsize=9)

    axis.set_title(f"{group_title}: {label}")
    axis.set_ylabel(ylabel_for(key))
    if show_schedule:
        draw_schedule_lines(axis)
    axis.set_xlabel("Epoch")
    return bool(ys)


def parse_training_log(log_path: Path) -> list[dict]:
    records: list[dict] = []
    for raw_line in log_path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line.startswith("{") or "'epoch'" not in line:
            continue
        try:
            record = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if isinstance(record, dict) and "epoch" in record:
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


def series_for(records: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for record in records:
        if key not in record:
            continue
        try:
            xs.append(float(record["epoch"]))
            ys.append(float(record[key]))
        except (TypeError, ValueError):
            continue
    return xs, ys


def draw_schedule_lines(axis) -> None:
    ymin, ymax = axis.get_ylim()
    for x, label in SCHEDULE_LINES:
        axis.axvline(x, color="#7a7a7a", linestyle="--", linewidth=0.9, alpha=0.45)
        axis.text(
            x,
            ymax,
            label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            color="#555555",
            alpha=0.85,
        )
    axis.set_ylim(ymin, ymax)


def select_metric_groups(group_ids: list[str] | None):
    if not group_ids:
        return METRIC_GROUPS

    by_id = {group_id: group for group_id, *group in METRIC_GROUPS}
    selected = []
    for group_id in group_ids:
        if group_id not in by_id:
            valid = ", ".join(by_id)
            raise ValueError(f"Unknown group '{group_id}'. Valid groups: {valid}")
        title, metrics = by_id[group_id]
        selected.append((group_id, title, metrics))
    return selected


def plot_metrics(
    records: list[dict],
    output_path: Path,
    smooth_window: int,
    metric_groups,
    layout: str,
    show_schedule: bool,
    split_metrics: bool,
    separate_files: bool,
) -> int:
    if not records:
        raise ValueError("No training metric records were found in the log.")

    plt.style.use("seaborn-v0_8-whitegrid")
    metric_specs = flatten_metric_specs(metric_groups)
    if not metric_specs:
        raise ValueError("No metric groups were selected.")

    if separate_files:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        saved_count = 0
        for idx, (group_title, key, label) in enumerate(metric_specs):
            fig, axis = plt.subplots(1, 1, figsize=(9.5, 4.8), sharex=True)
            plotted = plot_metric_axis(
                axis,
                records,
                group_title,
                key,
                label,
                smooth_window,
                show_schedule,
                idx,
            )
            if not plotted:
                plt.close(fig)
                continue
            fig.tight_layout()
            metric_output_path = output_path.with_name(f"{output_path.stem}_{key}{output_path.suffix}")
            fig.savefig(metric_output_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved_count += 1

        if saved_count == 0:
            raise ValueError("No remask metric records were found in the log.")
        return saved_count

    if split_metrics:
        num_metrics = len(metric_specs)
        if layout == "vertical":
            nrows, ncols = num_metrics, 1
            figsize = (11, max(3.2 * num_metrics, 4.5))
        elif layout == "horizontal":
            nrows, ncols = 1, num_metrics
            figsize = (max(4.8 * num_metrics, 8), 4.8)
        else:
            ncols = 1 if num_metrics == 1 else 2
            nrows = (num_metrics + ncols - 1) // ncols
            figsize = (15 if ncols == 2 else 8.5, max(3.6 * nrows, 4.8))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        if hasattr(axes, "flatten"):
            axes = axes.flatten().tolist()
        else:
            axes = [axes]

        plotted_any = False
        for idx, (axis, (group_title, key, label)) in enumerate(zip(axes, metric_specs)):
            plotted = plot_metric_axis(
                axis,
                records,
                group_title,
                key,
                label,
                smooth_window,
                show_schedule,
                idx,
            )
            plotted_any = plotted_any or plotted

        for axis in axes[num_metrics:]:
            axis.remove()
    else:
        if len(metric_groups) == 1:
            fig, axes = plt.subplots(1, 1, figsize=(8, 5), sharex=True)
            axes = [axes]
        elif len(metric_groups) == 2 and layout == "vertical":
            fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
            axes = axes.flatten()
        elif len(metric_groups) == 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
            axes = axes.flatten()

        plotted_any = False
        for axis, (_, title, metric_group) in zip(axes, metric_groups):
            group_plotted = False
            for idx, (key, label) in enumerate(metric_group):
                xs, ys = series_for(records, key)
                if not ys:
                    continue
                smoothed = moving_average(ys, smooth_window)
                color = color_for(idx)
                axis.plot(xs, ys, alpha=0.22, linewidth=1.0, color=color)
                axis.plot(xs, smoothed, label=label, linewidth=2.0, color=color)
                group_plotted = True
                plotted_any = True

            axis.set_title(title)
            axis.set_ylabel("Value")
            if group_plotted:
                axis.legend(loc="best", fontsize=9)
            if show_schedule:
                draw_schedule_lines(axis)
            axis.set_xlabel("Epoch")

    if not plotted_any:
        raise ValueError("No remask metric records were found in the log.")

    fig.suptitle(f"REMASK Training Metrics: {output_path.stem}", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot REMASK-specific SDAR training metrics from a training log."
    )
    parser.add_argument("log_path", type=Path, help="Path to the training log file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <log_dir>/<log_stem>_remask_metrics.png",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average window for the main plotted curves.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=[group_id for group_id, *_ in METRIC_GROUPS],
        default=None,
        help="Metric groups to plot. Defaults to all groups.",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "horizontal", "vertical"],
        default="auto",
        help="Subplot layout for selected groups. Defaults to horizontal for two groups.",
    )
    parser.add_argument(
        "--hide-schedule",
        action="store_true",
        help="Hide vertical schedule marker lines.",
    )
    parser.add_argument(
        "--split-metrics",
        action="store_true",
        help="Plot each metric on its own subplot instead of overlaying metrics within a group.",
    )
    parser.add_argument(
        "--separate-files",
        action="store_true",
        help="Save each metric as its own image file using the output stem as a prefix.",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = args.log_path.with_name(f"{args.log_path.stem}_remask_metrics.png")

    records = parse_training_log(args.log_path)
    metric_groups = select_metric_groups(args.groups)
    saved_count = plot_metrics(
        records,
        output_path,
        max(1, args.smooth_window),
        metric_groups,
        args.layout,
        not args.hide_schedule,
        args.split_metrics,
        args.separate_files,
    )
    if args.separate_files:
        print(f"Saved {saved_count} plots with prefix: {output_path}")
    else:
        print(f"Saved plot to: {output_path}")
    print(f"Parsed records: {len(records)}")


if __name__ == "__main__":
    main()
