#!/usr/bin/env python3

import argparse
import json
from bisect import bisect_left
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def load_best_scores(trace_path: Path):
    event_checks = 0
    active_checks = 0
    active_candidate_checks = 0
    triggered_checks = 0
    best_scores = []

    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            event_checks += 1
            if not record.get("remask_active"):
                continue
            active_checks += 1
            if int(record.get("candidate_count", 0)) <= 0:
                continue
            active_candidate_checks += 1
            if record.get("triggered"):
                triggered_checks += 1
            best_score = record.get("best_score")
            if best_score is not None:
                best_scores.append(float(best_score))

    return {
        "event_checks": event_checks,
        "active_checks": active_checks,
        "active_candidate_checks": active_candidate_checks,
        "triggered_checks": triggered_checks,
        "best_scores": np.array(best_scores, dtype=float),
    }


def build_trigger_curve(scores: np.ndarray, num_points: int = 201):
    thresholds = np.linspace(0.0, 1.0, num_points)
    if scores.size == 0:
        return thresholds, np.zeros_like(thresholds)

    sorted_scores = np.sort(scores)
    sorted_list = sorted_scores.tolist()
    rates = []
    total = sorted_scores.size
    for threshold in thresholds:
        idx = bisect_left(sorted_list, threshold)
        rates.append((total - idx) / total)
    return thresholds, np.array(rates, dtype=float)


def add_threshold_markers(ax, thresholds, rates_lookup):
    colors = {
        0.30: "#c1121f",
        0.50: "#f77f00",
        0.80: "#3a86ff",
        0.90: "#6a4c93",
    }
    for threshold in thresholds:
        color = colors.get(threshold, "#444444")
        ax.axvline(threshold, color=color, linestyle="--", linewidth=1.5, alpha=0.9)
        rate = rates_lookup.get(threshold)
        if rate is not None:
            ax.text(
                threshold,
                0.98,
                f"{threshold:.2f}\n{rate:.1%}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.85),
            )


def main():
    parser = argparse.ArgumentParser(
        description="Plot GSM8K remask score distribution and threshold-trigger curve."
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="GSM8K ReMask Score Distribution")
    args = parser.parse_args()

    stats = load_best_scores(args.trace)
    scores = stats["best_scores"]
    thresholds_to_mark = [0.30, 0.50, 0.80, 0.90]
    curve_thresholds, trigger_rates = build_trigger_curve(scores)

    rate_lookup = {}
    if scores.size:
        sorted_scores = np.sort(scores)
        total = sorted_scores.size
        as_list = sorted_scores.tolist()
        for threshold in thresholds_to_mark:
            idx = bisect_left(as_list, threshold)
            rate_lookup[threshold] = (total - idx) / total
    else:
        rate_lookup = {threshold: 0.0 for threshold in thresholds_to_mark}

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)

    hist_ax = axes[0]
    if scores.size:
        hist_ax.hist(
            scores,
            bins=np.linspace(0.0, 1.0, 41),
            color="#5b8e7d",
            edgecolor="white",
            alpha=0.9,
        )
    hist_ax.set_title(args.title)
    hist_ax.set_xlabel("Best ReMask Score in Active Candidate Check")
    hist_ax.set_ylabel("Checks")
    hist_ax.set_xlim(0.0, 1.0)
    hist_ax.grid(alpha=0.2, axis="y")
    add_threshold_markers(hist_ax, thresholds_to_mark, rate_lookup)

    summary_lines = [
        f"event checks: {stats['event_checks']:,}",
        f"active checks: {stats['active_checks']:,}",
        f"active candidate checks: {stats['active_candidate_checks']:,}",
        f"triggered @ current threshold: {stats['triggered_checks']:,}",
        f"trigger rate | active candidates: {stats['triggered_checks'] / stats['active_candidate_checks']:.1%}"
        if stats["active_candidate_checks"]
        else "trigger rate | active candidates: NA",
        f"score p50 / p90: {fmt(percentile(scores, 0.5))} / {fmt(percentile(scores, 0.9))}",
    ]
    hist_ax.text(
        0.015,
        0.98,
        "\n".join(summary_lines),
        transform=hist_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#bbbbbb", alpha=0.92),
    )

    curve_ax = axes[1]
    curve_ax.plot(curve_thresholds, trigger_rates, color="#1d3557", linewidth=2.5)
    curve_ax.fill_between(curve_thresholds, trigger_rates, color="#a8dadc", alpha=0.35)
    curve_ax.set_xlabel("ReMask Threshold")
    curve_ax.set_ylabel("Would-Trigger Rate")
    curve_ax.set_xlim(0.0, 1.0)
    curve_ax.set_ylim(0.0, 1.02)
    curve_ax.grid(alpha=0.25)
    for threshold in thresholds_to_mark:
        rate = rate_lookup[threshold]
        curve_ax.scatter([threshold], [rate], color="#c1121f" if threshold == 0.30 else "#1d3557", zorder=3)
        curve_ax.text(
            threshold,
            min(rate + 0.045, 0.99),
            f"{threshold:.2f}: {rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )
    curve_ax.set_title("Estimated Trigger Rate if Threshold Is Changed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"output={args.output}")
    print(f"event_checks={stats['event_checks']}")
    print(f"active_checks={stats['active_checks']}")
    print(f"active_candidate_checks={stats['active_candidate_checks']}")
    print(f"triggered_checks={stats['triggered_checks']}")
    if stats["active_candidate_checks"]:
        print(
            "trigger_rate_active_candidates="
            f"{stats['triggered_checks'] / stats['active_candidate_checks']:.6f}"
        )
    print(
        "best_score_quantiles="
        f"p10={fmt(percentile(scores, 0.10))} "
        f"p25={fmt(percentile(scores, 0.25))} "
        f"p50={fmt(percentile(scores, 0.50))} "
        f"p75={fmt(percentile(scores, 0.75))} "
        f"p90={fmt(percentile(scores, 0.90))} "
        f"p95={fmt(percentile(scores, 0.95))}"
    )
    for threshold in thresholds_to_mark:
        print(f"would_trigger_rate_at_{threshold:.2f}={rate_lookup[threshold]:.6f}")


if __name__ == "__main__":
    main()
