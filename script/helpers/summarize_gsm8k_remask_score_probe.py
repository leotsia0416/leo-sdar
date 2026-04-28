#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def percentile(values: list[float], q: float):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def fmt(value):
    if value is None:
        return 'NA'
    if isinstance(value, float):
        return f'{value:.6f}'
    if isinstance(value, bool):
        return '1' if value else '0'
    return str(value)


def safe_ratio(num: int, den: int):
    if den <= 0:
        return None
    return num / den


def extract_accuracy(results_path: Path):
    if not results_path.is_file():
        return None
    payload = json.loads(results_path.read_text(encoding='utf-8'))
    accuracy = payload.get('accuracy')
    return float(accuracy) if accuracy is not None else None


def build_summary(args):
    records = load_jsonl(args.event_trace)
    active_records = [record for record in records if record.get('remask_active')]
    active_candidate_records = [
        record for record in active_records if int(record.get('candidate_count', 0)) > 0
    ]
    triggered_records = [record for record in active_candidate_records if record.get('triggered')]
    best_scores = [
        float(record['best_score'])
        for record in active_candidate_records
        if record.get('best_score') is not None
    ]
    triggered_scores = [
        float(record['best_score'])
        for record in triggered_records
        if record.get('best_score') is not None
    ]
    score_margins = [
        float(record['score_margin'])
        for record in active_candidate_records
        if record.get('score_margin') is not None
    ]

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    threshold_rates = {}
    for threshold in thresholds:
        threshold_rates[threshold] = safe_ratio(
            sum(score >= threshold for score in best_scores),
            len(best_scores),
        )

    row = {
        'label': args.label,
        'subset_size': args.subset_size,
        'accuracy': extract_accuracy(args.results),
        'event_checks': len(records),
        'active_checks': len(active_records),
        'active_candidate_checks': len(active_candidate_records),
        'triggered_checks': len(triggered_records),
        'trigger_rate_active_candidates': safe_ratio(
            len(triggered_records),
            len(active_candidate_records),
        ),
        'best_score_p10': percentile(best_scores, 0.10),
        'best_score_p25': percentile(best_scores, 0.25),
        'best_score_p50': percentile(best_scores, 0.50),
        'best_score_p75': percentile(best_scores, 0.75),
        'best_score_p90': percentile(best_scores, 0.90),
        'best_score_p95': percentile(best_scores, 0.95),
        'best_score_min': min(best_scores) if best_scores else None,
        'best_score_max': max(best_scores) if best_scores else None,
        'triggered_best_score_p50': percentile(triggered_scores, 0.50),
        'triggered_best_score_p90': percentile(triggered_scores, 0.90),
        'score_margin_p50': percentile(score_margins, 0.50),
        'score_margin_p90': percentile(score_margins, 0.90),
    }
    for threshold, rate in threshold_rates.items():
        row[f'would_trigger_rate_at_{threshold:.2f}'] = rate
    return row


def print_text_summary(row: dict):
    print(f"label={row['label']}")
    print(f"subset_size={row['subset_size']}")
    print(f"accuracy={fmt(row['accuracy'])}")
    print(f"event_checks={row['event_checks']}")
    print(f"active_checks={row['active_checks']}")
    print(f"active_candidate_checks={row['active_candidate_checks']}")
    print(f"triggered_checks={row['triggered_checks']}")
    print(
        "trigger_rate_active_candidates="
        f"{fmt(row['trigger_rate_active_candidates'])}"
    )
    print(
        "best_score_active_candidates="
        f"p10={fmt(row['best_score_p10'])} "
        f"p25={fmt(row['best_score_p25'])} "
        f"p50={fmt(row['best_score_p50'])} "
        f"p75={fmt(row['best_score_p75'])} "
        f"p90={fmt(row['best_score_p90'])} "
        f"p95={fmt(row['best_score_p95'])} "
        f"min={fmt(row['best_score_min'])} "
        f"max={fmt(row['best_score_max'])}"
    )
    print(
        "best_score_triggered="
        f"p50={fmt(row['triggered_best_score_p50'])} "
        f"p90={fmt(row['triggered_best_score_p90'])}"
    )
    print(
        "score_margin_active_candidates="
        f"p50={fmt(row['score_margin_p50'])} "
        f"p90={fmt(row['score_margin_p90'])}"
    )
    print("would_trigger_rate_by_threshold")
    for key in sorted(k for k in row if k.startswith('would_trigger_rate_at_')):
        print(f"  {key.split('_at_')[1]}: {fmt(row[key])}")


def main():
    parser = argparse.ArgumentParser(
        description='Summarize GSM8K remask score distributions from an event trace.'
    )
    parser.add_argument('--label', required=True)
    parser.add_argument('--subset-size', type=int, required=True)
    parser.add_argument('--event-trace', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--tsv', action='store_true')
    args = parser.parse_args()

    row = build_summary(args)
    ordered_columns = [
        'label',
        'subset_size',
        'accuracy',
        'event_checks',
        'active_checks',
        'active_candidate_checks',
        'triggered_checks',
        'trigger_rate_active_candidates',
        'best_score_p10',
        'best_score_p25',
        'best_score_p50',
        'best_score_p75',
        'best_score_p90',
        'best_score_p95',
        'best_score_min',
        'best_score_max',
        'triggered_best_score_p50',
        'triggered_best_score_p90',
        'score_margin_p50',
        'score_margin_p90',
        'would_trigger_rate_at_0.30',
        'would_trigger_rate_at_0.40',
        'would_trigger_rate_at_0.50',
        'would_trigger_rate_at_0.60',
        'would_trigger_rate_at_0.70',
        'would_trigger_rate_at_0.80',
        'would_trigger_rate_at_0.90',
    ]

    if args.tsv:
        print('\t'.join(ordered_columns))
        print('\t'.join(fmt(row.get(column)) for column in ordered_columns))
        return

    print_text_summary(row)


if __name__ == '__main__':
    main()
