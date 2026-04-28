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


def extract_correct_flag(result_payload: dict):
    details = result_payload.get('details') or []
    if not details:
        return None
    correct = details[0].get('correct')
    if isinstance(correct, list):
        return bool(correct[0]) if correct else None
    if isinstance(correct, bool):
        return correct
    return None


def prompt_preview(trace_records: list[dict], event_records: list[dict]) -> str:
    source_records = trace_records or event_records
    if not source_records:
        return ''
    input_payload = source_records[0].get('input')
    if isinstance(input_payload, list) and input_payload:
        prompt = input_payload[0].get('prompt', '')
        if prompt:
            return prompt.splitlines()[0].replace('\t', ' ').strip()
    return ''


def build_row(args) -> list:
    trace_records = load_jsonl(args.trace)
    event_records = load_jsonl(args.event_trace)

    trace_record = trace_records[0] if trace_records else {}
    active_records = [record for record in event_records if record.get('remask_active')]
    candidate_records = [record for record in event_records if int(record.get('candidate_count', 0)) > 0]
    triggered_records = [record for record in event_records if record.get('triggered')]
    active_candidate_records = [
        record for record in active_records if int(record.get('candidate_count', 0)) > 0
    ]

    result_payload = {}
    if args.results.is_file():
        result_payload = json.loads(args.results.read_text(encoding='utf-8'))

    accuracy = result_payload.get('accuracy')
    correct = extract_correct_flag(result_payload)

    best_scores = [
        float(record['best_score'])
        for record in event_records
        if record.get('best_score') is not None
    ]
    score_margins = [
        float(record['score_margin'])
        for record in event_records
        if record.get('score_margin') is not None
    ]
    remasked_tokens_triggered = [
        int(record.get('remasked_tokens', 0))
        for record in triggered_records
    ]

    steps_with_remask = trace_record.get('steps_with_remask')
    total_remasked_tokens = trace_record.get('total_remasked_tokens')
    first_remask_block = event_records[0].get('first_remask_block') if event_records else None

    return [
        args.label,
        args.threshold,
        accuracy,
        correct,
        len(event_records),
        len(active_records),
        len(candidate_records),
        len(active_candidate_records),
        len(triggered_records),
        safe_ratio(len(triggered_records), len(event_records)),
        safe_ratio(len(triggered_records), len(active_records)),
        safe_ratio(len(triggered_records), len(active_candidate_records)),
        steps_with_remask,
        total_remasked_tokens,
        sum(remasked_tokens_triggered),
        safe_ratio(sum(remasked_tokens_triggered), len(triggered_records)),
        max(best_scores) if best_scores else None,
        max(score_margins) if score_margins else None,
        first_remask_block,
        prompt_preview(trace_records, event_records),
    ]


def main():
    parser = argparse.ArgumentParser(
        description='Summarize one single-example remask threshold sweep result.'
    )
    parser.add_argument('--label')
    parser.add_argument('--threshold')
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--event-trace', type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--header', action='store_true')
    args = parser.parse_args()

    columns = [
        'label',
        'threshold',
        'accuracy',
        'correct',
        'event_checks',
        'active_checks',
        'candidate_checks',
        'active_candidate_checks',
        'triggered_checks',
        'trigger_rate_all',
        'trigger_rate_active',
        'trigger_rate_active_with_candidates',
        'steps_with_remask',
        'total_remasked_tokens',
        'event_remasked_tokens',
        'avg_remasked_tokens_per_trigger',
        'max_best_score',
        'max_score_margin',
        'first_remask_block',
        'question_preview',
    ]
    if args.header:
        print('\t'.join(columns))
        return

    missing = [
        name
        for name, value in (
            ('label', args.label),
            ('threshold', args.threshold),
            ('trace', args.trace),
            ('event-trace', args.event_trace),
            ('results', args.results),
        )
        if value is None
    ]
    if missing:
        parser.error('missing required arguments: ' + ', '.join('--' + name for name in missing))

    row = build_row(args)
    print('\t'.join(fmt(value) for value in row))


if __name__ == '__main__':
    main()
