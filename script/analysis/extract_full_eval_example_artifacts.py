#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path('/work/leotsia0416/projects/SDAR')
OPENCOMPASS_ROOT = REPO_ROOT / 'evaluation' / 'opencompass'
if str(OPENCOMPASS_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENCOMPASS_ROOT))

from opencompass.datasets import MATHEvaluator, gsm8k_dataset_postprocess, math_postprocess_sdar


PROMPT_SUFFIX = (
    "Solve the problem step by step, but keep the reasoning concise. Once you "
    "know the answer, put the final answer within \\boxed{} and stop immediately. "
    "Do not provide alternative interpretations, repeated checks, or any text "
    "after the boxed answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract one example from a full OpenCompass eval run.')
    parser.add_argument('--exp-dir', required=True)
    parser.add_argument('--gsm8k-test', default='/work/leotsia0416/datasets/gsm8k/test.jsonl')
    parser.add_argument('--example-index', type=int, required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def load_example(path: str, index: int) -> dict:
    with Path(path).open('r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line_idx == index:
                return json.loads(line)
    raise IndexError(f'Example index {index} out of range for {path}')


def build_prompt(question: str) -> str:
    return f'{question}\n{PROMPT_SUFFIX}'


def normalize_prompt(prompt: str | None) -> str | None:
    if prompt is None:
        return None
    return prompt.rstrip()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    example = load_example(args.gsm8k_test, args.example_index)
    target_prompt = normalize_prompt(build_prompt(example['question']))

    matched_predictions = []
    for prediction_path in sorted(exp_dir.glob('predictions/**/*.json')):
        payload = json.loads(prediction_path.read_text(encoding='utf-8'))
        for key, record in payload.items():
            origin_prompt = record.get('origin_prompt') or []
            prompt = normalize_prompt(origin_prompt[0].get('prompt')) if origin_prompt else None
            if prompt == target_prompt:
                matched_predictions.append(
                    {
                        'source_path': str(prediction_path),
                        'record_key': key,
                        'record': record,
                    }
                )

    matched_trace_records = []
    trace_path = exp_dir / 'summary' / 'remask_trace.jsonl'
    if trace_path.is_file():
        for line in trace_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            input_payload = record.get('input') or []
            prompt = normalize_prompt(input_payload[0].get('prompt')) if input_payload else None
            if prompt == target_prompt:
                matched_trace_records.append(record)

    matched_event_records = []
    event_trace_path = exp_dir / 'summary' / 'remask_event_trace.jsonl'
    if event_trace_path.is_file():
        for line in event_trace_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            input_payload = record.get('input') or []
            prompt = normalize_prompt(input_payload[0].get('prompt')) if input_payload else None
            if prompt == target_prompt:
                matched_event_records.append(record)

    extracted = {
        'example_index': args.example_index,
        'question': example['question'],
        'gold_answer': example['answer'],
        'exp_dir': str(exp_dir),
        'matched_prediction_count': len(matched_predictions),
        'matched_trace_count': len(matched_trace_records),
        'matched_event_count': len(matched_event_records),
    }

    if matched_predictions:
        prediction_text = matched_predictions[0]['record'].get('prediction', '')
        pred_processed = math_postprocess_sdar(prediction_text)
        gold_processed = gsm8k_dataset_postprocess(example['answer'])
        detail = MATHEvaluator(version='v2').score([pred_processed], [gold_processed])['details'][0]
        extracted['evaluation'] = detail

    write_json(output_dir / 'summary.json', extracted)
    write_json(output_dir / 'predictions.json', matched_predictions)
    write_jsonl(output_dir / 'remask_trace.jsonl', matched_trace_records)
    write_jsonl(output_dir / 'remask_event_trace.jsonl', matched_event_records)
    print(output_dir / 'summary.json')


if __name__ == '__main__':
    main()
