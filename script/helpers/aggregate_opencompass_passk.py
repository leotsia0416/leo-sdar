#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def find_result_paths(root: Path, dataset_abbr: str) -> list[Path]:
    return sorted(root.glob(f'seed_*/**/results/*/{dataset_abbr}.json'))


def load_result(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Aggregate repeated OpenCompass result JSON files into pass@k.'
    )
    parser.add_argument('root', type=Path, help='Root directory containing seed_* subdirectories.')
    parser.add_argument(
        '--dataset-abbr',
        default='gsm8k',
        help='Dataset abbreviation used in OpenCompass result filenames.',
    )
    parser.add_argument(
        '--expected-k',
        type=int,
        default=None,
        help='Optional expected number of result files. Fails if discovery does not match.',
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        default=None,
        help='Where to save the aggregated pass@k JSON summary.',
    )
    parser.add_argument(
        '--output-txt',
        type=Path,
        default=None,
        help='Where to save the human-readable pass@k summary.',
    )
    args = parser.parse_args()

    result_paths = find_result_paths(args.root, args.dataset_abbr)
    if not result_paths:
        raise FileNotFoundError(
            f'No result JSON files were found under {args.root} for {args.dataset_abbr}.'
        )
    if args.expected_k is not None and len(result_paths) != args.expected_k:
        raise ValueError(
            f'Expected {args.expected_k} result files, found {len(result_paths)} under {args.root}.'
        )

    runs = []
    example_map: dict[str, dict] = {}
    for result_path in result_paths:
        result = load_result(result_path)
        runs.append(
            {
                'path': str(result_path),
                'accuracy': float(result['accuracy']),
            }
        )
        for detail in result['details']:
            example_abbr = detail['example_abbr']
            correct = bool(detail['correct'][0])
            pred = detail['pred'][0]
            answer = detail['answer'][0]
            record = example_map.setdefault(
                example_abbr,
                {
                    'example_abbr': example_abbr,
                    'answer': answer,
                    'num_correct': 0,
                    'pass_at_k': False,
                    'correct_run_indices': [],
                    'sample_preds': [],
                },
            )
            record['num_correct'] += int(correct)
            if correct:
                record['pass_at_k'] = True
                record['correct_run_indices'].append(len(runs) - 1)
            if len(record['sample_preds']) < 4:
                record['sample_preds'].append(pred)

    examples = sorted(example_map.values(), key=lambda item: item['example_abbr'])
    total = len(examples)
    passed = sum(1 for example in examples if example['pass_at_k'])
    pass_at_k = 100.0 * passed / total if total else 0.0

    summary = {
        'dataset_abbr': args.dataset_abbr,
        'k': len(result_paths),
        'pass@k': pass_at_k,
        'correct': passed,
        'total': total,
        'seed_runs': runs,
        'examples': examples,
    }

    output_json = args.output_json or args.root / f'{args.dataset_abbr}_pass_at_{len(result_paths)}.json'
    output_txt = args.output_txt or args.root / f'{args.dataset_abbr}_pass_at_{len(result_paths)}.txt'
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open('w') as f:
        json.dump(summary, f, indent=2)

    individual = [run['accuracy'] for run in runs]
    with output_txt.open('w') as f:
        f.write(f'dataset={args.dataset_abbr}\n')
        f.write(f'k={len(result_paths)}\n')
        f.write(f'pass@{len(result_paths)}={pass_at_k:.4f}\n')
        f.write(f'correct={passed}/{total}\n')
        f.write(
            'individual_accuracy='
            + ', '.join(f'{acc:.4f}' for acc in individual)
            + '\n'
        )

    print(f'Found {len(result_paths)} runs under {args.root}')
    print(f'pass@{len(result_paths)} = {pass_at_k:.4f} ({passed}/{total})')
    print(f'Wrote JSON summary to: {output_json}')
    print(f'Wrote text summary to: {output_txt}')


if __name__ == '__main__':
    main()
