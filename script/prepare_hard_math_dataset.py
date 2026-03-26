#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from datasets import Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cached hard MATH train splits to a single JSONL file.")
    parser.add_argument(
        "--cache-root",
        default="/work/leotsia0416/hf_cache/datasets/math",
        help="Root directory containing cached MATH subject splits.",
    )
    parser.add_argument(
        "--output",
        default="/work/leotsia0416/projects/SDAR/training/llama_factory_sdar/data/math_train_hard_local.jsonl",
        help="Output JSONL path.",
    )
    return parser.parse_args()


def iter_subject_arrows(cache_root: Path):
    for subject_dir in sorted(p for p in cache_root.iterdir() if p.is_dir()):
        candidates = sorted(subject_dir.glob('0.0.0/*/math-train.arrow'))
        if not candidates:
            continue
        yield subject_dir.name, candidates[-1]


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    per_subject = {}
    with output_path.open('w', encoding='utf-8') as fout:
        for subject, arrow_path in iter_subject_arrows(cache_root):
            ds = Dataset.from_file(str(arrow_path))
            count = 0
            for row in ds:
                question = (row.get('problem') or '').strip()
                answer = (row.get('solution') or '').strip()
                if not question or not answer:
                    continue
                record = {
                    'question': question,
                    'answer': answer,
                    'subject': row.get('type') or subject.replace('_', ' ').title(),
                    'level': row.get('level') or '',
                    'source': 'hendrycks_math',
                }
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
            per_subject[subject] = count
            total += count

    print(f'Wrote {total} samples to {output_path}')
    for subject, count in per_subject.items():
        print(f'{subject}: {count}')


if __name__ == '__main__':
    main()
