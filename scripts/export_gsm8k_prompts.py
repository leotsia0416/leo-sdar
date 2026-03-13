from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GSM8K samples into remask-policy prompt jsonl format."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="GSM8K split to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination jsonl path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of samples to export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_dataset("gsm8k", "main", split=args.split)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(dataset):
            if args.max_samples is not None and count >= args.max_samples:
                break

            prompt = {
                "prompt_id": f"gsm8k_{args.split}_{index:05d}",
                "prompt_text": sample["question"].strip(),
                "reference_text": sample["answer"].strip(),
                "metadata": {
                    "dataset": "gsm8k",
                    "split": args.split,
                },
            }
            handle.write(json.dumps(prompt, ensure_ascii=False) + "\n")
            count += 1

    print(f"exported {count} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
