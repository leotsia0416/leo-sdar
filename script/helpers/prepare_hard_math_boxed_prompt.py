#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


PROMPT_SUFFIX = (
    "Solve the problem step by step, but keep the reasoning concise. Once you "
    "know the answer, put the final answer within \\boxed{} and stop immediately. "
    "Do not provide alternative interpretations, repeated checks, or any text "
    "after the boxed answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add the boxed-answer instruction to hard MATH prompts.")
    parser.add_argument(
        "--input",
        default="/work/leotsia0416/projects/SDAR/training/llama_factory_sdar/data/math_train_hard_local.jsonl",
        help="Path to the source hard MATH JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="/work/leotsia0416/projects/SDAR/training/llama_factory_sdar/data/math_train_hard_boxed_prompt_local.jsonl",
        help="Path to the output JSONL file.",
    )
    return parser.parse_args()


def append_prompt_suffix(question: str) -> str:
    question = question.strip()
    if question.endswith(PROMPT_SUFFIX):
        return question
    return f"{question}\n{PROMPT_SUFFIX}"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    rewritten = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = (record.get("question") or "").strip()
            if not question:
                continue
            updated_question = append_prompt_suffix(question)
            if updated_question != question:
                rewritten += 1
            record["question"] = updated_question
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1

    print(f"Wrote {total} samples to {output_path}")
    print(f"Updated prompts: {rewritten}")


if __name__ == "__main__":
    main()
