#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


PROMPT_SUFFIX = (
    "Reason step by step, but keep the reasoning concise. End your response with "
    "exactly one final line in the form \\boxed{NUMBER}. Put only the final numeric "
    "answer inside the box, with no words or units. Do not output anything after the "
    "boxed answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a boxed-answer format-alignment dataset from GSM8K.")
    parser.add_argument("--input", required=True, help="Path to the GSM8K train.jsonl file.")
    parser.add_argument("--output", required=True, help="Path to the output alpaca-format JSON file.")
    parser.add_argument(
        "--max-reasoning-lines",
        type=int,
        default=6,
        help="Maximum number of reasoning lines to keep before the final boxed answer.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on the number of samples to export. <=0 means all samples.",
    )
    return parser.parse_args()


def _extract_final_answer(answer: str) -> str:
    final_line = ""
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if line.startswith("####"):
            final_line = line[4:].strip()
    if not final_line:
        matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", answer)
        if not matches:
            raise ValueError(f"Cannot extract final answer from: {answer!r}")
        return matches[-1].replace(",", "")

    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", final_line)
    if not matches:
        raise ValueError(f"Cannot parse numeric final answer from: {final_line!r}")
    return matches[-1].replace(",", "")


def _clean_reasoning(answer: str, max_reasoning_lines: int) -> list[str]:
    cleaned_lines: list[str] = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("####"):
            continue
        line = re.sub(r"<<[^<>]*>>", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        if line:
            cleaned_lines.append(line)

    if max_reasoning_lines > 0:
        cleaned_lines = cleaned_lines[:max_reasoning_lines]

    return cleaned_lines


def convert_record(record: dict[str, str], max_reasoning_lines: int) -> dict[str, str]:
    question = record["question"].strip()
    answer = record["answer"]
    final_answer = _extract_final_answer(answer)
    reasoning_lines = _clean_reasoning(answer, max_reasoning_lines=max_reasoning_lines)
    reasoning = "\n".join(reasoning_lines).strip()
    if not reasoning:
        reasoning = f"The final answer is {final_answer}."

    return {
        "instruction": f"{question}\n{PROMPT_SUFFIX}",
        "output": f"<think>\n{reasoning}\n</think>\n\n\\boxed{{{final_answer}}}",
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    samples = []
    with input_path.open("r", encoding="utf-8") as fp:
        for line_idx, line in enumerate(fp):
            if args.max_samples > 0 and len(samples) >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            converted = convert_record(record, max_reasoning_lines=args.max_reasoning_lines)
            converted["source_index"] = line_idx
            samples.append(converted)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
