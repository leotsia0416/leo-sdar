#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from statistics import mean


def percentile(values, q):
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
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def describe(values):
    if not values:
        return "count=0"
    return "count={count} mean={mean} p50={p50} p90={p90} min={min} max={max}".format(
        count=len(values),
        mean=fmt(mean(values)),
        p50=fmt(percentile(values, 0.5)),
        p90=fmt(percentile(values, 0.9)),
        min=fmt(min(values)),
        max=fmt(max(values)),
    )


def prompt_preview(record):
    input_payload = record.get("input")
    if isinstance(input_payload, list) and input_payload:
        prompt = input_payload[0].get("prompt", "")
        if prompt:
            return prompt.splitlines()[0]
    return ""


def build_example_key(record):
    return json.dumps(record.get("input"), ensure_ascii=False, sort_keys=True)


def summarize_examples(records):
    grouped = {}
    for record in records:
        key = build_example_key(record)
        grouped.setdefault(
            key,
            {
                "prompt": prompt_preview(record),
                "checks": 0,
                "triggers": 0,
                "first_check_block": None,
                "first_check_progress": None,
                "first_trigger_block": None,
                "first_trigger_progress": None,
                "max_best_score": None,
                "max_score_margin": None,
            },
        )
        item = grouped[key]
        item["checks"] += 1
        block = record.get("generated_blocks")
        progress = record.get("remask_progress")
        if item["first_check_block"] is None or block < item["first_check_block"]:
            item["first_check_block"] = block
            item["first_check_progress"] = progress
        score = record.get("best_score")
        if score is not None and (item["max_best_score"] is None or score > item["max_best_score"]):
            item["max_best_score"] = score
        margin = record.get("score_margin")
        if margin is not None and (item["max_score_margin"] is None or margin > item["max_score_margin"]):
            item["max_score_margin"] = margin
        if record.get("triggered"):
            item["triggers"] += 1
            if item["first_trigger_block"] is None or block < item["first_trigger_block"]:
                item["first_trigger_block"] = block
                item["first_trigger_progress"] = progress
    return list(grouped.values())


def main():
    parser = argparse.ArgumentParser(description="Summarize detailed remask event traces.")
    parser.add_argument("trace_path", help="Path to remask event trace jsonl.")
    parser.add_argument("--top-k", type=int, default=20, help="Show top K examples by trigger count.")
    args = parser.parse_args()

    trace_path = Path(args.trace_path)
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not records:
        print(f"trace_path={trace_path}")
        print("num_event_records=0")
        return

    examples = summarize_examples(records)
    triggered_records = [r for r in records if r.get("triggered")]
    score_records = [r for r in records if r.get("best_score") is not None]
    triggered_examples = [e for e in examples if e["triggers"] > 0]

    print(f"trace_path={trace_path}")
    print(f"num_event_records={len(records)}")
    print(f"num_examples_with_checks={len(examples)}")
    print(f"num_examples_with_triggers={len(triggered_examples)}")
    print(f"trigger_rate_per_check={fmt(len(triggered_records) / len(records))}")
    print(f"checks_per_example={describe([e['checks'] for e in examples])}")
    print(f"triggers_per_example={describe([e['triggers'] for e in examples])}")
    print(f"candidate_count={describe([int(r['candidate_count']) for r in records])}")
    print(f"best_score_all_checks={describe([float(r['best_score']) for r in score_records])}")
    print(f"score_margin_all_checks={describe([float(r['score_margin']) for r in score_records if r.get('score_margin') is not None])}")
    print(f"best_score_triggered_checks={describe([float(r['best_score']) for r in triggered_records if r.get('best_score') is not None])}")
    print(f"remasked_tokens_triggered_checks={describe([int(r['remasked_tokens']) for r in triggered_records])}")
    print(f"first_check_block={describe([int(e['first_check_block']) for e in examples if e['first_check_block'] is not None])}")
    print(f"first_check_progress={describe([float(e['first_check_progress']) for e in examples if e['first_check_progress'] is not None])}")
    print(f"first_trigger_block={describe([int(e['first_trigger_block']) for e in triggered_examples if e['first_trigger_block'] is not None])}")
    print(f"first_trigger_progress={describe([float(e['first_trigger_progress']) for e in triggered_examples if e['first_trigger_progress'] is not None])}")

    top_examples = sorted(
        examples,
        key=lambda item: (item["triggers"], item["checks"], item["max_score_margin"] or float("-inf")),
        reverse=True,
    )[: max(0, args.top_k)]
    print("top_examples=prompt | checks | triggers | first_check_block | first_trigger_block | max_score_margin")
    for item in top_examples:
        print(
            " - {prompt} | {checks} | {triggers} | {first_check} | {first_trigger} | {margin}".format(
                prompt=item["prompt"],
                checks=item["checks"],
                triggers=item["triggers"],
                first_check=item["first_check_block"],
                first_trigger=item["first_trigger_block"],
                margin=fmt(item["max_score_margin"]),
            )
        )


if __name__ == "__main__":
    main()
