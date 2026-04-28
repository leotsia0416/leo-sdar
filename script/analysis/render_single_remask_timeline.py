#!/usr/bin/env python3

import argparse
from datetime import datetime
import html
import json
from pathlib import Path

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HTML_DIR = REPO_ROOT / 'artifacts' / 'html'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Render a single-case remask timeline HTML.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--noremask-event-trace', required=True)
    parser.add_argument('--noremask-predictions', required=True)
    parser.add_argument('--noremask-results', required=True)
    parser.add_argument('--remask-event-trace', required=True)
    parser.add_argument('--remask-predictions', required=True)
    parser.add_argument('--remask-results', required=True)
    parser.add_argument('--output')
    return parser.parse_args()


def default_output_path() -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return DEFAULT_HTML_DIR / f'single_remask_timeline_{timestamp}.html'


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_jsonl(path: str):
    trace_path = Path(path)
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def first_prediction_entry(path: str) -> dict:
    payload = load_json(path)
    first_key = sorted(payload.keys(), key=lambda x: int(x))[0]
    return payload[first_key]


def first_result_detail(path: str) -> dict:
    payload = load_json(path)
    return payload['details'][0]


def first_value(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def escape_pre(text: str | None) -> str:
    if not text:
        return '<em>None</em>'
    return f'<pre>{html.escape(text)}</pre>'


def decode_token_ids(tokenizer, token_ids, *, keep_mask_tokens: bool = False) -> str | None:
    if token_ids is None:
        return None
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    if keep_mask_tokens:
        return text
    return text.replace('<|MASK|>', '')


def record_text(record: dict | None, *, prefix: str, tokenizer, keep_mask_tokens: bool = False) -> str | None:
    if record is None:
        return None
    text_key = f'{prefix}_text'
    if text_key in record:
        text = record.get(text_key)
        if text is None:
            return None
        return text if keep_mask_tokens else text.replace('<|MASK|>', '')

    token_key = f'{prefix}_token_ids'
    return decode_token_ids(
        tokenizer,
        record.get(token_key),
        keep_mask_tokens=keep_mask_tokens,
    )


def metric_list(detail: dict, trace_records: list[dict], title: str) -> str:
    steps = sum(int(record.get('triggered', False)) for record in trace_records)
    tokens = sum(int(record.get('remasked_tokens', 0)) for record in trace_records)
    pred_value = first_value(detail.get('pred'))
    correct_value = first_value(detail.get('correct'))
    return (
        f'<div class="card">'
        f'<h3>{html.escape(title)}</h3>'
        f'<div><strong>Extracted answer:</strong> {html.escape(str(pred_value))}</div>'
        f'<div><strong>Correct:</strong> {html.escape(str(correct_value))}</div>'
        f'<div><strong>Trace blocks:</strong> {len(trace_records)}</div>'
        f'<div><strong>Triggered remask steps:</strong> {steps}</div>'
        f'<div><strong>Total remasked tokens:</strong> {tokens}</div>'
        f'</div>'
    )


def timeline_table(noremask_trace: list[dict], remask_trace: list[dict], tokenizer) -> str:
    rows = []
    max_len = max(len(noremask_trace), len(remask_trace))
    for idx in range(max_len):
        n_record = noremask_trace[idx] if idx < len(noremask_trace) else None
        r_record = remask_trace[idx] if idx < len(remask_trace) else None
        block_idx = None
        if r_record is not None:
            block_idx = r_record.get('block_idx')
        elif n_record is not None:
            block_idx = n_record.get('block_idx')

        remask_meta = ''
        if r_record is not None:
            remask_meta = (
                f'active={r_record.get("remask_active")}<br>'
                f'triggered={r_record.get("triggered")}<br>'
                f'remasked_tokens={r_record.get("remasked_tokens")}<br>'
                f'candidate_count={r_record.get("candidate_count")}<br>'
                f'best_score={r_record.get("best_score")}<br>'
                f'positions={html.escape(str(r_record.get("remasked_positions", [])))}'
            )

        rows.append(
            '<tr>'
            f'<td>{html.escape(str(block_idx))}</td>'
            f'<td>{escape_pre(record_text(n_record, prefix="generated_after", tokenizer=tokenizer))}</td>'
            f'<td>{escape_pre(record_text(r_record, prefix="generated_before", tokenizer=tokenizer))}</td>'
            f'<td>{escape_pre(record_text(r_record, prefix="generated_with_masks", tokenizer=tokenizer, keep_mask_tokens=True))}</td>'
            f'<td>{escape_pre(record_text(r_record, prefix="generated_after", tokenizer=tokenizer))}</td>'
            f'<td>{remask_meta}</td>'
            '</tr>'
        )

    return (
        '<table>'
        '<thead><tr>'
        '<th>Block</th>'
        '<th>Noremask After Block</th>'
        '<th>Remask Before Decision</th>'
        '<th>Remask Masked Snapshot</th>'
        '<th>Remask After Block</th>'
        '<th>Remask Metadata</th>'
        '</tr></thead>'
        '<tbody>'
        + ''.join(rows)
        + '</tbody></table>'
    )


def main() -> None:
    args = parse_args()

    noremask_trace = load_jsonl(args.noremask_event_trace)
    remask_trace = load_jsonl(args.remask_event_trace)
    noremask_prediction = first_prediction_entry(args.noremask_predictions)
    remask_prediction = first_prediction_entry(args.remask_predictions)
    noremask_result = first_result_detail(args.noremask_results)
    remask_result = first_result_detail(args.remask_results)
    noremask_pred = first_value(noremask_result.get('pred'))
    remask_pred = first_value(remask_result.get('pred'))
    noremask_correct = bool(first_value(noremask_result.get('correct')))
    remask_correct = bool(first_value(remask_result.get('correct')))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    question = noremask_prediction['origin_prompt'][0]['prompt']
    gold = noremask_prediction['gold']

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SDAR Remask Timeline</title>
  <style>
    :root {{
      --bg: #f6f4ef;
      --fg: #1e1f1d;
      --muted: #6a6c67;
      --panel: #ffffff;
      --line: #d8d2c4;
      --accent: #1f6f78;
      --accent-2: #c44536;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f3efe6 0%, #faf8f2 100%);
      color: var(--fg);
    }}
    main {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px 0;
    }}
    .lead {{
      color: var(--muted);
      margin-bottom: 20px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      margin-bottom: 20px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.04);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .card {{
      background: #fffdf8;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
    }}
    .outputs {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .output-title {{
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 10px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #fbfaf6;
      border: 1px solid #e4dece;
      border-radius: 12px;
      padding: 14px;
      margin: 0;
      font-family: "Iosevka", "SFMono-Regular", monospace;
      font-size: 13px;
      line-height: 1.45;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    th, td {{
      border: 1px solid var(--line);
      vertical-align: top;
      padding: 10px;
      font-size: 13px;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f2ede2;
      z-index: 1;
    }}
    td:nth-child(1) {{ width: 60px; }}
    td:nth-child(6) {{
      width: 180px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .answer-good {{
      color: var(--accent);
      font-weight: 700;
    }}
    .answer-bad {{
      color: var(--accent-2);
      font-weight: 700;
    }}
  </style>
</head>
<body>
<main>
  <div class="panel">
    <h1>SDAR Single-Case Remask Timeline</h1>
    <div class="lead">One GSM8K example, block-by-block. Noremask and remask outputs are aligned by generation block.</div>
    <h2>Question</h2>
    {escape_pre(question)}
    <h2 style="margin-top:16px;">Gold</h2>
    {escape_pre(gold)}
  </div>

  <div class="grid">
    {metric_list(noremask_result, noremask_trace, 'Noremask Summary')}
    {metric_list(remask_result, remask_trace, 'Remask Summary')}
  </div>

  <div class="panel">
    <h2>Final Outputs</h2>
    <div class="outputs">
      <div>
        <div class="output-title">Noremask Final Output</div>
        <div class="lead">Answer: <span class="{'answer-good' if noremask_correct else 'answer-bad'}">{html.escape(str(noremask_pred))}</span></div>
        {escape_pre(noremask_prediction['prediction'])}
      </div>
      <div>
        <div class="output-title">Remask Final Output</div>
        <div class="lead">Answer: <span class="{'answer-good' if remask_correct else 'answer-bad'}">{html.escape(str(remask_pred))}</span></div>
        {escape_pre(remask_prediction['prediction'])}
      </div>
    </div>
  </div>

  <div class="panel">
    <h2>Timeline</h2>
    {timeline_table(noremask_trace, remask_trace, tokenizer)}
  </div>
</main>
</body>
</html>
"""

    output_path = Path(args.output) if args.output else default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding='utf-8')
    print(output_path)


if __name__ == '__main__':
    main()
