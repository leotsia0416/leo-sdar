#!/usr/bin/env python3

import argparse
from datetime import datetime
import html
import json
from pathlib import Path

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HTML_DIR = REPO_ROOT / 'artifacts' / 'html'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render an extracted remask case into a full step-by-step HTML timeline.'
    )
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--remask-case-dir', required=True)
    parser.add_argument('--noremask-case-dir')
    parser.add_argument('--output')
    return parser.parse_args()


def default_output_path(remask_case_dir: str) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    case_name = Path(remask_case_dir).resolve().name
    return DEFAULT_HTML_DIR / f'{case_name}_timeline_{timestamp}.html'


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def escape_pre(text: str | None) -> str:
    if not text:
        return '<pre><em>None</em></pre>'
    return f'<pre>{html.escape(text)}</pre>'


def load_extracted_case(case_dir: Path) -> dict:
    summary = load_json(case_dir / 'summary.json')
    predictions = load_json(case_dir / 'predictions.json')
    prediction_record = predictions[0]['record'] if predictions else {}
    return {
        'summary': summary,
        'prediction_record': prediction_record,
        'prediction_text': prediction_record.get('prediction', ''),
        'question': summary.get('question') or (
            (prediction_record.get('origin_prompt') or [{}])[0].get('prompt', '')
        ),
        'gold': summary.get('gold_answer', ''),
        'event_trace': load_jsonl(case_dir / 'remask_event_trace.jsonl'),
        'trace': load_jsonl(case_dir / 'remask_trace.jsonl'),
        'evaluation': summary.get('evaluation', {}),
    }


def build_tokenizer(model_path: Path) -> tuple[Tokenizer, int]:
    config = load_json(model_path / 'tokenizer_config.json')
    tokenizer = Tokenizer(
        models.BPE.from_file(
            str(model_path / 'vocab.json'),
            str(model_path / 'merges.txt'),
            unk_token=config.get('unk_token'),
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=config.get('add_prefix_space', False),
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel()

    added = []
    content_to_id: dict[str, int] = {}
    for token_id, meta in sorted(
        config.get('added_tokens_decoder', {}).items(),
        key=lambda item: int(item[0]),
    ):
        content = meta['content']
        content_to_id[content] = int(token_id)
        added.append(
            AddedToken(
                content,
                lstrip=meta.get('lstrip', False),
                normalized=meta.get('normalized', False),
                rstrip=meta.get('rstrip', False),
                single_word=meta.get('single_word', False),
                special=meta.get('special', False),
            ))
    if added:
        tokenizer.add_special_tokens(added)

    mask_content = config.get('mask_token') or '<|MASK|>'
    mask_id = content_to_id[mask_content]
    return tokenizer, mask_id


def decode_text(tokenizer: Tokenizer, token_ids: list[int] | None) -> str:
    if token_ids is None:
        return ''
    return tokenizer.decode(token_ids)


def contiguous_spans(indices: list[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(indices))
    if not ordered:
        return []
    spans: list[tuple[int, int]] = []
    start = ordered[0]
    prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        spans.append((start, prev + 1))
        start = idx
        prev = idx
    spans.append((start, prev + 1))
    return spans


def decode_segment(tokenizer: Tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ''
    return tokenizer.decode(token_ids)


def wrap(text: str, class_name: str) -> str:
    if not text:
        return ''
    return f'<span class="{class_name}">{html.escape(text)}</span>'


def render_plain(tokenizer: Tokenizer, token_ids: list[int]) -> str:
    return html.escape(decode_segment(tokenizer, token_ids))


def render_with_suffix_highlight(
    tokenizer: Tokenizer,
    prev_ids: list[int],
    curr_ids: list[int],
) -> str:
    common = 0
    max_common = min(len(prev_ids), len(curr_ids))
    while common < max_common and prev_ids[common] == curr_ids[common]:
        common += 1

    prefix = render_plain(tokenizer, curr_ids[:common])
    suffix_text = decode_segment(tokenizer, curr_ids[common:])
    if not suffix_text:
        return prefix
    return prefix + wrap(suffix_text, 'add')


def render_with_marked_spans(
    tokenizer: Tokenizer,
    token_ids: list[int],
    spans: list[tuple[int, int]],
    class_name: str,
) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        start = max(0, min(start, len(token_ids)))
        end = max(start, min(end, len(token_ids)))
        if cursor < start:
            parts.append(render_plain(tokenizer, token_ids[cursor:start]))
        if start < end:
            parts.append(wrap(decode_segment(tokenizer, token_ids[start:end]), class_name))
        cursor = end
    if cursor < len(token_ids):
        parts.append(render_plain(tokenizer, token_ids[cursor:]))
    return ''.join(parts)


def render_with_masks(
    tokenizer: Tokenizer,
    token_ids: list[int],
    spans: list[tuple[int, int]],
) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        start = max(0, min(start, len(token_ids)))
        end = max(start, min(end, len(token_ids)))
        if cursor < start:
            parts.append(render_plain(tokenizer, token_ids[cursor:start]))
        if start < end:
            parts.append('<span class="mask">[MASK]</span>')
        cursor = end
    if cursor < len(token_ids):
        parts.append(render_plain(tokenizer, token_ids[cursor:]))
    return ''.join(parts)


def remasked_local_positions(record: dict) -> list[int]:
    prompt_length = int(record.get('prompt_length', 0))
    before_ids = record.get('generated_before_token_ids') or []
    local = []
    for pos in record.get('remasked_positions', []) or []:
        local_idx = int(pos) - prompt_length
        if 0 <= local_idx < len(before_ids):
            local.append(local_idx)
    return local


def build_steps(tokenizer: Tokenizer, event_trace: list[dict]) -> list[dict]:
    steps: list[dict] = []
    prev_after_ids: list[int] = []

    for frame_idx, record in enumerate(event_trace):
        before_ids = record.get('generated_before_token_ids') or []
        after_ids = record.get('generated_after_token_ids') or []
        local_positions = remasked_local_positions(record)
        spans = contiguous_spans(local_positions)

        base_meta = {
            'frame_idx': frame_idx,
            'block_idx': int(record.get('block_idx', -1)),
            'generated_blocks': int(record.get('generated_blocks', -1)),
            'triggered': bool(record.get('triggered', False)),
            'remasked_tokens': int(record.get('remasked_tokens', 0)),
            'candidate_count': int(record.get('candidate_count', 0)),
            'best_score': record.get('best_score'),
            'positions': record.get('remasked_positions', []) or [],
        }

        if not record.get('triggered'):
            steps.append({
                **base_meta,
                'phase': 'generate',
                'phase_title': 'Normal generation',
                'phase_short': 'Generate',
                'content_html': render_with_suffix_highlight(tokenizer, prev_after_ids, after_ids),
            })
            prev_after_ids = after_ids
            continue

        steps.append({
            **base_meta,
            'phase': 'before',
            'phase_title': 'Before remask decision',
            'phase_short': 'Before',
            'content_html': render_with_suffix_highlight(tokenizer, prev_after_ids, before_ids),
        })
        steps.append({
            **base_meta,
            'phase': 'delete',
            'phase_title': 'Selected span is deleted',
            'phase_short': 'Delete',
            'content_html': render_with_marked_spans(tokenizer, before_ids, spans, 'del'),
        })
        steps.append({
            **base_meta,
            'phase': 'mask',
            'phase_title': 'Deleted span becomes [MASK]',
            'phase_short': 'Mask',
            'content_html': render_with_masks(tokenizer, before_ids, spans),
        })
        steps.append({
            **base_meta,
            'phase': 'regen',
            'phase_title': 'Masked span is regenerated',
            'phase_short': 'Regenerate',
            'content_html': render_with_marked_spans(tokenizer, after_ids, spans, 'add'),
        })
        prev_after_ids = after_ids

    return steps


def metric_card(title: str, case: dict) -> str:
    evaluation = case.get('evaluation', {})
    pred_value = evaluation.get('pred')
    correct_value = evaluation.get('correct')
    trace = case.get('trace', [])
    event_trace = case.get('event_trace', [])
    triggered = sum(int(record.get('triggered', False)) for record in event_trace)
    remasked = sum(int(record.get('remasked_tokens', 0)) for record in event_trace)
    return (
        f'<div class="card">'
        f'<h3>{html.escape(title)}</h3>'
        f'<div><strong>Extracted answer:</strong> {html.escape(str(pred_value))}</div>'
        f'<div><strong>Correct:</strong> {html.escape(str(correct_value))}</div>'
        f'<div><strong>Trace summaries:</strong> {len(trace)}</div>'
        f'<div><strong>Event frames:</strong> {len(event_trace)}</div>'
        f'<div><strong>Triggered remask steps:</strong> {triggered}</div>'
        f'<div><strong>Total remasked tokens:</strong> {remasked}</div>'
        f'</div>'
    )


def final_output_panel(title: str, text: str, correct: bool | None, pred: str | None) -> str:
    if correct is None:
        answer_html = ''
    else:
        cls = 'answer-good' if correct else 'answer-bad'
        answer_html = (
            f'<div class="lead">Answer: '
            f'<span class="{cls}">{html.escape(str(pred))}</span></div>'
        )
    return (
        '<div>'
        f'<div class="output-title">{html.escape(title)}</div>'
        f'{answer_html}'
        f'{escape_pre(text)}'
        '</div>'
    )


def compact_reference_card(title: str, pred: str | None, correct: bool | None, text: str) -> str:
    answer_cls = 'answer-good' if correct else 'answer-bad'
    return (
        '<div class="card">'
        f'<h3>{html.escape(title)}</h3>'
        f'<div><strong>Final answer:</strong> <span class="{answer_cls}">{html.escape(str(pred))}</span></div>'
        f'<div style="margin-top:8px;"><strong>Why it is only a reference:</strong> this is a separate run, so pre-remask text can already drift.</div>'
        f'<div style="margin-top:12px;">{escape_pre(text.strip())}</div>'
        '</div>'
    )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    remask_case = load_extracted_case(Path(args.remask_case_dir))
    noremask_case = load_extracted_case(Path(args.noremask_case_dir)) if args.noremask_case_dir else None
    tokenizer, _mask_id = build_tokenizer(model_path)

    question = remask_case['question']
    gold = remask_case['gold']
    steps = build_steps(tokenizer, remask_case['event_trace'])
    triggered_steps = sum(1 for step in steps if step['phase'] != 'generate')
    first_remask_step_idx = next(
        (idx for idx, step in enumerate(steps) if step['phase'] == 'before'),
        0,
    )

    step_payload = []
    for step_idx, step in enumerate(steps):
        step_payload.append({
            'step_idx': step_idx,
            'frame_idx': step['frame_idx'],
            'block_idx': step['block_idx'],
            'generated_blocks': step['generated_blocks'],
            'triggered': step['triggered'],
            'phase': step['phase'],
            'phase_title': step['phase_title'],
            'phase_short': step['phase_short'],
            'remasked_tokens': step['remasked_tokens'],
            'candidate_count': step['candidate_count'],
            'best_score': step['best_score'],
            'positions': step['positions'],
            'content_html': step['content_html'],
        })

    remask_eval = remask_case.get('evaluation', {})
    noremask_eval = noremask_case.get('evaluation', {}) if noremask_case else {}

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SDAR Remask Timeline</title>
  <style>
    :root {{
      --bg: #f6f2e9;
      --fg: #1e1f1d;
      --muted: #696b66;
      --panel: #fffdf7;
      --line: #ddd3bf;
      --good: #12624f;
      --bad: #b0382a;
      --mask: #8a5a00;
      --accent: #1a648b;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f3eee4 0%, #fbf8f1 100%);
      color: var(--fg);
    }}
    main {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: 0 10px 24px rgba(0,0,0,.04);
    }}
    .lead {{
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .outputs {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .card {{
      background: #fffaf1;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
    }}
    .output-title {{
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 10px;
    }}
    pre, .stage-box {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      padding: 14px;
      border: 1px solid #e6dcc9;
      border-radius: 12px;
      background: #fbfaf6;
      font-family: "Iosevka", "SFMono-Regular", monospace;
      font-size: 13px;
      line-height: 1.5;
      min-height: 320px;
    }}
    .controls {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 10px;
      padding: 8px 12px;
      cursor: pointer;
      font: inherit;
    }}
    input[type=range] {{
      width: 380px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 14px;
    }}
    .badges {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 12px;
    }}
    .badge.good {{
      background: #d8f1e8;
      color: var(--good);
    }}
    .badge.bad {{
      background: #f8ddd7;
      color: var(--bad);
    }}
    .badge.info {{
      background: #dcecf8;
      color: var(--accent);
    }}
    .stagebar {{
      display: flex;
      gap: 8px;
      margin: 10px 0 14px;
      flex-wrap: wrap;
    }}
    .stagepill {{
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
    }}
    .stagepill.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }}
    .answer-good {{
      color: var(--good);
      font-weight: 700;
    }}
    .answer-bad {{
      color: var(--bad);
      font-weight: 700;
    }}
    .del {{
      color: var(--bad);
      background: #ffe6e1;
      text-decoration: line-through;
      text-decoration-thickness: 2px;
    }}
    .add {{
      color: var(--good);
      background: #dff7e6;
      font-weight: 700;
    }}
    .mask {{
      color: var(--mask);
      background: #fff0b8;
      font-weight: 700;
      padding: 0 3px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
<main>
  <div class="panel">
    <h1>Remask Generation Timeline</h1>
    <div class="lead">
      Normal blocks keep moving the text forward. When remask triggers, that block expands into:
      original text, red strikethrough deletion, [MASK], then green regenerated text.
    </div>
    <div class="badges">
      <span class="badge {'good' if remask_eval.get('correct') else 'bad'}">remask: {html.escape(str(remask_eval.get('pred')))}</span>
      {'<span class="badge ' + ('good' if noremask_eval.get('correct') else 'bad') + '">noremask ref: ' + html.escape(str(noremask_eval.get('pred'))) + '</span>' if noremask_case else ''}
      <span class="badge info">timeline steps: {len(step_payload)}</span>
      <span class="badge info">remask substeps: {triggered_steps}</span>
      <span class="badge info">first remask step: {first_remask_step_idx + 1}</span>
    </div>
  </div>

  <div class="panel">
    <h2>Question</h2>
    {escape_pre(question)}
    <h2 style="margin-top:16px;">Gold</h2>
    {escape_pre(gold)}
  </div>

  <div class="grid">
    {metric_card('Remask Summary', remask_case)}
    {metric_card('Noremask Summary', noremask_case) if noremask_case else '<div class="card"><h3>Noremask Summary</h3><div>No noremask case dir provided.</div></div>'}
  </div>

  <div class="panel">
    <h2>What This Page Is Showing</h2>
    <div class="lead">
      The animation below is causal only for the remask run itself. It does not compare two runs block by block.
      `noremask` is kept only as a final-answer reference, because the two runs already drift before the first remask.
    </div>
    <div class="outputs">
      {final_output_panel('Remask Final Output', remask_case['prediction_text'], remask_eval.get('correct'), remask_eval.get('pred'))}
      {compact_reference_card('Noremask Reference', noremask_eval.get('pred'), noremask_eval.get('correct'), noremask_case['prediction_text']) if noremask_case else '<div class="card"><h3>Noremask Reference</h3><div>Not provided.</div></div>'}
    </div>
  </div>

  <div class="panel">
    <h2>Remask-Only Timeline</h2>
    <div class="controls">
      <button id="prev">Prev</button>
      <button id="play">Play</button>
      <button id="next">Next</button>
      <button id="jumpRemask">Jump to First Remask</button>
      <input id="slider" type="range" min="0" max="{max(len(step_payload) - 1, 0)}" value="0">
      <span id="frameLabel" class="meta"></span>
    </div>
    <div id="status" class="lead"></div>
    <div class="stagebar">
      <span class="stagepill" id="pill-generate">Generate</span>
      <span class="stagepill" id="pill-before">Before</span>
      <span class="stagepill" id="pill-delete">Delete</span>
      <span class="stagepill" id="pill-mask">Mask</span>
      <span class="stagepill" id="pill-regen">Regenerate</span>
    </div>
    <div id="stageTitle" style="font-weight:700;margin-bottom:8px;"></div>
    <div id="stageBox" class="stage-box"></div>
  </div>
</main>
<script>
const steps = {json.dumps(step_payload, ensure_ascii=False)};
const slider = document.getElementById('slider');
const frameLabel = document.getElementById('frameLabel');
const status = document.getElementById('status');
const stageTitle = document.getElementById('stageTitle');
const stageBox = document.getElementById('stageBox');
const playBtn = document.getElementById('play');
const jumpRemaskBtn = document.getElementById('jumpRemask');
const firstRemaskStepIdx = {first_remask_step_idx};
const pills = {{
  generate: document.getElementById('pill-generate'),
  before: document.getElementById('pill-before'),
  delete: document.getElementById('pill-delete'),
  mask: document.getElementById('pill-mask'),
  regen: document.getElementById('pill-regen'),
}};
let stepIdx = 0;
let timer = null;

function fmtScore(value) {{
  return value === null || value === undefined ? 'n/a' : Number(value).toFixed(3);
}}

function render() {{
  const step = steps[stepIdx];
  slider.value = stepIdx;
  frameLabel.textContent = `step ${{stepIdx + 1}} / ${{steps.length}} | frame=${{step.frame_idx + 1}} | block=${{step.block_idx}} | gen_blocks=${{step.generated_blocks}}`;
  status.textContent = `phase=${{step.phase_short}} | triggered=${{step.triggered}} | remasked_tokens=${{step.remasked_tokens}} | candidate_count=${{step.candidate_count}} | best_score=${{fmtScore(step.best_score)}} | positions=${{JSON.stringify(step.positions)}}`;
  stageTitle.textContent = step.phase_title;
  stageBox.innerHTML = step.content_html;
  for (const [name, el] of Object.entries(pills)) {{
    el.classList.toggle('active', name === step.phase);
  }}
}}

function stop() {{
  if (timer) {{
    clearInterval(timer);
    timer = null;
    playBtn.textContent = 'Play';
  }}
}}

function play() {{
  if (timer) {{
    stop();
    return;
  }}
  playBtn.textContent = 'Pause';
  timer = setInterval(() => {{
    if (stepIdx >= steps.length - 1) {{
      stop();
      return;
    }}
    stepIdx += 1;
    render();
  }}, 850);
}}

document.getElementById('prev').onclick = () => {{
  stop();
  stepIdx = Math.max(0, stepIdx - 1);
  render();
}};
document.getElementById('next').onclick = () => {{
  stop();
  stepIdx = Math.min(steps.length - 1, stepIdx + 1);
  render();
}};
slider.oninput = (event) => {{
  stop();
  stepIdx = Number(event.target.value);
  render();
}};
playBtn.onclick = play;
jumpRemaskBtn.onclick = () => {{
  stop();
  stepIdx = firstRemaskStepIdx;
  render();
}};
render();
</script>
</body>
</html>
"""

    output_path = Path(args.output) if args.output else default_output_path(args.remask_case_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding='utf-8')
    print(output_path)


if __name__ == '__main__':
    main()
