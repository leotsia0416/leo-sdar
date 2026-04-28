#!/usr/bin/env python3

import argparse
import gc
import json
import math
import os
import struct
import sys
from pathlib import Path

import torch
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM


REPO_ROOT = Path('/work/leotsia0416/projects/SDAR')
OPENCOMPASS_ROOT = REPO_ROOT / 'evaluation' / 'opencompass'
if str(OPENCOMPASS_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENCOMPASS_ROOT))

from opencompass.datasets import MATHEvaluator, gsm8k_dataset_postprocess, math_postprocess_sdar
from opencompass.models.huggingface_bd3 import _patch_block_diffusion_3d_mask
from opencompass.models.huggingface_sdar_gap import SDARGapwithChatTemplate


PROMPT_SUFFIX = (
    "Solve the problem step by step, but keep the reasoning concise. Once you "
    "know the answer, put the final answer within \\boxed{} and stop immediately. "
    "Do not provide alternative interpretations, repeated checks, or any text "
    "after the boxed answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run one GSM8K example with and without remask.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--gsm8k-test', default='/work/leotsia0416/datasets/gsm8k/test.jsonl')
    parser.add_argument('--example-index', type=int, required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def load_example(path: str, index: int) -> dict:
    with Path(path).open('r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line_idx == index:
                return json.loads(line)
    raise IndexError(f'Example index {index} out of range for {path}')


def build_input_record(question: str) -> list[dict]:
    return [
        {
            'role': 'HUMAN',
            'prompt': f'{question}\n{PROMPT_SUFFIX}',
        }
    ]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _load_shard_headers(model_path: Path) -> tuple[dict, dict]:
    index_payload = json.loads((model_path / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    weight_map = index_payload['weight_map']
    shard_to_names: dict[str, list[str]] = {}
    for name, shard_name in weight_map.items():
        shard_to_names.setdefault(shard_name, []).append(name)
    return weight_map, shard_to_names


def _read_safetensor_header(shard_path: Path) -> tuple[int, dict]:
    with shard_path.open('rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header_len, header


def _load_f32_tensor_to_device(
    file_obj,
    *,
    data_base: int,
    start: int,
    end: int,
    shape: list[int],
    device: str,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    file_obj.seek(data_base + start)
    target = torch.empty(shape, device=device, dtype=target_dtype)
    target_flat = target.view(-1)
    total_elems = math.prod(shape)
    chunk_elems = (8 << 20) // 4
    loaded = 0
    while loaded < total_elems:
        current_elems = min(chunk_elems, total_elems - loaded)
        raw = file_obj.read(current_elems * 4)
        tensor = torch.frombuffer(memoryview(raw), dtype=torch.float32)
        gpu_chunk = torch.empty(current_elems, device=device, dtype=torch.float32)
        gpu_chunk.copy_(tensor)
        target_flat[loaded:loaded + current_elems].copy_(gpu_chunk)
        loaded += current_elems
        del gpu_chunk
        del tensor
        raw = None
    return target


def build_stream_loaded_model(model_path: str, device: str) -> torch.nn.Module:
    model_dir = Path(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    _, shard_to_names = _load_shard_headers(model_dir)
    for shard_name in sorted(shard_to_names):
        shard_path = model_dir / shard_name
        header_len, header = _read_safetensor_header(shard_path)
        data_base = 8 + header_len
        with shard_path.open('rb') as f:
            for tensor_name in shard_to_names[shard_name]:
                info = header[tensor_name]
                if info['dtype'] != 'F32':
                    raise ValueError(f'Unsupported dtype for {tensor_name}: {info["dtype"]}')
                start, end = info['data_offsets']
                shape = info['shape']
                tensor = _load_f32_tensor_to_device(
                    f,
                    data_base=data_base,
                    start=start,
                    end=end,
                    shape=shape,
                    device=device,
                    target_dtype=torch.bfloat16,
                )
                set_module_tensor_to_device(
                    model,
                    tensor_name,
                    device,
                    value=tensor,
                    dtype=torch.bfloat16,
                )
                del tensor
        gc.collect()

    meta_buffers = [name for name, buf in model.named_buffers() if getattr(buf, 'is_meta', False)]
    for buffer_name in meta_buffers:
        parent_name, _, local_name = buffer_name.rpartition('.')
        module = model.get_submodule(parent_name) if parent_name else model
        if local_name == 'inv_freq' and hasattr(module, 'rope_init_fn') and hasattr(module, 'config'):
            inv_freq, attention_scaling = module.rope_init_fn(module.config, torch.device(device))
            module.register_buffer('inv_freq', inv_freq, persistent=False)
            module.original_inv_freq = module.inv_freq
            module.attention_scaling = attention_scaling
            continue
        raise ValueError(f'Unhandled meta buffer after streaming load: {buffer_name}')

    model.eval()
    _patch_block_diffusion_3d_mask(model)
    if getattr(model, 'generation_config', None) is not None:
        model.generation_config.do_sample = False
    return model


def run_setting(
    model: SDARGapwithChatTemplate,
    input_record: list[dict],
    gold_answer: str,
    output_dir: Path,
    *,
    setting_name: str,
    remask_threshold: float,
    remask_interval_blocks: int,
    remask_window_blocks: int,
    remask_start_tokens: int,
    remask_prefix_guard_tokens: int,
    remask_tail_guard_blocks: int,
) -> dict:
    trace_path = output_dir / f'{setting_name}_trace.jsonl'
    event_trace_path = output_dir / f'{setting_name}_event_trace.jsonl'
    for path in (trace_path, event_trace_path):
        if path.exists():
            path.unlink()

    os.environ['SDAR_REMASK_TRACE_PATH'] = str(trace_path)
    os.environ['SDAR_REMASK_EVENT_TRACE_PATH'] = str(event_trace_path)

    generation_settings = dict(
        mask_id=model.model.config.mask_token_id,
        remasking_strategy=getattr(
            model.model.config,
            'gap_rollout_strategy',
            'low_confidence_dynamic',
        ),
        max_out_len=1536,
        decode_backend='gap',
        block_length=4,
        denoising_steps=4,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        confidence_threshold=0.95,
        remask_threshold=remask_threshold,
        remask_start_ratio=0.0,
        remask_interval_blocks=remask_interval_blocks,
        remask_window_blocks=remask_window_blocks,
        remask_start_tokens=remask_start_tokens,
        remask_prefix_guard_tokens=remask_prefix_guard_tokens,
        remask_tail_guard_blocks=remask_tail_guard_blocks,
    )
    write_json(output_dir / f'{setting_name}_settings.json', generation_settings)

    prediction = model.generate(
        [input_record],
        **generation_settings,
    )[0]

    pred_processed = math_postprocess_sdar(prediction)
    gold_processed = gsm8k_dataset_postprocess(gold_answer)
    result = MATHEvaluator(version='v2').score([pred_processed], [gold_processed])

    prediction_payload = {
        '0': {
            'origin_prompt': input_record,
            'prediction': prediction,
            'gold': gold_answer,
        }
    }
    result_payload = {
        'accuracy': result.get('accuracy'),
        'details': result.get('details'),
    }
    write_json(output_dir / f'{setting_name}_predictions.json', prediction_payload)
    write_json(output_dir / f'{setting_name}_results.json', result_payload)

    return {
        'prediction_path': str(output_dir / f'{setting_name}_predictions.json'),
        'result_path': str(output_dir / f'{setting_name}_results.json'),
        'settings_path': str(output_dir / f'{setting_name}_settings.json'),
        'trace_path': str(trace_path),
        'event_trace_path': str(event_trace_path),
    }


def main() -> None:
    args = parse_args()
    example = load_example(args.gsm8k_test, args.example_index)
    input_record = build_input_record(example['question'])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = SDARGapwithChatTemplate(
        path=args.model_path,
        tokenizer_only=True,
        model_kwargs={},
        tokenizer_kwargs={},
        generation_kwargs={},
        max_seq_len=32768,
        stop_words=[],
    )
    wrapper.model = build_stream_loaded_model(args.model_path, args.device)

    noremask_files = run_setting(
        wrapper,
        input_record,
        example['answer'],
        output_dir,
        setting_name='noremask',
        remask_threshold=1.0,
        remask_interval_blocks=1,
        remask_window_blocks=5,
        remask_start_tokens=192,
        remask_prefix_guard_tokens=192,
        remask_tail_guard_blocks=1,
    )
    remask_files = run_setting(
        wrapper,
        input_record,
        example['answer'],
        output_dir,
        setting_name='remask',
        remask_threshold=0.30,
        remask_interval_blocks=2,
        remask_window_blocks=3,
        remask_start_tokens=192,
        remask_prefix_guard_tokens=192,
        remask_tail_guard_blocks=1,
    )

    summary = {
        'example_index': args.example_index,
        'question': example['question'],
        'gold_answer': example['answer'],
        'noremask': noremask_files,
        'remask': remask_files,
    }
    write_json(output_dir / 'summary.json', summary)
    print(output_dir / 'summary.json')


if __name__ == '__main__':
    main()
