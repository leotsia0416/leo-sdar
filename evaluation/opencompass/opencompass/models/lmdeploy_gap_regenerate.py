import json
import os
import subprocess
import sys
import traceback
import time
from typing import List

import torch

from .huggingface_above_v4_33 import _convert_chat_messages, _format_with_fast_chat_template
from .lmdeploy_with_tf_above_v4_33 import GenerationConfig, LMDeploywithChatTemplate


class LMDeployGapRegeneratewithChatTemplate(LMDeploywithChatTemplate):
    """LMDeploy generation with external rollback/regenerate on the last block.

    This route does not mutate LMDeploy's internal DLLM mask state. It uses
    LMDeploy for block-wise generation, and a separate HF scorer to decide
    whether the latest generated block should be rolled back and regenerated.
    """

    def _load_model(self, path: str, added_model_kwargs: dict = dict()):
        added_model_kwargs = added_model_kwargs.copy()
        self.gap_remask_threshold = float(added_model_kwargs.pop('gap_remask_threshold', 0.5))
        self.gap_remask_interval_blocks = int(added_model_kwargs.pop('gap_remask_interval_blocks', 1))
        self.gap_remask_start_block = int(added_model_kwargs.pop('gap_remask_start_block', 0))
        self.gap_remask_max_tokens_per_block = int(
            added_model_kwargs.pop('gap_remask_max_tokens_per_block', 0)
        )
        self.gap_regenerate_window_blocks = int(added_model_kwargs.pop('gap_regenerate_window_blocks', 1))
        self.gap_regenerate_check_interval_blocks = int(
            added_model_kwargs.pop('gap_regenerate_check_interval_blocks', 0)
        )
        self.gap_regenerate_max_total_rollbacks = int(
            added_model_kwargs.pop('gap_regenerate_max_total_rollbacks', 1)
        )
        self.gap_regenerate_max_total_tokens = int(
            added_model_kwargs.pop('gap_regenerate_max_total_tokens', 4)
        )
        self.gap_score_subprocess = None
        super()._load_model(path, added_model_kwargs)
        self._start_gap_scorer(path)

    def _worker_script_path(self) -> str:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        return os.path.join(repo_root, 'script', 'gap_regen_score_worker.py')

    def _start_gap_scorer(self, path: str):
        device_name = os.getenv('SDAR_LMDEPLOY_GAP_SCORE_DEVICE', 'cuda:0').strip() or 'cuda:0'
        if device_name.startswith('cuda') and not torch.cuda.is_available():
            device_name = 'cpu'
        self.gap_score_device = torch.device(device_name)
        worker_script = self._worker_script_path()
        worker_env = os.environ.copy()
        if device_name.startswith('cuda'):
            visible = worker_env.get('CUDA_VISIBLE_DEVICES', '').strip()
            slurm_visible = (
                worker_env.get('SLURM_STEP_GPUS', '').strip()
                or worker_env.get('SLURM_JOB_GPUS', '').strip()
            )
            requested_idx = 0
            if ':' in device_name:
                try:
                    requested_idx = int(device_name.split(':', 1)[1])
                except ValueError:
                    requested_idx = 0
            source = visible or slurm_visible
            if source:
                gpu_list = [item.strip() for item in source.split(',') if item.strip()]
                if gpu_list:
                    bound_idx = min(max(requested_idx, 0), len(gpu_list) - 1)
                    worker_env['CUDA_VISIBLE_DEVICES'] = gpu_list[bound_idx]
                    device_name = 'cuda:0'
        worker_env['SDAR_GAP_SCORE_MODEL_PATH'] = path
        worker_env['SDAR_GAP_SCORE_DEVICE'] = device_name
        worker_env['PYTHONUNBUFFERED'] = '1'
        worker_env['TOKENIZERS_PARALLELISM'] = 'false'
        worker_cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        self.gap_score_subprocess = subprocess.Popen(
            [sys.executable, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=worker_cwd,
            env=worker_env,
        )
        ready_line = self.gap_score_subprocess.stdout.readline().strip()
        if not ready_line:
            stderr_output = self.gap_score_subprocess.stderr.read().strip()
            raise RuntimeError(f'Gap scorer worker failed to start: {stderr_output}')
        ready = json.loads(ready_line)
        if ready.get('status') != 'ready':
            raise RuntimeError(f'Gap scorer worker init failed: {ready}')

    def _query_gap_scorer(self, prompt_ids: torch.LongTensor, generated_ids: torch.LongTensor):
        if self.gap_score_subprocess is None or self.gap_score_subprocess.stdin is None:
            raise RuntimeError('Gap scorer worker is not available.')
        payload = dict(
            cmd='score_window',
            prompt_ids=prompt_ids.tolist(),
            generated_ids=generated_ids.tolist(),
            block_length=int(os.getenv('SDAR_BLOCK_LENGTH', '4')),
            threshold=self.gap_remask_threshold,
            start_block=self.gap_remask_start_block,
            interval_blocks=self.gap_remask_interval_blocks,
            window_blocks=self.gap_regenerate_window_blocks,
            max_tokens_per_block=self.gap_remask_max_tokens_per_block,
        )
        self.gap_score_subprocess.stdin.write(json.dumps(payload, ensure_ascii=True) + '\n')
        self.gap_score_subprocess.stdin.flush()
        line = self.gap_score_subprocess.stdout.readline()
        if not line:
            stderr_output = ''
            if self.gap_score_subprocess.stderr is not None:
                stderr_output = self.gap_score_subprocess.stderr.read().strip()
            raise RuntimeError(f'Gap scorer worker exited unexpectedly. stderr={stderr_output}')
        response = json.loads(line)
        if response.get('status') != 'ok':
            raise RuntimeError(f'Gap scorer worker error: {response}')
        return response.get('proposal')

    def _build_prompts(self, inputs: List[str]) -> List[str]:
        messages = _convert_chat_messages(inputs)
        if self.fastchat_template:
            messages = _format_with_fast_chat_template(messages, self.fastchat_template)
        else:
            messages = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]
            if self.tokenizer.bos_token:
                bos_token = self.tokenizer.bos_token
                messages = [message.removeprefix(bos_token) if message.startswith(bos_token) else message for message in messages]
        return messages

    def _make_generation_config(self, max_out_len: int, stopping_criteria: List[str], **kwargs):
        default_generation_kwargs = {
            'temperature': 0,
            'max_new_tokens': max_out_len,
            'stop_words': list(set(self.stop_words + stopping_criteria)),
        }
        sampling_kwargs = default_generation_kwargs.copy()
        sampling_kwargs.update(self.generation_kwargs)
        sampling_kwargs.update(kwargs)
        return GenerationConfig(**sampling_kwargs)

    def _retokenize_generated_ids(self, prompt: str, generated_text: str) -> torch.LongTensor:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        full_ids = self.tokenizer(prompt + generated_text, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        return full_ids[len(prompt_ids):]

    def _proposed_remask_window(self, prompt_ids: torch.LongTensor, generated_ids: torch.LongTensor):
        return self._query_gap_scorer(prompt_ids, generated_ids)

    def _should_score_window(self, generated_ids: torch.LongTensor, block_length: int) -> bool:
        num_generated_tokens = int(generated_ids.numel())
        if num_generated_tokens < block_length:
            return False
        num_generated_blocks = num_generated_tokens // block_length
        if num_generated_blocks <= 0:
            return False
        last_block_idx = num_generated_blocks - 1
        first_block_idx = max(0, last_block_idx - self.gap_regenerate_window_blocks + 1)
        if last_block_idx < self.gap_remask_start_block:
            return False
        # Skip scorer entirely when the trailing window contains no block that could ever trigger.
        eligible_in_window = False
        for block_idx in range(first_block_idx, last_block_idx + 1):
            if block_idx < self.gap_remask_start_block:
                continue
            if (block_idx - self.gap_remask_start_block) % self.gap_remask_interval_blocks == 0:
                eligible_in_window = True
                break
        if not eligible_in_window:
            return False
        check_interval = self.gap_regenerate_check_interval_blocks
        if check_interval <= 0:
            check_interval = self.gap_regenerate_window_blocks
        check_interval = max(1, min(check_interval, self.gap_regenerate_window_blocks))
        return ((last_block_idx + 1) % check_interval) == 0

    def _append_trace(self, trace_record: dict):
        trace_path = os.getenv('SDAR_LMDEPLOY_REMASK_TRACE_PATH', '').strip()
        if not trace_path:
            return
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        with open(trace_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trace_record, ensure_ascii=True) + '\n')

    def generate(self, inputs, max_out_len, stopping_criteria=[], **kwargs):
        prompts = self._build_prompts(inputs)
        block_length = int(os.getenv('SDAR_BLOCK_LENGTH', '4'))
        prompt_ids = [
            self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
            for prompt in prompts
        ]
        states = []
        start_time = time.time()
        for idx, prompt in enumerate(prompts):
            states.append(
                dict(
                    prompt=prompt,
                    prompt_ids=prompt_ids[idx],
                    generated_ids=torch.empty((0,), dtype=prompt_ids[idx].dtype),
                    applied_rollbacks=0,
                    applied_rollback_tokens=0,
                    proposed_steps=0,
                    proposed_tokens=0,
                    proposed_block_ids=[],
                    rollback_window_blocks=[],
                    done=False,
                    output_text='',
                )
            )
        self.logger.info(
            'GAP external regenerate begin: batch=%d max_out_len=%s',
            len(prompts),
            max_out_len,
        )
        try:
            active = list(range(len(states)))
            while active:
                batch_prompts = []
                batch_prefixes = []
                next_active = []
                chunk_len = block_length
                for idx in active:
                    state = states[idx]
                    remaining = max_out_len - int(state['generated_ids'].numel())
                    if remaining <= 0 or state['done']:
                        state['done'] = True
                        continue
                    chunk_len = min(chunk_len, max(1, remaining))
                    prefix_text = self.tokenizer.decode(state['generated_ids'], skip_special_tokens=False)
                    batch_prefixes.append(prefix_text)
                    batch_prompts.append(state['prompt'] + prefix_text)
                    next_active.append(idx)
                active = next_active
                if not active:
                    break
                gen_config = self._make_generation_config(chunk_len, stopping_criteria, **kwargs)
                batch_outputs = self.pipe(batch_prompts, gen_config=gen_config)
                new_active = []
                for local_idx, idx in enumerate(active):
                    state = states[idx]
                    chunk_text = batch_outputs[local_idx].text
                    if not chunk_text:
                        state['done'] = True
                        continue
                    prev_generated_len = int(state['generated_ids'].numel())
                    candidate_text = batch_prefixes[local_idx] + chunk_text
                    candidate_ids = self._retokenize_generated_ids(state['prompt'], candidate_text)
                    if candidate_ids.numel() <= prev_generated_len:
                        state['done'] = True
                        continue
                    state['generated_ids'] = candidate_ids[:max_out_len]
                    proposal = None
                    if self._should_score_window(state['generated_ids'], block_length):
                        proposal = self._proposed_remask_window(state['prompt_ids'], state['generated_ids'])
                    if proposal is not None:
                        state['proposed_steps'] += 1
                        state['proposed_tokens'] += int(proposal['proposed_tokens'])
                        state['proposed_block_ids'].extend(
                            int(block['generated_block_idx']) for block in proposal.get('triggered_blocks', [])
                        )
                        can_rollback = (
                            proposal['proposed_tokens'] > 0
                            and state['applied_rollbacks'] < self.gap_regenerate_max_total_rollbacks
                            and int(state['generated_ids'].numel()) >= block_length
                        )
                        if can_rollback:
                            available_blocks = max(1, int(state['generated_ids'].numel()) // block_length)
                            rollback_blocks = min(self.gap_regenerate_window_blocks, available_blocks)
                            rollback_tokens = rollback_blocks * block_length
                            remaining_budget = self.gap_regenerate_max_total_tokens - state['applied_rollback_tokens']
                            rollback_tokens = min(rollback_tokens, remaining_budget, int(state['generated_ids'].numel()))
                            can_rollback = rollback_tokens >= block_length
                        if can_rollback:
                            state['generated_ids'] = state['generated_ids'][:-rollback_tokens]
                            state['applied_rollbacks'] += 1
                            state['applied_rollback_tokens'] += rollback_tokens
                            state['rollback_window_blocks'].append(int(rollback_tokens // block_length))
                            new_active.append(idx)
                            continue
                    added_tokens = int(state['generated_ids'].numel()) - prev_generated_len
                    if int(state['generated_ids'].numel()) < max_out_len and added_tokens >= chunk_len:
                        new_active.append(idx)
                    else:
                        state['done'] = True
                active = new_active
        except Exception:
            for state in states:
                self._append_trace(
                    dict(
                        gap_route='external_regenerate',
                        error='generate_batch_exception',
                        prompt_chars=len(state['prompt']),
                        generated_tokens=int(state['generated_ids'].numel()),
                        proposed_steps=state['proposed_steps'],
                        proposed_tokens=state['proposed_tokens'],
                        proposed_block_ids=state['proposed_block_ids'],
                        rollback_window_blocks=state['rollback_window_blocks'],
                        applied_rollbacks=state['applied_rollbacks'],
                        applied_rollback_tokens=state['applied_rollback_tokens'],
                        traceback=traceback.format_exc(),
                    )
                )
            self.logger.exception('GAP external regenerate batch failed: batch=%d', len(prompts))
            raise
        outputs = []
        total_elapsed = time.time() - start_time
        for state in states:
            output_text = self.tokenizer.decode(state['generated_ids'], skip_special_tokens=False)
            for stop in stopping_criteria:
                output_text = output_text.split(stop)[0]
            state['output_text'] = output_text
            outputs.append(output_text)
            self._append_trace(
                dict(
                    batch_size=len(prompts),
                    max_out_len=max_out_len,
                    gap_route='external_regenerate',
                    gap_remask_threshold=self.gap_remask_threshold,
                    gap_remask_start_block=self.gap_remask_start_block,
                    gap_remask_interval_blocks=self.gap_remask_interval_blocks,
                    gap_regenerate_window_blocks=self.gap_regenerate_window_blocks,
                    gap_regenerate_check_interval_blocks=self.gap_regenerate_check_interval_blocks or self.gap_regenerate_window_blocks,
                    gap_regenerate_max_total_rollbacks=self.gap_regenerate_max_total_rollbacks,
                    gap_regenerate_max_total_tokens=self.gap_regenerate_max_total_tokens,
                    gap_remask_max_tokens_per_block=self.gap_remask_max_tokens_per_block,
                    proposed_steps=state['proposed_steps'],
                    proposed_tokens=state['proposed_tokens'],
                    proposed_block_ids=state['proposed_block_ids'],
                    rollback_window_blocks=state['rollback_window_blocks'],
                    applied_rollbacks=state['applied_rollbacks'],
                    applied_rollback_tokens=state['applied_rollback_tokens'],
                    prompt_chars=[len(state['prompt'])],
                    output_chars=[len(output_text)],
                    elapsed=total_elapsed,
                )
            )
        self.logger.info('GAP external regenerate end: batch=%d', len(prompts))
        return outputs

    def __del__(self):
        proc = getattr(self, 'gap_score_subprocess', None)
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({'cmd': 'shutdown'}) + '\n')
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
