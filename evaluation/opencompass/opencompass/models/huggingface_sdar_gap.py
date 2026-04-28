import json
import math
import os
from typing import List, Optional

import torch
from transformers.cache_utils import DynamicCache

from opencompass.registry import MODELS

from .huggingface_bd3 import (
    BD3withChatTemplate,
    _build_stop_token_sequences,
    _convert_chat_messages,
    _format_with_fast_chat_template,
    _get_stopping_criteria,
    _match_stop_sequences,
    _prepare_block_diffusion_batch,
    block_diffusion_generate,
    get_num_transfer_tokens,
    sample_with_temperature_topk_topp,
)


def _collect_trace_token_snapshots(
    x: torch.LongTensor,
    window_inputs: torch.LongTensor,
    prompt_length: int,
    window_token_start: int,
):
    generated_window_start = max(prompt_length - window_token_start, 0)
    generated_window = window_inputs[:, generated_window_start:]
    prefix = x[:, prompt_length:window_token_start]
    if prefix.shape[1] > 0:
        generated_prefix = torch.cat((prefix, generated_window), dim=1)
    else:
        generated_prefix = generated_window
    return generated_prefix.detach().cpu().tolist(), generated_window.detach().cpu().tolist()


def _map_gap_strategy_to_bd3(strategy: str) -> str:
    if strategy == 'random':
        return 'random'
    return 'low_confidence'


def _select_block_positions(
    block_scores: torch.FloatTensor,
    masked_local_indices: torch.LongTensor,
    num_transfer_tokens: int,
    strategy: str,
    confidence_threshold: float,
) -> torch.LongTensor:
    k = min(num_transfer_tokens, int(masked_local_indices.numel()))
    if k <= 0:
        return masked_local_indices[:0]

    if strategy == 'low_confidence_dynamic':
        high_conf_mask = block_scores > confidence_threshold
        if int(high_conf_mask.sum().item()) >= num_transfer_tokens:
            return masked_local_indices[high_conf_mask]

        topk = torch.topk(block_scores, k=k, sorted=False).indices
        return masked_local_indices[topk]
    if strategy == 'low_confidence_static':
        topk = torch.topk(block_scores, k=k, sorted=False).indices
        return masked_local_indices[topk]
    if strategy == 'sequential':
        return masked_local_indices[:k]

    raise ValueError(f'Unsupported rollout strategy: {strategy}')


def _select_window_transfer_tokens(
    masked_indices: torch.BoolTensor,
    proposal_scores_full: torch.FloatTensor,
    block_size: int,
    num_transfer_tokens: int,
    strategy: str,
    confidence_threshold: float,
) -> torch.BoolTensor:
    reveal_mask = torch.zeros_like(masked_indices)
    if num_transfer_tokens <= 0:
        return reveal_mask

    batch_size, seq_len = masked_indices.shape
    for batch_idx in range(batch_size):
        for block_start in range(0, seq_len, block_size):
            block_end = min(block_start + block_size, seq_len)
            block_mask = masked_indices[batch_idx, block_start:block_end]
            if not block_mask.any():
                continue

            masked_local_indices = torch.nonzero(block_mask, as_tuple=False).flatten()
            block_scores = proposal_scores_full[batch_idx, block_start:block_end][masked_local_indices]
            chosen = _select_block_positions(
                block_scores=block_scores,
                masked_local_indices=masked_local_indices,
                num_transfer_tokens=num_transfer_tokens,
                strategy=strategy,
                confidence_threshold=confidence_threshold,
            )
            reveal_mask[batch_idx, block_start:block_end][chosen] = True

    return reveal_mask


@torch.no_grad()
def _decode_masked_window(
    model,
    window_inputs,
    window_attention_mask,
    window_position_ids,
    prefix_cache,
    mask_id,
    block_length,
    denoising_steps,
    temperature,
    top_k,
    top_p,
    rollout_strategy,
    confidence_threshold,
):
    rollout_steps = max(1, int(denoising_steps))
    transfer_schedule = get_num_transfer_tokens(block_length, rollout_steps).to(window_inputs.device)
    current_inputs = window_inputs.clone()
    current_stage = 0

    while True:
        masked_indices = current_inputs.eq(mask_id)
        if not masked_indices.any():
            break

        outputs = model(
            current_inputs,
            attention_mask=window_attention_mask,
            position_ids=window_position_ids,
            past_key_values=prefix_cache,
            use_cache=True,
            store_kv=False,
        )
        logits = outputs.logits
        logits[..., mask_id] = float('-inf')

        proposal_ids, proposal_scores = sample_with_temperature_topk_topp(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        proposal_ids = torch.where(masked_indices, proposal_ids, current_inputs)
        proposal_scores_full = torch.where(
            masked_indices,
            proposal_scores,
            torch.full_like(proposal_scores, float('-inf')),
        )

        if current_stage >= rollout_steps:
            current_inputs[masked_indices] = proposal_ids[masked_indices]
            break

        reveal_mask = _select_window_transfer_tokens(
            masked_indices=masked_indices,
            proposal_scores_full=proposal_scores_full,
            block_size=block_length,
            num_transfer_tokens=int(transfer_schedule[current_stage].item()),
            strategy=rollout_strategy,
            confidence_threshold=confidence_threshold,
        )
        fill_mask = reveal_mask if reveal_mask.any() else masked_indices
        current_inputs[fill_mask] = proposal_ids[fill_mask]
        current_stage += 1

    return current_inputs


@torch.no_grad()
def _select_window_remask_tokens(
    model,
    window_inputs,
    window_attention_mask,
    window_position_ids,
    prefix_cache,
    candidate_mask,
    remask_threshold,
):
    batch_size = window_inputs.shape[0]
    remask_mask = torch.zeros_like(candidate_mask)
    best_scores = torch.full(
        (batch_size,),
        float('-inf'),
        dtype=torch.float32,
        device=window_inputs.device,
    )
    candidate_counts = candidate_mask.sum(dim=1, dtype=torch.long)
    if not candidate_mask.any():
        return remask_mask, best_scores, candidate_counts

    outputs = model(
        window_inputs,
        attention_mask=window_attention_mask,
        position_ids=window_position_ids,
        past_key_values=prefix_cache,
        use_cache=True,
        store_kv=False,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1]
    remask_probs = torch.sigmoid(model.gap_remask_head(hidden_states).squeeze(-1))
    candidate_scores = remask_probs.masked_fill(~candidate_mask, float('-inf'))
    best_scores = candidate_scores.max(dim=1).values
    remask_mask = candidate_mask & remask_probs.ge(remask_threshold)
    return remask_mask, best_scores, candidate_counts


@torch.no_grad()
def _commit_cached_block(
    model,
    cache,
    x,
    block_diffusion_attention_mask,
    position_ids,
    block_start,
    block_end,
):
    if block_end <= block_start:
        return

    cur_x = x[:, block_start:block_end]
    cur_attn_mask = block_diffusion_attention_mask[:, :, block_start:block_end, :block_end]
    cur_position_ids = position_ids[:, block_start:block_end]
    model(
        cur_x,
        attention_mask=cur_attn_mask,
        position_ids=cur_position_ids,
        past_key_values=cache,
        use_cache=True,
        store_kv=True,
    )


@torch.no_grad()
def block_diffusion_gap_generate(
    model,
    tokenizer,
    prompt,
    mask_id,
    gen_length=128,
    block_length=8,
    denoising_steps=8,
    temperature=0.0,
    top_k=0,
    top_p=1.0,
    remasking_strategy='low_confidence_dynamic',
    confidence_threshold=1.0,
    remask_threshold=0.15,
    remask_start_ratio=0.0,
    remask_interval_blocks=1,
    remask_window_blocks=3,
    remask_start_tokens=None,
    remask_prefix_guard_tokens=None,
    remask_tail_guard_blocks=0,
    stopping_criteria_sequences=None,
    trace_inputs=None,
):
    input_ids, prompt_length, total_length, block_diffusion_attention_mask, position_ids, x = (
        _prepare_block_diffusion_batch(prompt, mask_id, gen_length, block_length, model.device)
    )
    batch_size = input_ids.shape[0]

    num_blocks = total_length // block_length
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length
    prefix_cache = DynamicCache()
    gap_enabled = bool(getattr(model.config, 'gap_enable', False)) and hasattr(model, 'gap_remask_head')
    remask_window_blocks = max(1, int(remask_window_blocks))
    remask_interval_blocks = max(1, int(remask_interval_blocks))
    remask_start_tokens = (
        None if remask_start_tokens is None else max(0, int(remask_start_tokens))
    )
    if remask_prefix_guard_tokens is None:
        remask_prefix_guard_tokens = remask_start_tokens or 0
    else:
        remask_prefix_guard_tokens = max(0, int(remask_prefix_guard_tokens))
    remask_tail_guard_blocks = max(0, int(remask_tail_guard_blocks))
    remask_disabled = (not gap_enabled) or remask_threshold >= 1.0
    if remask_disabled:
        gap_enabled = False
        # No remask means no future rollback, so we can commit every finished block immediately.
        remask_window_blocks = 1
    total_generation_blocks = max(num_blocks - prefill_blocks, 1)
    if remask_start_tokens is None:
        first_remask_block = max(
            remask_window_blocks,
            int(math.ceil(total_generation_blocks * remask_start_ratio)),
        )
    else:
        first_remask_block = max(
            remask_window_blocks,
            int(math.ceil(remask_start_tokens / block_length)),
        )
    window_start_block = prefill_blocks
    steps_with_remask = torch.zeros(batch_size, dtype=torch.long, device=x.device)
    total_remasked_tokens = torch.zeros(batch_size, dtype=torch.long, device=x.device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
    event_trace_path = os.getenv('SDAR_REMASK_EVENT_TRACE_PATH')
    event_trace_records = [] if event_trace_path else None
    if trace_inputs is None:
        trace_inputs = [None] * batch_size

    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:, :, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=prefix_cache,
            use_cache=True,
            store_kv=True,
        )

    for block_idx in range(prefill_blocks, num_blocks):
        if bool(finished.all().item()):
            break

        while block_idx - window_start_block >= remask_window_blocks:
            commit_start = window_start_block * block_length
            commit_end = commit_start + block_length
            _commit_cached_block(
                model=model,
                cache=prefix_cache,
                x=x,
                block_diffusion_attention_mask=block_diffusion_attention_mask,
                position_ids=position_ids,
                block_start=commit_start,
                block_end=commit_end,
            )
            window_start_block += 1

        window_token_start = window_start_block * block_length
        window_token_end = (block_idx + 1) * block_length
        window_slice = slice(window_token_start, window_token_end)
        frozen_window_inputs = x[:, window_slice].clone()
        window_inputs = x[:, window_slice].clone()
        window_attention_mask = block_diffusion_attention_mask[:, :, window_token_start:window_token_end, :window_token_end]
        window_position_ids = position_ids[:, window_slice]

        window_inputs = _decode_masked_window(
            model=model,
            window_inputs=window_inputs,
            window_attention_mask=window_attention_mask,
            window_position_ids=window_position_ids,
            prefix_cache=prefix_cache,
            mask_id=mask_id,
            block_length=block_length,
            denoising_steps=denoising_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rollout_strategy=remasking_strategy,
            confidence_threshold=confidence_threshold,
        )
        trace_records_by_row = None
        if event_trace_records is not None:
            full_before_token_ids, window_before_token_ids = _collect_trace_token_snapshots(
                x=x,
                window_inputs=window_inputs,
                prompt_length=prompt_length,
                window_token_start=window_token_start,
            )
            trace_records_by_row = {}
            for row_idx in range(batch_size):
                if bool(finished[row_idx].item()):
                    continue
                trace_records_by_row[row_idx] = {
                    'input': trace_inputs[row_idx],
                    'block_idx': int(block_idx),
                    'window_start_block': int(window_start_block),
                    'window_token_start': int(window_token_start),
                    'window_token_end': int(window_token_end),
                    'generated_blocks': None,
                    'total_generation_blocks': int(total_generation_blocks),
                    'first_remask_block': int(first_remask_block),
                    'remask_progress': None,
                    'prompt_length': int(prompt_length),
                    'remask_active': False,
                    'candidate_count': 0,
                    'best_score': None,
                    'score_margin': None,
                    'remask_threshold': float(remask_threshold),
                    'candidate_positions': [],
                    'remasked_positions': [],
                    'triggered': False,
                    'remasked_tokens': 0,
                    'generated_before_token_ids': full_before_token_ids[row_idx],
                    'window_before_token_ids': window_before_token_ids[row_idx],
                    'generated_with_masks_token_ids': None,
                    'window_with_masks_token_ids': None,
                    'generated_after_token_ids': full_before_token_ids[row_idx],
                    'window_after_token_ids': window_before_token_ids[row_idx],
                }

        generated_blocks = (block_idx - prefill_blocks) + 1
        remask_progress = generated_blocks / total_generation_blocks
        remask_active = gap_enabled and generated_blocks >= first_remask_block
        if remask_start_tokens is None:
            remask_active = remask_active and remask_progress >= remask_start_ratio
        remask_active = remask_active and (generated_blocks - first_remask_block) % remask_interval_blocks == 0
        if trace_records_by_row is not None:
            for row_idx, record in trace_records_by_row.items():
                record['generated_blocks'] = int(generated_blocks)
                record['remask_progress'] = float(remask_progress)
                record['remask_active'] = bool(remask_active)

        if remask_active:
            stop_finished = torch.zeros_like(finished)
            if stopping_criteria_sequences:
                generated_prefix = torch.cat(
                    (
                        x[:, prompt_length:window_token_start],
                        window_inputs[:, max(prompt_length - window_token_start, 0):],
                    ),
                    dim=1,
                )
                stop_finished = _match_stop_sequences(generated_prefix, stopping_criteria_sequences)
            global_positions = torch.arange(window_token_start, window_token_end, device=x.device).unsqueeze(0)
            active_rows = ~(finished | stop_finished)
            candidate_mask = (
                global_positions.ge(prompt_length + remask_prefix_guard_tokens)
                & window_inputs.ne(mask_id)
                & active_rows.unsqueeze(1)
            )
            if remask_tail_guard_blocks > 0:
                tail_guard_start = window_token_end - remask_tail_guard_blocks * block_length
                candidate_mask &= global_positions.lt(tail_guard_start)
            remask_mask, best_scores, candidate_counts = _select_window_remask_tokens(
                model=model,
                window_inputs=window_inputs,
                window_attention_mask=window_attention_mask,
                window_position_ids=window_position_ids,
                prefix_cache=prefix_cache,
                candidate_mask=candidate_mask,
                remask_threshold=remask_threshold,
            )
            remask_counts = remask_mask.sum(dim=1)
            if trace_records_by_row is not None:
                for row_idx in range(batch_size):
                    if not bool(active_rows[row_idx].item()):
                        continue
                    best_score_value = None
                    score_margin = None
                    if torch.isfinite(best_scores[row_idx]):
                        best_score_value = float(best_scores[row_idx].item())
                        score_margin = best_score_value - float(remask_threshold)
                    record = trace_records_by_row.get(row_idx)
                    if record is None:
                        continue
                    record['candidate_count'] = int(candidate_counts[row_idx].item())
                    record['best_score'] = best_score_value
                    record['score_margin'] = score_margin
                    record['triggered'] = bool(remask_counts[row_idx].item() > 0)
                    record['remasked_tokens'] = int(remask_counts[row_idx].item())
                    record['candidate_positions'] = [
                        int(pos)
                        for pos in global_positions[0][candidate_mask[row_idx]].detach().cpu().tolist()
                    ]
                    record['remasked_positions'] = [
                        int(pos)
                        for pos in global_positions[0][remask_mask[row_idx]].detach().cpu().tolist()
                    ]
            if remask_counts.any():
                # In inference we remask only the tokens the head flags inside the latest window.
                window_inputs[remask_mask] = mask_id
                if trace_records_by_row is not None:
                    full_masked_token_ids, window_masked_token_ids = _collect_trace_token_snapshots(
                        x=x,
                        window_inputs=window_inputs,
                        prompt_length=prompt_length,
                        window_token_start=window_token_start,
                    )
                    for row_idx, record in trace_records_by_row.items():
                        if not record['triggered']:
                            continue
                        record['generated_with_masks_token_ids'] = full_masked_token_ids[row_idx]
                        record['window_with_masks_token_ids'] = window_masked_token_ids[row_idx]
                steps_with_remask += remask_counts.gt(0).to(dtype=torch.long)
                total_remasked_tokens += remask_counts.to(dtype=torch.long)
                window_inputs = _decode_masked_window(
                    model=model,
                    window_inputs=window_inputs,
                    window_attention_mask=window_attention_mask,
                    window_position_ids=window_position_ids,
                    prefix_cache=prefix_cache,
                    mask_id=mask_id,
                    block_length=block_length,
                    denoising_steps=denoising_steps,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    rollout_strategy=remasking_strategy,
                    confidence_threshold=confidence_threshold,
                )
                if trace_records_by_row is not None:
                    full_after_token_ids, window_after_token_ids = _collect_trace_token_snapshots(
                        x=x,
                        window_inputs=window_inputs,
                        prompt_length=prompt_length,
                        window_token_start=window_token_start,
                    )
                    for row_idx, record in trace_records_by_row.items():
                        if not record['triggered']:
                            continue
                        record['generated_after_token_ids'] = full_after_token_ids[row_idx]
                        record['window_after_token_ids'] = window_after_token_ids[row_idx]

        if finished.any():
            window_inputs[finished] = frozen_window_inputs[finished]
        x[:, window_slice] = window_inputs
        if trace_records_by_row is not None:
            for row_idx in sorted(trace_records_by_row):
                event_trace_records.append(trace_records_by_row[row_idx])

        if stopping_criteria_sequences:
            generated_prefix = x[:, prompt_length:window_token_end]
            finished |= _match_stop_sequences(generated_prefix, stopping_criteria_sequences)
        if bool(finished.all().item()):
            break

    trace_path = os.getenv('SDAR_REMASK_TRACE_PATH')
    if trace_path:
        with open(trace_path, 'a', encoding='utf-8') as f:
            for row_idx in range(batch_size):
                record = {
                    'input': trace_inputs[row_idx],
                    'steps_with_remask': int(steps_with_remask[row_idx].item()),
                    'total_remasked_tokens': int(total_remasked_tokens[row_idx].item()),
                    'gap_enabled': gap_enabled,
                    'remask_start_ratio': remask_start_ratio,
                    'remask_start_tokens': remask_start_tokens,
                    'remask_prefix_guard_tokens': remask_prefix_guard_tokens,
                    'remask_interval_blocks': remask_interval_blocks,
                    'remask_window_blocks': remask_window_blocks,
                    'remask_tail_guard_blocks': remask_tail_guard_blocks,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    if event_trace_path and event_trace_records:
        with open(event_trace_path, 'a', encoding='utf-8') as f:
            for record in event_trace_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    return x


@MODELS.register_module()
class SDARGapwithChatTemplate(BD3withChatTemplate):
    def _generate_ar_baseline(
        self,
        tokens,
        batch_size: int,
        max_out_len: int,
        min_out_len: Optional[int],
        stopping_criteria: List[str],
        generation_kwargs,
    ) -> List[str]:
        hf_generation_kwargs = {}
        if stopping_criteria:
            hf_generation_kwargs['stopping_criteria'] = _get_stopping_criteria(
                stopping_criteria,
                self.tokenizer,
                batch_size,
            )
        if max_out_len is not None:
            hf_generation_kwargs['max_new_tokens'] = max_out_len
        if min_out_len is not None:
            hf_generation_kwargs['min_new_tokens'] = min_out_len
        hf_generation_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        hf_generation_kwargs['use_cache'] = True

        temperature = float(generation_kwargs.get('temperature', 0.0))
        top_k = int(generation_kwargs.get('top_k', 0))
        top_p = float(generation_kwargs.get('top_p', 1.0))
        do_sample = temperature > 0.0 or top_k > 1 or top_p < 1.0
        hf_generation_kwargs['do_sample'] = do_sample
        if do_sample:
            hf_generation_kwargs['temperature'] = max(temperature, 1e-5)
            if top_k > 0:
                hf_generation_kwargs['top_k'] = top_k
            if top_p < 1.0:
                hf_generation_kwargs['top_p'] = top_p

        outputs = self.model.generate(**tokens, **hf_generation_kwargs)
        outputs = outputs[:, tokens['input_ids'].shape[1]:]

        decodeds = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        decodeds = [t.replace('<|MASK|>', '') for t in decodeds]
        for stop in stopping_criteria:
            decodeds = [t.split(stop)[0] for t in decodeds]
        return decodeds

    def generate(
        self,
        inputs: List[str],
        max_out_len: int,
        min_out_len: Optional[int] = None,
        stopping_criteria: List[str] = [],
        **kwargs,
    ) -> List[str]:
        messages = _convert_chat_messages(inputs)
        batch_size = len(messages)

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len,
        )

        if self.fastchat_template:
            messages = _format_with_fast_chat_template(messages, self.fastchat_template)
            tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)
        else:
            messages = [
                self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                for m in messages
            ]
            tokenize_kwargs['add_special_tokens'] = False
            tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

        if self.mode == 'mid':
            max_prompt_len = self.max_seq_len - max_out_len
            half_max_prompt_len = max_prompt_len // 2
            if half_max_prompt_len > 0 and tokens['input_ids'].shape[1] > max_prompt_len:
                for key in tokens.keys():
                    if tokens[key].shape[1] > max_prompt_len:
                        field_values = tokens[key]
                        tokens[key] = torch.cat(
                            (field_values[:, :half_max_prompt_len], field_values[:, -half_max_prompt_len:]),
                            dim=1,
                        )

        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update(kwargs)
        decode_backend = os.getenv(
            'SDAR_DECODE_BACKEND',
            generation_kwargs.get('decode_backend', 'gap'),
        ).lower()

        stopping_criteria = list(set(stopping_criteria + self.stop_words))

        if decode_backend == 'ar':
            return self._generate_ar_baseline(
                tokens=tokens,
                batch_size=batch_size,
                max_out_len=max_out_len,
                min_out_len=min_out_len,
                stopping_criteria=stopping_criteria,
                generation_kwargs=generation_kwargs,
            )
        if decode_backend not in {'gap', 'bd3'}:
            raise ValueError(f'Unsupported decode backend: {decode_backend}')

        if stopping_criteria:
            generation_kwargs['stopping_criteria'] = _get_stopping_criteria(
                stopping_criteria, self.tokenizer, batch_size
            )
        if max_out_len is not None:
            generation_kwargs['max_new_tokens'] = max_out_len
        if min_out_len is not None:
            generation_kwargs['min_new_tokens'] = min_out_len
        generation_kwargs['pad_token_id'] = self.tokenizer.pad_token_id

        stopping_criteria_sequences = _build_stop_token_sequences(
            stopping_criteria, self.tokenizer
        )

        seq_len = tokens['input_ids'].shape[1]
        prompt_bucket_size = max(
            1,
            int(os.getenv('SDAR_PROMPT_BUCKET_SIZE', generation_kwargs.get('prompt_bucket_size', 32))),
        )
        if 'attention_mask' in tokens:
            prompt_lengths = tokens['attention_mask'].sum(dim=1, dtype=torch.long).tolist()
        else:
            prompt_lengths = [seq_len] * batch_size

        grouped_indices = {}
        for row_idx, prompt_len in enumerate(prompt_lengths):
            bucket_key = ((int(prompt_len) + prompt_bucket_size - 1) // prompt_bucket_size) * prompt_bucket_size
            grouped_indices.setdefault(bucket_key, []).append(row_idx)

        decodeds = [''] * batch_size
        for _, row_indices in sorted(grouped_indices.items()):
            row_index_tensor = torch.tensor(row_indices, dtype=torch.long, device=self.model.device)
            group_tokens = {}
            group_prompt_len = max(prompt_lengths[row_idx] for row_idx in row_indices)
            for key, value in tokens.items():
                group_value = value.index_select(0, row_index_tensor)
                if group_value.dim() >= 2 and group_value.shape[1] == seq_len:
                    # Tokenizer uses left padding, so keep the right-aligned prompt suffix.
                    group_value = group_value[:, -group_prompt_len:]
                group_tokens[key] = group_value

            if decode_backend == 'bd3':
                outputs = block_diffusion_generate(
                    self.model,
                    self.tokenizer,
                    group_tokens,
                    mask_id=generation_kwargs.get('mask_id', self.model.config.mask_token_id),
                    gen_length=generation_kwargs.get('gen_length', max_out_len),
                    block_length=generation_kwargs.get(
                        'block_length', getattr(self.model.config, 'block_size', 4)
                    ),
                    denoising_steps=generation_kwargs.get(
                        'denoising_steps',
                        generation_kwargs.get('block_length', getattr(self.model.config, 'block_size', 4)),
                    ),
                    temperature=generation_kwargs.get('temperature', 0.0),
                    top_k=generation_kwargs.get('top_k', 0),
                    top_p=generation_kwargs.get('top_p', 1.0),
                    remasking=_map_gap_strategy_to_bd3(
                        generation_kwargs.get(
                            'remasking_strategy',
                            getattr(self.model.config, 'gap_rollout_strategy', 'low_confidence_dynamic'),
                        )
                    ),
                    threshold=generation_kwargs.get(
                        'confidence_threshold',
                        getattr(self.model.config, 'gap_rollout_confidence_threshold', 0.95),
                    ),
                    stopping_criteria_sequences=stopping_criteria_sequences,
                )[:, group_tokens['input_ids'].shape[1] :]
            else:
                outputs = block_diffusion_gap_generate(
                    self.model,
                    self.tokenizer,
                    group_tokens,
                    mask_id=generation_kwargs.get('mask_id', self.model.config.mask_token_id),
                    gen_length=generation_kwargs.get('gen_length', max_out_len),
                    block_length=generation_kwargs.get(
                        'block_length', getattr(self.model.config, 'block_size', 4)
                    ),
                    denoising_steps=generation_kwargs.get(
                        'denoising_steps',
                        generation_kwargs.get('block_length', getattr(self.model.config, 'block_size', 4)),
                    ),
                    temperature=generation_kwargs.get('temperature', 0.0),
                    top_k=generation_kwargs.get('top_k', 0),
                    top_p=generation_kwargs.get('top_p', 1.0),
                    remasking_strategy=generation_kwargs.get(
                        'remasking_strategy',
                        getattr(self.model.config, 'gap_rollout_strategy', 'low_confidence_dynamic'),
                    ),
                    confidence_threshold=generation_kwargs.get(
                        'confidence_threshold',
                        getattr(self.model.config, 'gap_rollout_confidence_threshold', 0.95),
                    ),
                    remask_threshold=generation_kwargs.get(
                        'remask_threshold', getattr(self.model.config, 'gap_remask_threshold', 0.5)
                    ),
                    remask_start_ratio=generation_kwargs.get('remask_start_ratio', 0.0),
                    remask_interval_blocks=generation_kwargs.get('remask_interval_blocks', 1),
                    remask_window_blocks=generation_kwargs.get(
                        'remask_window_blocks',
                        getattr(self.model.config, 'gap_remask_window_blocks', 5),
                    ),
                    remask_start_tokens=generation_kwargs.get('remask_start_tokens', 192),
                    remask_prefix_guard_tokens=generation_kwargs.get(
                        'remask_prefix_guard_tokens',
                        generation_kwargs.get('remask_start_tokens', 192),
                    ),
                    remask_tail_guard_blocks=generation_kwargs.get('remask_tail_guard_blocks', 1),
                    stopping_criteria_sequences=stopping_criteria_sequences,
                    trace_inputs=[inputs[row_idx] for row_idx in row_indices],
                )[:, group_tokens['input_ids'].shape[1] :]

            group_decodeds = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            group_decodeds = [text.replace('<|MASK|>', '') for text in group_decodeds]
            for stop in stopping_criteria:
                group_decodeds = [text.split(stop)[0] for text in group_decodeds]
            for local_idx, row_idx in enumerate(row_indices):
                decodeds[row_idx] = group_decodeds[local_idx]

        return decodeds
