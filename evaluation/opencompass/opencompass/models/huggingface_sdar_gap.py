import json
import os
from typing import List, Optional

import torch

from opencompass.registry import MODELS

from .huggingface_bd3 import (
    BD3withChatTemplate,
    _prepare_block_diffusion_batch,
    _convert_chat_messages,
    _format_with_fast_chat_template,
    _get_stopping_criteria,
    get_num_transfer_tokens,
    sample_with_temperature_topk_topp,
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
    remask_threshold=0.5,
    stopping_criteria_idx=None,
    trace_inputs=None,
):
    from transformers.cache_utils import DynamicCache

    input_ids, prompt_length, total_length, block_diffusion_attention_mask, position_ids, x = (
        _prepare_block_diffusion_batch(prompt, mask_id, gen_length, block_length, model.device)
    )
    batch_size = input_ids.shape[0]
    past_key_values = DynamicCache()
    num_blocks = total_length // block_length
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:, :, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
        )

    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)
    gap_enabled = bool(getattr(model.config, 'gap_enable', False)) and hasattr(model, 'gap_remask_head')
    finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
    remask_steps_per_batch = [0 for _ in range(batch_size)]
    remask_tokens_per_batch = [0 for _ in range(batch_size)]

    for num_block in range(prefill_blocks, num_blocks):
        block_slice = slice(num_block * block_length, (num_block + 1) * block_length)
        cur_x = x[:, block_slice].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, :, block_slice, : (num_block + 1) * block_length
        ]
        cur_position_ids = position_ids[:, block_slice]

        for step in range(denoising_steps + 1):
            mask_index = cur_x.eq(mask_id)
            if mask_index.sum() == 0:
                model(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True,
                )
                break

            outputs = model(
                cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
                output_hidden_states=gap_enabled,
            )
            logits = outputs.logits
            logits[..., mask_id] = float('-inf')

            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            x0 = torch.where(mask_index, x0, cur_x)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for row_idx in range(batch_size):
                row_mask = mask_index[row_idx]
                masked_count = int(row_mask.sum().item())
                if masked_count == 0:
                    continue

                row_scores = torch.where(row_mask, x0_p[row_idx], torch.full_like(x0_p[row_idx], float('-inf')))
                k = min(int(num_transfer_tokens[step].item()), masked_count)

                if remasking_strategy == 'sequential':
                    masked_positions = row_mask.nonzero(as_tuple=True)[0]
                    transfer_index[row_idx, masked_positions[:k]] = True
                elif remasking_strategy == 'low_confidence_static':
                    _, idx = torch.topk(row_scores, k=k)
                    transfer_index[row_idx, idx] = True
                elif remasking_strategy == 'low_confidence_dynamic':
                    high_conf_mask = row_scores > confidence_threshold
                    if int(high_conf_mask.sum().item()) >= k:
                        transfer_index[row_idx] = high_conf_mask
                    else:
                        _, idx = torch.topk(row_scores, k=k)
                        transfer_index[row_idx, idx] = True
                else:
                    raise ValueError(f'Unknown remasking strategy: {remasking_strategy}')

            if gap_enabled:
                hidden_states = outputs.hidden_states[-1]
                remask_probs = torch.sigmoid(model.gap_remask_head(hidden_states).squeeze(-1))
                remask_index = transfer_index & remask_probs.ge(remask_threshold)
                remask_counts = remask_index.sum(dim=1).tolist()
                for row_idx, remask_count in enumerate(remask_counts):
                    if remask_count > 0:
                        remask_steps_per_batch[row_idx] += 1
                        remask_tokens_per_batch[row_idx] += int(remask_count)
                transfer_index = transfer_index & ~remask_index

                for row_idx in range(batch_size):
                    if transfer_index[row_idx].any() or not mask_index[row_idx].any():
                        continue

                    candidates = (mask_index[row_idx] & ~remask_index[row_idx]).nonzero(as_tuple=True)[0]
                    if candidates.numel() > 0:
                        candidate_scores = x0_p[row_idx, candidates]
                        best_idx = candidates[torch.argmax(candidate_scores)]
                    else:
                        candidates = mask_index[row_idx].nonzero(as_tuple=True)[0]
                        remask_scores = remask_probs[row_idx, candidates]
                        best_idx = candidates[torch.argmin(remask_scores)]
                    transfer_index[row_idx, best_idx] = True

            cur_x[transfer_index] = x0[transfer_index]

        x[:, block_slice] = cur_x

        if stopping_criteria_idx is not None:
            generated_prefix = x[:, prompt_length : (num_block + 1) * block_length]
            for stop_idx in stopping_criteria_idx:
                finished |= generated_prefix.eq(stop_idx).any(dim=1)
            if bool(finished.all().item()):
                break

    trace_path = os.getenv('SDAR_REMASK_TRACE_PATH')
    if trace_path:
        with open(trace_path, 'a', encoding='utf-8') as f:
            for row_idx in range(batch_size):
                record = {
                    'input': trace_inputs[row_idx] if trace_inputs is not None else None,
                    'steps_with_remask': remask_steps_per_batch[row_idx],
                    'total_remasked_tokens': remask_tokens_per_batch[row_idx],
                    'gap_enabled': gap_enabled,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    return x


@MODELS.register_module()
class SDARGapwithChatTemplate(BD3withChatTemplate):
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

        stopping_criteria = list(set(stopping_criteria + self.stop_words))
        if stopping_criteria:
            generation_kwargs['stopping_criteria'] = _get_stopping_criteria(
                stopping_criteria, self.tokenizer, batch_size
            )
        if max_out_len is not None:
            generation_kwargs['max_new_tokens'] = max_out_len
        if min_out_len is not None:
            generation_kwargs['min_new_tokens'] = min_out_len
        generation_kwargs['pad_token_id'] = self.tokenizer.pad_token_id

        stopping_criteria_idx = [
            self.tokenizer.encode(stop, add_special_tokens=False)[0]
            for stop in stopping_criteria
        ]

        outputs = block_diffusion_gap_generate(
            self.model,
            self.tokenizer,
            tokens,
            mask_id=generation_kwargs.get('mask_id', self.model.config.mask_token_id),
            gen_length=generation_kwargs.get('gen_length', max_out_len),
            block_length=generation_kwargs.get('block_length', getattr(self.model.config, 'block_size', 4)),
            denoising_steps=generation_kwargs.get(
                'denoising_steps', generation_kwargs.get('block_length', getattr(self.model.config, 'block_size', 4))
            ),
            temperature=generation_kwargs.get('temperature', 1.0),
            top_k=generation_kwargs.get('top_k', 0),
            top_p=generation_kwargs.get('top_p', 1.0),
            remasking_strategy=generation_kwargs.get('remasking_strategy', 'low_confidence_dynamic'),
            confidence_threshold=generation_kwargs.get('confidence_threshold', 1.0),
            remask_threshold=generation_kwargs.get(
                'remask_threshold', getattr(self.model.config, 'gap_remask_threshold', 0.5)
            ),
            stopping_criteria_idx=stopping_criteria_idx,
            trace_inputs=inputs,
        )[:, tokens['input_ids'].shape[1] :]

        decodeds = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        decodeds = [t.replace('<|MASK|>', '') for t in decodeds]
        for stop in stopping_criteria:
            decodeds = [t.split(stop)[0] for t in decodeds]
        return decodeds
