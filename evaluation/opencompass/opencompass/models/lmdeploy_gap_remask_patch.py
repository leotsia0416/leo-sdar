"""Runtime patches for a native frontier-block GAP remask route on installed LMDeploy."""

from collections.abc import Iterable
import os

import torch
from torch import nn


GAP_REMASK_STRATEGY = 'gap_remask'
_PATCHED = False
_GAP_REMASK_HEAD_LOADED = False
_REMASK_TRACE = dict(
    steps_with_remask=0,
    total_remasked_tokens=0,
    remasked_tokens_per_step=[],
    total_eligible_blocks=0,
    eligible_blocks_per_step=[],
    max_remask_prob=0.0,
    mean_eligible_remask_prob=0.0,
    eligible_prob_samples=0,
    gap_logits_missing_steps=0,
    pre_unmasked_blocks_total=0,
    pre_unmasked_blocks_per_step=[],
    get_logits_hook_steps=0,
    gap_logits_set_steps=0,
)
_STEP_DEBUG_COUNTER = 0


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _step_trace_path() -> str:
    return os.getenv('SDAR_LMDEPLOY_REMASK_STEP_TRACE_PATH', '').strip()


def _tensor_shape(value) -> list[int] | None:
    if value is None or not hasattr(value, 'shape'):
        return None
    return [int(x) for x in value.shape]


def _append_step_trace(record: dict):
    path = _step_trace_path()
    if not path:
        return
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=True) + '\n')


def gap_remask_head_loaded() -> bool:
    return _GAP_REMASK_HEAD_LOADED


def reset_remask_trace():
    global _REMASK_TRACE, _STEP_DEBUG_COUNTER
    _STEP_DEBUG_COUNTER = 0
    _REMASK_TRACE = dict(
        steps_with_remask=0,
        total_remasked_tokens=0,
        remasked_tokens_per_step=[],
        total_eligible_blocks=0,
        eligible_blocks_per_step=[],
        max_remask_prob=0.0,
        mean_eligible_remask_prob=0.0,
        eligible_prob_samples=0,
        gap_logits_missing_steps=0,
        pre_unmasked_blocks_total=0,
        pre_unmasked_blocks_per_step=[],
        get_logits_hook_steps=0,
        gap_logits_set_steps=0,
    )


def get_remask_trace():
    return dict(_REMASK_TRACE)


def _record_remask(remask_mask: torch.Tensor):
    remasked = int(remask_mask.sum().item())
    if remasked <= 0:
        return
    _REMASK_TRACE['steps_with_remask'] += 1
    _REMASK_TRACE['total_remasked_tokens'] += remasked
    _REMASK_TRACE['remasked_tokens_per_step'].append(remasked)


def _record_gap_probe(eligible: torch.Tensor, remask_probs: torch.Tensor):
    eligible_blocks = int(eligible.sum().item())
    _REMASK_TRACE['total_eligible_blocks'] += eligible_blocks
    _REMASK_TRACE['eligible_blocks_per_step'].append(eligible_blocks)
    if eligible_blocks <= 0:
        return
    eligible_probs = remask_probs[eligible]
    if eligible_probs.numel() <= 0:
        return
    current_max = float(eligible_probs.max().item())
    current_mean = float(eligible_probs.mean().item())
    _REMASK_TRACE['max_remask_prob'] = max(_REMASK_TRACE['max_remask_prob'], current_max)
    total_samples = _REMASK_TRACE['eligible_prob_samples']
    prev_mean = _REMASK_TRACE['mean_eligible_remask_prob']
    new_samples = int(eligible_probs.numel())
    _REMASK_TRACE['mean_eligible_remask_prob'] = (
        (prev_mean * total_samples) + (current_mean * new_samples)
    ) / (total_samples + new_samples)
    _REMASK_TRACE['eligible_prob_samples'] = total_samples + new_samples


def _record_pre_step_state(prev_block_unmasked: torch.Tensor, gap_remask_logits_present: bool):
    count = int(prev_block_unmasked.sum().item())
    _REMASK_TRACE['pre_unmasked_blocks_total'] += count
    _REMASK_TRACE['pre_unmasked_blocks_per_step'].append(count)
    if not gap_remask_logits_present:
        _REMASK_TRACE['gap_logits_missing_steps'] += 1


def _patch_strategy_parser():
    from lmdeploy.pytorch.config import UnmaskingStrategy

    if getattr(UnmaskingStrategy.from_str, '_sdar_gap_remask_patched', False):
        return

    original_from_str = UnmaskingStrategy.from_str.__func__

    @classmethod
    def from_str(cls, strategy: str):
        if str(strategy).lower() == GAP_REMASK_STRATEGY:
            return GAP_REMASK_STRATEGY
        return original_from_str(cls, strategy)

    from_str._sdar_gap_remask_patched = True
    UnmaskingStrategy.from_str = from_str


def _patch_sdar_model():
    from lmdeploy.pytorch.models.sdar import SDARForCausalLM
    from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

    if getattr(SDARForCausalLM, '_sdar_gap_remask_patched', False):
        return

    original_init = SDARForCausalLM.__init__

    def __init__(self, config, ctx_mgr, dtype=None, device=None):
        original_init(self, config, ctx_mgr, dtype=dtype, device=device)
        self.gap_remask_head = nn.Linear(config.hidden_size, 1, bias=True, dtype=dtype, device=device)

    def get_gap_remask_logits(self, hidden_states: torch.Tensor):
        return self.gap_remask_head(hidden_states).squeeze(-1).float()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        global _GAP_REMASK_HEAD_LOADED
        stacked_params_mapping = [
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if 'rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name:
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
            if name.startswith('gap_remask_head.'):
                _GAP_REMASK_HEAD_LOADED = True

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                load_weight(param, loaded_weight)

    SDARForCausalLM.__init__ = __init__
    SDARForCausalLM.get_gap_remask_logits = get_gap_remask_logits
    SDARForCausalLM.load_weights = load_weights
    SDARForCausalLM._sdar_gap_remask_patched = True


def _patch_agent_get_logits():
    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

    if getattr(BaseModelAgent.get_logits, '_sdar_gap_remask_patched', False):
        return

    original_get_logits = BaseModelAgent.get_logits

    def _resolve_gap_model(patched_model):
        model = patched_model
        if hasattr(model, 'get_gap_remask_logits'):
            return model
        if hasattr(model, 'get_model'):
            inner = model.get_model()
            if hasattr(inner, 'get_gap_remask_logits'):
                return inner
        inner = getattr(model, 'model', None)
        if inner is not None and hasattr(inner, 'get_gap_remask_logits'):
            return inner
        return None

    def get_logits(self, hidden_states):
        _REMASK_TRACE['get_logits_hook_steps'] += 1
        self.agent_strategy._sdar_gap_remask_logits = None
        gap_model = _resolve_gap_model(self.patched_model)
        if gap_model is not None:
            if isinstance(hidden_states, (tuple, list)):
                hidden_states = hidden_states[0]
            gap_logits = gap_model.get_gap_remask_logits(hidden_states)
            if gap_logits is not None:
                self.agent_strategy._sdar_gap_remask_logits = gap_logits
                _REMASK_TRACE['gap_logits_set_steps'] += 1
        return original_get_logits(self, hidden_states)

    get_logits._sdar_gap_remask_patched = True
    BaseModelAgent.get_logits = get_logits


def _patch_dllm_sequence():
    from lmdeploy.pytorch.strategies.dllm.sequence import SchedulerSequenceDLLM, DLLMSequenceStrategy

    if getattr(SchedulerSequenceDLLM, '_sdar_gap_remask_patched', False):
        return

    original_post_init = SchedulerSequenceDLLM.__post_init__
    original_inputs = SchedulerSequenceDLLM._update_token_ids_inputs
    original_decode = SchedulerSequenceDLLM._update_token_ids_decode
    original_update_token_ids = SchedulerSequenceDLLM.update_token_ids
    original_update_running = DLLMSequenceStrategy.update_running

    def __post_init__(self):
        original_post_init(self)
        self.gap_remask_budget = 1
        self.gap_remask_last_history = -1

    def _update_token_ids_inputs(self, token_ids, dllm_mask):
        original_inputs(self, token_ids, dllm_mask)
        self.gap_remask_budget = 1
        self.gap_remask_last_history = -1

    def _update_token_ids_decode(self, token_ids, dllm_mask):
        prev_history_ids = self.num_history_ids
        original_decode(self, token_ids, dllm_mask)
        if self.num_history_ids != prev_history_ids:
            self.gap_remask_budget = 1

    def update_token_ids(self, token_ids, *args, gap_remask_budget=None, gap_remask_last_history=None, **kwargs):
        original_update_token_ids(self, token_ids, *args, **kwargs)
        if gap_remask_budget is not None:
            self.gap_remask_budget = int(gap_remask_budget)
        if gap_remask_last_history is not None:
            self.gap_remask_last_history = int(gap_remask_last_history)

    def update_running(self, running, batched_outputs, model_inputs, delta, **kwargs):
        original_update_running(self, running, batched_outputs, model_inputs, delta, **kwargs)
        gap_budget = getattr(batched_outputs.extra_outputs, 'gap_remask_budget', None)
        gap_last_history = getattr(batched_outputs.extra_outputs, 'gap_remask_last_history', None)
        if gap_budget is not None:
            gap_budget = gap_budget.view(len(running)).tolist()
        if gap_last_history is not None:
            gap_last_history = gap_last_history.view(len(running)).tolist()
        for idx, msg in enumerate(running):
            if gap_budget is not None and hasattr(msg, 'gap_remask_budget'):
                msg.gap_remask_budget = int(gap_budget[idx])
            if gap_last_history is not None and hasattr(msg, 'gap_remask_last_history'):
                msg.gap_remask_last_history = int(gap_last_history[idx])

    SchedulerSequenceDLLM.__post_init__ = __post_init__
    SchedulerSequenceDLLM._update_token_ids_inputs = _update_token_ids_inputs
    SchedulerSequenceDLLM._update_token_ids_decode = _update_token_ids_decode
    SchedulerSequenceDLLM.update_token_ids = update_token_ids
    DLLMSequenceStrategy.update_running = update_running
    SchedulerSequenceDLLM._sdar_gap_remask_patched = True


def _patch_dllm_model_agent():
    from lmdeploy.pytorch import consts
    from lmdeploy.pytorch.strategies.dllm.model_agent import DLLMModelAgentStrategy

    if getattr(DLLMModelAgentStrategy, '_sdar_gap_remask_patched', False):
        return

    DLLM_MASKED = consts.DLLM_MASKED

    original_make_extra_inputs = DLLMModelAgentStrategy.make_extra_inputs
    original_update_extra_inputs = DLLMModelAgentStrategy.update_extra_inputs
    original_make_extra_outputs = DLLMModelAgentStrategy.make_extra_outputs
    original_update_prefill_for_next_step = DLLMModelAgentStrategy.update_prefill_for_next_step
    original_update_decoding_for_next_step = DLLMModelAgentStrategy.update_decoding_for_next_step
    original_post_sampling = DLLMModelAgentStrategy.post_sampling

    def _next_gap_budget(self, budget: torch.Tensor | None, next_extra_inputs):
        if budget is None:
            return None
        dllm_mask = next_extra_inputs.dllm_mask.view(-1, self.block_size)
        advanced = (dllm_mask == DLLM_MASKED).all(dim=1)
        return torch.where(advanced, torch.ones_like(budget), budget)

    def make_extra_inputs(self, seqs, model_inputs):
        extra_inputs = original_make_extra_inputs(self, seqs, model_inputs)
        budgets = [getattr(seq, 'gap_remask_budget', 1) for seq in seqs]
        last_histories = [getattr(seq, 'gap_remask_last_history', -1) for seq in seqs]
        extra_inputs.gap_remask_budget = torch.tensor(budgets, dtype=torch.int64, device=extra_inputs.dllm_mask.device)
        extra_inputs.gap_remask_last_history = torch.tensor(
            last_histories, dtype=torch.int64, device=extra_inputs.dllm_mask.device
        )
        return extra_inputs

    def update_extra_inputs(self, extra_inputs, delta):
        next_extra_inputs = original_update_extra_inputs(self, extra_inputs, delta)
        budget = getattr(extra_inputs, 'gap_remask_budget', None)
        if budget is not None:
            next_extra_inputs.gap_remask_budget = budget[delta.indices]
        last_history = getattr(extra_inputs, 'gap_remask_last_history', None)
        if last_history is not None:
            next_extra_inputs.gap_remask_last_history = last_history[delta.indices]
        return next_extra_inputs

    def make_extra_outputs(self, extra_inputs):
        extra_outputs = original_make_extra_outputs(self, extra_inputs)
        extra_outputs.gap_remask_budget = getattr(extra_inputs, 'gap_remask_budget', None)
        extra_outputs.gap_remask_last_history = getattr(extra_inputs, 'gap_remask_last_history', None)
        return extra_outputs

    def update_prefill_for_next_step(self, model_inputs, extra_inputs, next_token_ids, model_metas, extra_outputs):
        next_inputs, next_extra_inputs = original_update_prefill_for_next_step(
            self, model_inputs, extra_inputs, next_token_ids, model_metas, extra_outputs
        )
        budget = getattr(extra_outputs, 'gap_remask_budget', getattr(extra_inputs, 'gap_remask_budget', None))
        last_history = getattr(
            extra_outputs,
            'gap_remask_last_history',
            getattr(extra_inputs, 'gap_remask_last_history', None),
        )
        next_extra_inputs.gap_remask_budget = _next_gap_budget(self, budget, next_extra_inputs)
        next_extra_inputs.gap_remask_last_history = last_history
        return next_inputs, next_extra_inputs

    def update_decoding_for_next_step(self, model_inputs, next_token_ids, model_metas, extra_inputs, **kwargs):
        next_inputs, next_extra_inputs = original_update_decoding_for_next_step(
            self, model_inputs, next_token_ids, model_metas, extra_inputs, **kwargs
        )
        budget = getattr(extra_inputs, 'gap_remask_budget', None)
        last_history = getattr(extra_inputs, 'gap_remask_last_history', None)
        next_extra_inputs.gap_remask_budget = _next_gap_budget(self, budget, next_extra_inputs)
        next_extra_inputs.gap_remask_last_history = last_history
        return next_inputs, next_extra_inputs

    def post_sampling(self, inputs, logits, next_token_ids, extra_inputs):
        gap_logits = getattr(self, '_sdar_gap_remask_logits', None)
        if getattr(self.unmasking_processor.dllm_config, 'unmasking_strategy', None) != GAP_REMASK_STRATEGY:
            return original_post_sampling(self, inputs, logits, next_token_ids, extra_inputs)
        dllm_mask = extra_inputs.dllm_mask
        input_ids = inputs.input_ids
        input_ids = self.slice_outputs(input_ids.flatten(), inputs.seq_length)
        dllm_mask, next_token_ids, gap_budget, gap_last_history = self.unmasking_processor(
            logits,
            input_ids,
            next_token_ids,
            dllm_mask,
            gap_remask_logits=gap_logits,
            history_lengths=inputs.history_lengths,
            gap_remask_budget=getattr(extra_inputs, 'gap_remask_budget', None),
            gap_remask_last_history=getattr(extra_inputs, 'gap_remask_last_history', None),
        )
        extra_inputs.dllm_mask = dllm_mask
        extra_inputs.gap_remask_budget = gap_budget
        extra_inputs.gap_remask_last_history = gap_last_history
        return next_token_ids, extra_inputs

    DLLMModelAgentStrategy.make_extra_inputs = make_extra_inputs
    DLLMModelAgentStrategy.update_extra_inputs = update_extra_inputs
    DLLMModelAgentStrategy.make_extra_outputs = make_extra_outputs
    DLLMModelAgentStrategy.update_prefill_for_next_step = update_prefill_for_next_step
    DLLMModelAgentStrategy.update_decoding_for_next_step = update_decoding_for_next_step
    DLLMModelAgentStrategy.post_sampling = post_sampling
    DLLMModelAgentStrategy._sdar_gap_remask_patched = True


def _patch_unmasking():
    from lmdeploy.pytorch import consts
    from lmdeploy.pytorch.strategies.dllm.unmasking import UnmaskingProcessor

    if getattr(UnmaskingProcessor.__call__, '_sdar_gap_remask_patched', False):
        return

    DLLM_MASKED = consts.DLLM_MASKED
    DLLM_UNMASKED = consts.DLLM_UNMASKED
    DLLM_CACHED = consts.DLLM_CACHED

    original_call = UnmaskingProcessor.__call__

    def _gap_remask(self,
                    dllm_mask: torch.Tensor,
                    gap_remask_logits: torch.Tensor | None,
                    history_lengths: torch.Tensor | None,
                    gap_remask_budget: torch.Tensor | None,
                    gap_remask_last_history: torch.Tensor | None,
                    finishable_blocks: torch.Tensor):
        if gap_remask_logits is None or history_lengths is None:
            return dllm_mask, gap_remask_budget, gap_remask_last_history, torch.zeros_like(dllm_mask, dtype=torch.bool)

        block_size = self.dllm_config.block_length
        threshold = _float_env('SDAR_LMDEPLOY_GAP_REMASK_THRESHOLD', 0.5)
        start_block = _int_env('SDAR_LMDEPLOY_GAP_REMASK_START_BLOCK', 0)
        interval_blocks = max(1, _int_env('SDAR_LMDEPLOY_GAP_REMASK_INTERVAL_BLOCKS', 1))
        max_tokens_per_block = _int_env('SDAR_LMDEPLOY_GAP_REMASK_MAX_TOKENS_PER_BLOCK', 0)

        dllm_mask = dllm_mask.view(-1, block_size)
        remask_probs = torch.sigmoid(gap_remask_logits.float()).view(-1, block_size)
        if gap_remask_budget is None:
            gap_remask_budget = torch.ones(dllm_mask.size(0), dtype=torch.int64, device=dllm_mask.device)
        else:
            gap_remask_budget = gap_remask_budget.to(device=dllm_mask.device)
        if gap_remask_last_history is None:
            gap_remask_last_history = torch.full(
                (dllm_mask.size(0),), -1, dtype=torch.int64, device=dllm_mask.device
            )
        else:
            gap_remask_last_history = gap_remask_last_history.to(device=dllm_mask.device)

        if history_lengths.numel() != dllm_mask.size(0):
            return dllm_mask, gap_remask_budget, gap_remask_last_history, torch.zeros_like(dllm_mask, dtype=torch.bool)

        processor_last_history = getattr(self, '_sdar_gap_remask_last_history', None)
        if processor_last_history is not None and processor_last_history.numel() == dllm_mask.size(0):
            processor_last_history = processor_last_history.to(device=dllm_mask.device)
            processor_last_history = torch.where(
                history_lengths < processor_last_history,
                torch.full_like(processor_last_history, -1),
                processor_last_history,
            )
            gap_remask_last_history = torch.maximum(gap_remask_last_history, processor_last_history)

        block_ids = torch.div(history_lengths, block_size, rounding_mode='floor')
        eligible = finishable_blocks & (block_ids >= start_block) & (gap_remask_budget > 0)
        eligible &= history_lengths != gap_remask_last_history
        eligible &= ((block_ids - start_block) % interval_blocks == 0)
        _record_gap_probe(eligible, remask_probs)
        if not eligible.any():
            return dllm_mask, gap_remask_budget, gap_remask_last_history, torch.zeros_like(dllm_mask, dtype=torch.bool)

        candidate_mask = remask_probs >= threshold
        candidate_mask &= eligible[:, None]
        if max_tokens_per_block > 0:
            topk = min(max_tokens_per_block, block_size)
            top_vals, top_idx = remask_probs.topk(topk, dim=-1)
            top_mask = torch.zeros_like(candidate_mask)
            top_mask.scatter_(1, top_idx, top_vals >= threshold)
            candidate_mask &= top_mask

        remask_mask = candidate_mask.clone()
        remasked_blocks = candidate_mask.any(dim=1)
        gap_remask_budget = torch.where(remasked_blocks, torch.zeros_like(gap_remask_budget), gap_remask_budget)
        gap_remask_last_history = torch.where(remasked_blocks, history_lengths, gap_remask_last_history)
        self._sdar_gap_remask_last_history = gap_remask_last_history.detach()
        _record_remask(remask_mask)
        return dllm_mask, gap_remask_budget, gap_remask_last_history, remask_mask

    def __call__(self,
                 logits: torch.Tensor,
                 input_ids: torch.Tensor,
                 token_ids: torch.Tensor,
                 dllm_mask: torch.Tensor,
                 gap_remask_logits: torch.Tensor | None = None,
                 history_lengths: torch.Tensor | None = None,
                 gap_remask_budget: torch.Tensor | None = None,
                 gap_remask_last_history: torch.Tensor | None = None):
        global _STEP_DEBUG_COUNTER
        strategy = self.dllm_config.unmasking_strategy
        if strategy != GAP_REMASK_STRATEGY:
            return (
                *original_call(self, logits, input_ids, token_ids, dllm_mask),
                gap_remask_budget,
                gap_remask_last_history,
            )

        block_size = self.dllm_config.block_length
        dllm_mask = dllm_mask.unflatten(0, (-1, block_size))
        pre_step_mask = dllm_mask.clone()
        step_idx = _STEP_DEBUG_COUNTER
        _STEP_DEBUG_COUNTER += 1
        is_same = (pre_step_mask == pre_step_mask[:, :1]).all(dim=1)
        first_mask = pre_step_mask[:, 0]
        prev_block_unmasked = is_same & (first_mask == DLLM_UNMASKED)
        pre_masked_tokens = int((pre_step_mask == DLLM_MASKED).sum().item())
        pre_unmasked_blocks = int(prev_block_unmasked.sum().item())

        token_ids = torch.where(pre_step_mask.flatten() != DLLM_MASKED, input_ids, token_ids)

        base_next_mask = self.sequential(pre_step_mask.flatten()).unflatten(0, (-1, block_size))
        finishable_blocks = (~prev_block_unmasked) & (base_next_mask == DLLM_UNMASKED).all(dim=1)
        _record_pre_step_state(finishable_blocks, gap_remask_logits is not None)

        _, gap_remask_budget, gap_remask_last_history, remask_mask = _gap_remask(
            self,
            pre_step_mask,
            gap_remask_logits,
            history_lengths,
            gap_remask_budget,
            gap_remask_last_history,
            finishable_blocks,
        )
        remasked_blocks = remask_mask.any(dim=1)

        dllm_mask = base_next_mask.clone()
        cache_blocks = prev_block_unmasked
        dllm_mask[cache_blocks] = DLLM_CACHED

        if remasked_blocks.any():
            dllm_mask = dllm_mask.clone()
            dllm_mask[remask_mask] = DLLM_MASKED

        post_masked_tokens = int((dllm_mask == DLLM_MASKED).sum().item())
        cached_blocks = int((dllm_mask == DLLM_CACHED).all(dim=1).sum().item())
        remasked_tokens = int(remask_mask.sum().item())
        if _step_trace_path() and (
            step_idx < 256
            or remasked_tokens > 0
            or step_idx % 128 == 0
        ):
            max_prob = None
            mean_prob = None
            if gap_remask_logits is not None:
                probs = torch.sigmoid(gap_remask_logits.float())
                if probs.numel() > 0:
                    max_prob = float(probs.max().item())
                    mean_prob = float(probs.mean().item())
            _append_step_trace(
                dict(
                    step=step_idx,
                    num_blocks=int(pre_step_mask.size(0)),
                    block_size=int(block_size),
                    pre_masked_tokens=pre_masked_tokens,
                    post_masked_tokens=post_masked_tokens,
                    pre_unmasked_blocks=pre_unmasked_blocks,
                    finishable_blocks=int(finishable_blocks.sum().item()),
                    cached_blocks=cached_blocks,
                    remasked_blocks=int(remasked_blocks.sum().item()),
                    remasked_tokens=remasked_tokens,
                    history_lengths_shape=_tensor_shape(history_lengths),
                    history_lengths=(
                        [int(x) for x in history_lengths.detach().cpu().flatten().tolist()[:8]]
                        if history_lengths is not None else None
                    ),
                    budget_shape=_tensor_shape(gap_remask_budget),
                    budget=(
                        [int(x) for x in gap_remask_budget.detach().cpu().flatten().tolist()[:8]]
                        if gap_remask_budget is not None else None
                    ),
                    last_history_shape=_tensor_shape(gap_remask_last_history),
                    last_history=(
                        [int(x) for x in gap_remask_last_history.detach().cpu().flatten().tolist()[:8]]
                        if gap_remask_last_history is not None else None
                    ),
                    gap_logits_shape=_tensor_shape(gap_remask_logits),
                    gap_logits_present=gap_remask_logits is not None,
                    max_remask_prob=max_prob,
                    mean_remask_prob=mean_prob,
                )
            )

        return dllm_mask.flatten(), token_ids, gap_remask_budget, gap_remask_last_history

    __call__._sdar_gap_remask_patched = True
    UnmaskingProcessor.__call__ = __call__


def apply_gap_remask_patch():
    global _PATCHED
    if _PATCHED:
        return
    _patch_strategy_parser()
    _patch_sdar_model()
    _patch_agent_get_logits()
    _patch_dllm_sequence()
    _patch_dllm_model_agent()
    _patch_unmasking()
    _PATCHED = True
