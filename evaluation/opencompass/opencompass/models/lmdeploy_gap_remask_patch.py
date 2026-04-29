"""Runtime patches for an experimental SDAR GAP remask LMDeploy route.

The patch is intentionally opt-in. It is applied only by
``LMDeployGapRemaskwithChatTemplate`` and only activates when
``dllm_unmasking_strategy='gap_remask'`` is passed to LMDeploy.
"""

import os
from copy import deepcopy
from typing import Iterable, Tuple

import torch
from torch import nn


GAP_REMASK_STRATEGY = 'gap_remask'
_LAST_REMASK_LOGITS = None
_GAP_REMASK_HEAD_LOADED = False
_PATCHED = False
_REMASK_TRACE = dict(
    steps_with_remask=0,
    total_remasked_tokens=0,
    remasked_tokens_per_step=[],
)


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


def reset_remask_trace():
    """Reset per-generate remask stats."""
    global _REMASK_TRACE
    _REMASK_TRACE = dict(
        steps_with_remask=0,
        total_remasked_tokens=0,
        remasked_tokens_per_step=[],
    )


def get_remask_trace():
    """Return a copy of current remask stats."""
    return deepcopy(_REMASK_TRACE)


def _record_remask_stats(remask_mask: torch.Tensor):
    global _REMASK_TRACE
    remasked_tokens = int(remask_mask.sum().item())
    if remasked_tokens <= 0:
        return
    _REMASK_TRACE['steps_with_remask'] += 1
    _REMASK_TRACE['total_remasked_tokens'] += remasked_tokens
    _REMASK_TRACE['remasked_tokens_per_step'].append(remasked_tokens)


def _patch_strategy_parser():
    from lmdeploy.pytorch.config import UnmaskingStrategy

    original_from_str = UnmaskingStrategy.from_str.__func__
    if getattr(UnmaskingStrategy.from_str, '_sdar_gap_remask_patched', False):
        return

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
    original_get_logits = SDARForCausalLM.get_logits

    def __init__(self, config, ctx_mgr, dtype=None, device=None):
        original_init(self, config, ctx_mgr, dtype=dtype, device=device)
        self.gap_remask_head = nn.Linear(config.hidden_size, 1, bias=True, dtype=dtype, device=device)

    def get_logits(self, hidden_states: torch.Tensor):
        global _LAST_REMASK_LOGITS
        if hasattr(self, 'gap_remask_head'):
            _LAST_REMASK_LOGITS = self.gap_remask_head(hidden_states).squeeze(-1)
        return original_get_logits(self, hidden_states)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
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
            if name not in params_dict and not any(weight_name in name for _, weight_name, _ in stacked_params_mapping):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)

    SDARForCausalLM.__init__ = __init__
    SDARForCausalLM.get_logits = get_logits
    SDARForCausalLM.load_weights = load_weights
    SDARForCausalLM._sdar_gap_remask_patched = True


def _select_gap_remask_mask(dllm_mask: torch.Tensor) -> torch.Tensor:
    from lmdeploy.pytorch import consts

    if _LAST_REMASK_LOGITS is None:
        return torch.zeros_like(dllm_mask, dtype=torch.bool)
    if not _GAP_REMASK_HEAD_LOADED and os.getenv(
        'SDAR_LMDEPLOY_ALLOW_RANDOM_REMASK_HEAD', ''
    ).strip().lower() not in {'1', 'true', 'yes', 'on'}:
        return torch.zeros_like(dllm_mask, dtype=torch.bool)

    block_size = dllm_mask.size(-1)
    threshold = _float_env('SDAR_LMDEPLOY_GAP_REMASK_THRESHOLD', 0.5)
    max_tokens = _int_env('SDAR_LMDEPLOY_GAP_REMASK_MAX_TOKENS_PER_BLOCK', 0)
    start_block = _int_env('SDAR_LMDEPLOY_GAP_REMASK_START_BLOCK', 0)
    interval = max(1, _int_env('SDAR_LMDEPLOY_GAP_REMASK_INTERVAL_BLOCKS', 1))

    remask_logits = _LAST_REMASK_LOGITS.reshape(-1)[-dllm_mask.numel():].view_as(dllm_mask)
    remask_probs = torch.sigmoid(remask_logits)

    block_ids = torch.arange(dllm_mask.size(0), device=dllm_mask.device)
    active_blocks = block_ids.ge(start_block) & ((block_ids - start_block).remainder(interval) == 0)
    candidate_mask = dllm_mask.eq(consts.DLLM_UNMASKED) & active_blocks[:, None]
    remask_mask = candidate_mask & remask_probs.ge(threshold)

    if max_tokens > 0:
        scores = remask_probs.masked_fill(~remask_mask, float('-inf'))
        capped = torch.zeros_like(remask_mask)
        k = min(max_tokens, block_size)
        if k > 0 and remask_mask.any():
            _, indices = scores.topk(k, dim=-1)
            capped.scatter_(1, indices, True)
            remask_mask &= capped

    return remask_mask


def _get_pending_remask_state(processor, mask_view: torch.Tensor) -> torch.Tensor:
    pending = getattr(processor, '_sdar_gap_remask_pending', None)
    if pending is None or pending.device != mask_view.device or pending.numel() != mask_view.size(0):
        pending = torch.zeros(mask_view.size(0), device=mask_view.device, dtype=torch.bool)
        processor._sdar_gap_remask_pending = pending
    return pending


def _patch_unmasking_processor():
    from lmdeploy.pytorch import consts
    from lmdeploy.pytorch.config import UnmaskingStrategy
    from lmdeploy.pytorch.strategies.dllm.unmasking import UnmaskingProcessor

    if getattr(UnmaskingProcessor, '_sdar_gap_remask_patched', False):
        return

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        strategy = self.dllm_config.unmasking_strategy
        if strategy != GAP_REMASK_STRATEGY:
            if strategy is None:
                return dllm_mask
            block_size = self.dllm_config.block_length
            mask_view = dllm_mask.unflatten(0, (-1, block_size))
            is_same = (mask_view == mask_view[:, :1]).all(dim=1)
            first_mask = mask_view[:, 0]
            is_block_unmasked = is_same & (first_mask == consts.DLLM_UNMASKED)
            mask_view[is_block_unmasked] = consts.DLLM_CACHED
            dllm_mask = mask_view.flatten()
            token_ids = torch.where(dllm_mask != consts.DLLM_MASKED, input_ids, token_ids)
            if strategy == UnmaskingStrategy.LOW_CONFIDENCE_STATIC:
                dllm_mask = self.low_confidence_static(logits, token_ids, dllm_mask)
            elif strategy == UnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC:
                dllm_mask = self.low_confidence_dynamic(logits, token_ids, dllm_mask)
            elif strategy == UnmaskingStrategy.SEQUENTIAL:
                dllm_mask = self.sequential(dllm_mask)
            else:
                raise RuntimeError(f'strategy {strategy} not supported.')
            return dllm_mask, token_ids

        block_size = self.dllm_config.block_length
        mask_view = dllm_mask.unflatten(0, (-1, block_size)).clone()
        fully_unmasked = mask_view.eq(consts.DLLM_UNMASKED).all(dim=1)
        if fully_unmasked.any():
            pending_remask = _get_pending_remask_state(self, mask_view)
            remask_mask = _select_gap_remask_mask(mask_view)
            remask_mask &= ~pending_remask[:, None]
            has_remask = remask_mask.any(dim=1)
            _record_remask_stats(remask_mask)
            cache_blocks = fully_unmasked & (~has_remask | pending_remask)
            mask_view[cache_blocks] = consts.DLLM_CACHED
            mask_view[remask_mask] = consts.DLLM_MASKED
            pending_remask[has_remask] = True
            pending_remask[cache_blocks] = False

        dllm_mask = mask_view.flatten()
        token_ids = torch.where(dllm_mask != consts.DLLM_MASKED, input_ids, token_ids)

        # Reuse the standard confidence scheduler to refill remasked positions.
        if (dllm_mask == consts.DLLM_MASKED).any():
            dllm_mask = self.low_confidence_dynamic(logits, token_ids, dllm_mask)
        return dllm_mask, token_ids

    UnmaskingProcessor.__call__ = __call__
    UnmaskingProcessor._sdar_gap_remask_patched = True


def apply_gap_remask_patch():
    global _PATCHED
    if _PATCHED:
        return
    _patch_strategy_parser()
    _patch_sdar_model()
    _patch_unmasking_processor()
    _PATCHED = True


def gap_remask_head_loaded() -> bool:
    return _GAP_REMASK_HEAD_LOADED
