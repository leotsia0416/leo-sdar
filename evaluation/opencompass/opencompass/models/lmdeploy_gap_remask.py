import json
import os
import time

from .lmdeploy_gap_remask_patch import (
    GAP_REMASK_STRATEGY,
    apply_gap_remask_patch,
    gap_remask_head_loaded,
    get_remask_trace,
    reset_remask_trace,
)
from .lmdeploy_with_tf_above_v4_33 import LMDeploywithChatTemplate


class LMDeployGapRemaskwithChatTemplate(LMDeploywithChatTemplate):
    """LMDeploy SDAR wrapper with installed-LMDeploy GAP remask patching."""

    def _append_remask_trace(self, inputs, outputs, max_out_len):
        trace_path = os.getenv('SDAR_LMDEPLOY_REMASK_TRACE_PATH', '').strip()
        if not trace_path:
            return
        trace = get_remask_trace()
        record = dict(
            batch_size=len(inputs),
            max_out_len=max_out_len,
            gap_remask_threshold=os.getenv('SDAR_LMDEPLOY_GAP_REMASK_THRESHOLD', ''),
            gap_remask_start_block=os.getenv('SDAR_LMDEPLOY_GAP_REMASK_START_BLOCK', ''),
            gap_remask_interval_blocks=os.getenv('SDAR_LMDEPLOY_GAP_REMASK_INTERVAL_BLOCKS', ''),
            gap_remask_max_tokens_per_block=os.getenv('SDAR_LMDEPLOY_GAP_REMASK_MAX_TOKENS_PER_BLOCK', ''),
            gap_remask_head_loaded=gap_remask_head_loaded(),
            steps_with_remask=trace['steps_with_remask'],
            total_remasked_tokens=trace['total_remasked_tokens'],
            remasked_tokens_per_step=trace['remasked_tokens_per_step'],
            total_eligible_blocks=trace['total_eligible_blocks'],
            eligible_blocks_per_step=trace['eligible_blocks_per_step'],
            max_remask_prob=trace['max_remask_prob'],
            mean_eligible_remask_prob=trace['mean_eligible_remask_prob'],
            gap_logits_missing_steps=trace['gap_logits_missing_steps'],
            pre_unmasked_blocks_total=trace['pre_unmasked_blocks_total'],
            pre_unmasked_blocks_per_step=trace['pre_unmasked_blocks_per_step'],
            get_logits_hook_steps=trace['get_logits_hook_steps'],
            gap_logits_set_steps=trace['gap_logits_set_steps'],
            prompt_chars=[len(str(item)) for item in inputs],
            output_chars=[len(str(item)) for item in outputs],
        )
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        with open(trace_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=True) + '\n')

    def _load_model(self, path: str, added_model_kwargs: dict = dict()):
        added_model_kwargs = added_model_kwargs.copy()
        os.environ['SDAR_LMDEPLOY_GAP_REMASK_THRESHOLD'] = str(
            added_model_kwargs.pop('gap_remask_threshold', 0.5)
        )
        os.environ['SDAR_LMDEPLOY_GAP_REMASK_INTERVAL_BLOCKS'] = str(
            added_model_kwargs.pop('gap_remask_interval_blocks', 1)
        )
        os.environ['SDAR_LMDEPLOY_GAP_REMASK_START_BLOCK'] = str(
            added_model_kwargs.pop('gap_remask_start_block', 0)
        )
        os.environ['SDAR_LMDEPLOY_GAP_REMASK_MAX_TOKENS_PER_BLOCK'] = str(
            added_model_kwargs.pop('gap_remask_max_tokens_per_block', 0)
        )
        added_model_kwargs['dllm_unmasking_strategy'] = GAP_REMASK_STRATEGY
        apply_gap_remask_patch()
        super()._load_model(path, added_model_kwargs)
        if not gap_remask_head_loaded():
            self.logger.warning(
                'GAP remask head weights were not loaded; remask decisions will use random initialization.'
            )

    def generate(self, inputs, max_out_len, stopping_criteria=[], **kwargs):
        reset_remask_trace()
        if os.getenv('SDAR_LMDEPLOY_GAP_REMASK_DEBUG', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
            prompt_lens = [len(str(item)) for item in inputs]
            self.logger.info(
                'GAP remask generate begin: batch=%d max_out_len=%s prompt_chars_min=%s prompt_chars_max=%s',
                len(inputs),
                max_out_len,
                min(prompt_lens) if prompt_lens else 0,
                max(prompt_lens) if prompt_lens else 0,
            )
            start = time.time()
            outputs = super().generate(inputs, max_out_len, stopping_criteria=stopping_criteria, **kwargs)
            self._append_remask_trace(inputs, outputs, max_out_len)
            trace = get_remask_trace()
            self.logger.info(
                'GAP remask generate end: batch=%d elapsed=%.2fs output_chars_min=%s output_chars_max=%s '
                'remask_steps=%d remask_tokens=%d',
                len(inputs),
                time.time() - start,
                min(len(str(item)) for item in outputs) if outputs else 0,
                max(len(str(item)) for item in outputs) if outputs else 0,
                trace['steps_with_remask'],
                trace['total_remasked_tokens'],
            )
            return outputs
        outputs = super().generate(inputs, max_out_len, stopping_criteria=stopping_criteria, **kwargs)
        self._append_remask_trace(inputs, outputs, max_out_len)
        return outputs
