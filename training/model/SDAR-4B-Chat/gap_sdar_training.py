from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F


@dataclass
class GapRemaskOutputs:
    full_candidate_mask: torch.BoolTensor
    remask_target_flat: torch.BoolTensor
    remask_pred_full: torch.BoolTensor
    z_accept: torch.LongTensor
    z_proj: torch.LongTensor
    projected_mask: torch.BoolTensor
    projected_p_mask: torch.FloatTensor
    remask_loss: torch.Tensor
    metrics: Dict[str, float]


def _scatter_flat_mask(base_mask: torch.BoolTensor, selected_flat_mask: torch.BoolTensor) -> torch.BoolTensor:
    full_mask = torch.zeros_like(base_mask)
    full_mask[base_mask] = selected_flat_mask
    return full_mask


def build_p_mask_full(
    masked_indices: torch.BoolTensor,
    p_mask: torch.FloatTensor,
    shape: torch.Size,
    default_p_mask: float,
) -> torch.FloatTensor:
    p_mask_full = torch.full(shape, default_p_mask, dtype=torch.float32, device=masked_indices.device)
    p_mask_full[masked_indices] = p_mask.float()
    return p_mask_full


def get_num_transfer_tokens(block_length: int, steps: int) -> torch.LongTensor:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.full((steps,), base, dtype=torch.long)
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


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

    if strategy == "low_confidence_dynamic":
        high_conf_mask = block_scores > confidence_threshold
        if int(high_conf_mask.sum().item()) >= num_transfer_tokens:
            return masked_local_indices[high_conf_mask]

        topk = torch.topk(block_scores, k=k, sorted=False).indices
        return masked_local_indices[topk]
    if strategy == "low_confidence_static":
        topk = torch.topk(block_scores, k=k, sorted=False).indices
        return masked_local_indices[topk]
    if strategy == "sequential":
        return masked_local_indices[:k]

    raise ValueError(f"Unsupported rollout strategy: {strategy}")


def _resolve_num_transfer_tokens(
    num_transfer_tokens: Union[int, torch.Tensor, Sequence[int]],
    batch_idx: int,
) -> int:
    if torch.is_tensor(num_transfer_tokens):
        if num_transfer_tokens.numel() == 1:
            return int(num_transfer_tokens.item())
        return int(num_transfer_tokens[batch_idx].item())

    if isinstance(num_transfer_tokens, Sequence) and not isinstance(num_transfer_tokens, (str, bytes)):
        if len(num_transfer_tokens) == 1:
            return int(num_transfer_tokens[0])
        return int(num_transfer_tokens[batch_idx])

    return int(num_transfer_tokens)


def select_policy_transfer_tokens(
    masked_indices: torch.BoolTensor,
    proposal_scores_full: torch.FloatTensor,
    num_tokens,
    block_size: int,
    num_transfer_tokens: Union[int, torch.Tensor, Sequence[int]],
    strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.95,
    scope: str = "all",
) -> torch.BoolTensor:
    if scope not in {"all", "frontier_block"}:
        raise ValueError(f"Unsupported rollout scope: {scope}")

    reveal_mask = torch.zeros_like(masked_indices)
    if torch.is_tensor(num_transfer_tokens):
        if int(num_transfer_tokens.max().item()) <= 0:
            return reveal_mask
    elif isinstance(num_transfer_tokens, Sequence) and not isinstance(num_transfer_tokens, (str, bytes)):
        if max(int(x) for x in num_transfer_tokens) <= 0:
            return reveal_mask
    elif int(num_transfer_tokens) <= 0:
        return reveal_mask

    for batch_idx, packed_lengths in enumerate(num_tokens):
        current_num_transfer_tokens = _resolve_num_transfer_tokens(num_transfer_tokens, batch_idx)
        cursor = 0
        for sample_len_tensor in packed_lengths:
            sample_len = int(sample_len_tensor.item())
            sample_end = cursor + sample_len

            for block_start in range(cursor, sample_end, block_size):
                block_end = min(block_start + block_size, sample_end)
                block_mask = masked_indices[batch_idx, block_start:block_end]
                if not block_mask.any():
                    continue

                masked_local_indices = torch.nonzero(block_mask, as_tuple=False).flatten()
                block_scores = proposal_scores_full[batch_idx, block_start:block_end][masked_local_indices]
                chosen = _select_block_positions(
                    block_scores=block_scores,
                    masked_local_indices=masked_local_indices,
                    num_transfer_tokens=current_num_transfer_tokens,
                    strategy=strategy,
                    confidence_threshold=confidence_threshold,
                )
                reveal_mask[batch_idx, block_start:block_end][chosen] = True

                if scope == "frontier_block":
                    break

            cursor = sample_end

    return reveal_mask


def select_teacher_forced_rollout_tokens(
    masked_indices: torch.BoolTensor,
    proposal_scores_full: torch.FloatTensor,
    num_tokens,
    block_size: int,
    num_transfer_tokens: Union[int, torch.Tensor, Sequence[int]],
    strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.95,
    scope: str = "all",
) -> torch.BoolTensor:
    return select_policy_transfer_tokens(
        masked_indices=masked_indices,
        proposal_scores_full=proposal_scores_full,
        num_tokens=num_tokens,
        block_size=block_size,
        num_transfer_tokens=num_transfer_tokens,
        strategy=strategy,
        confidence_threshold=confidence_threshold,
        scope=scope,
    )


def build_rollout_scope_mask(
    masked_indices: torch.BoolTensor,
    reference_mask: torch.BoolTensor,
    num_tokens,
    block_size: int,
    scope: str = "all",
) -> torch.BoolTensor:
    if scope not in {"all", "frontier_block"}:
        raise ValueError(f"Unsupported rollout scope: {scope}")

    if scope == "all":
        return reference_mask.clone()

    scope_mask = torch.zeros_like(reference_mask)
    for batch_idx, packed_lengths in enumerate(num_tokens):
        cursor = 0
        for sample_len_tensor in packed_lengths:
            sample_len = int(sample_len_tensor.item())
            sample_end = cursor + sample_len

            for block_start in range(cursor, sample_end, block_size):
                block_end = min(block_start + block_size, sample_end)
                if not masked_indices[batch_idx, block_start:block_end].any():
                    continue

                scope_mask[batch_idx, block_start:block_end] = reference_mask[batch_idx, block_start:block_end]
                break

            cursor = sample_end

    return scope_mask


def build_rollout_p_mask(
    masked_indices: torch.BoolTensor,
    labels: torch.LongTensor,
    num_tokens,
    target_scope_mask: Optional[torch.BoolTensor] = None,
    per_block: bool = False,
    block_size: Optional[int] = None,
    eps: float = 1e-3,
) -> torch.FloatTensor:
    p_mask_full = torch.full(masked_indices.shape, eps, dtype=torch.float32, device=masked_indices.device)

    for batch_idx, packed_lengths in enumerate(num_tokens):
        cursor = 0
        for sample_len_tensor in packed_lengths:
            sample_len = int(sample_len_tensor.item())
            sample_end = cursor + sample_len
            if per_block:
                if block_size is None:
                    raise ValueError("block_size must be provided when per_block=True")
                for block_start in range(cursor, sample_end, block_size):
                    block_end = min(block_start + block_size, sample_end)
                    block_target_mask = labels[batch_idx, block_start:block_end].ne(-100)
                    if target_scope_mask is not None:
                        block_target_mask = block_target_mask & target_scope_mask[batch_idx, block_start:block_end]
                    target_count = int(block_target_mask.sum().item())
                    if target_count == 0:
                        continue
                    block_masked = masked_indices[batch_idx, block_start:block_end] & block_target_mask
                    block_p_mask = max(block_masked.sum().item() / target_count, eps)
                    p_mask_full[batch_idx, block_start:block_end][block_masked] = block_p_mask
            else:
                sample_target_mask = labels[batch_idx, cursor:sample_end].ne(-100)
                if target_scope_mask is not None:
                    sample_target_mask = sample_target_mask & target_scope_mask[batch_idx, cursor:sample_end]
                target_count = int(sample_target_mask.sum().item())
                if target_count > 0:
                    sample_masked = masked_indices[batch_idx, cursor:sample_end] & sample_target_mask
                    sample_p_mask = max(sample_masked.sum().item() / target_count, eps)
                    p_mask_full[batch_idx, cursor:sample_end][sample_masked] = sample_p_mask
            cursor = sample_end

    return p_mask_full[masked_indices]


def apply_gap_remask(
    noisy_input_ids: torch.LongTensor,
    clean_input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    masked_indices: torch.BoolTensor,
    p_mask: torch.FloatTensor,
    proposal_ids: torch.LongTensor,
    remask_logits: torch.FloatTensor,
    candidate_mask_full: torch.BoolTensor,
    mask_token_id: int,
    remask_threshold: float,
    remask_loss_weight: float,
    remask_default_p_mask: float,
    target_scope_mask: Optional[torch.BoolTensor] = None,
    ignore_index: int = -100,
) -> GapRemaskOutputs:
    full_candidate_mask = candidate_mask_full & masked_indices
    candidate_mask_flat = full_candidate_mask[masked_indices]

    clean_targets_flat = clean_input_ids[masked_indices]
    remask_target_flat = candidate_mask_flat & proposal_ids.ne(clean_targets_flat)

    z_accept = noisy_input_ids.clone()
    if candidate_mask_flat.any():
        z_accept[full_candidate_mask] = clean_input_ids[full_candidate_mask]

    candidate_logits = remask_logits[candidate_mask_flat]
    candidate_targets = remask_target_flat[candidate_mask_flat].float()
    remask_pred_flat = torch.zeros_like(candidate_mask_flat)
    if candidate_logits.numel() > 0:
        remask_loss = F.binary_cross_entropy_with_logits(candidate_logits, candidate_targets)
        remask_pred_flat[candidate_mask_flat] = torch.sigmoid(candidate_logits) >= remask_threshold
    else:
        remask_loss = remask_logits.sum() * 0.0

    remask_pred_full = _scatter_flat_mask(masked_indices, remask_pred_flat)
    z_proj = z_accept.clone()
    z_proj[remask_pred_full] = mask_token_id

    if target_scope_mask is None:
        target_scope_mask = labels.ne(ignore_index)
    else:
        target_scope_mask = target_scope_mask & labels.ne(ignore_index)

    projected_mask = z_proj.eq(mask_token_id) & target_scope_mask
    if not projected_mask.any():
        fallback_mask = full_candidate_mask & target_scope_mask
        if not fallback_mask.any():
            fallback_mask = masked_indices & target_scope_mask
        if not fallback_mask.any():
            fallback_mask = target_scope_mask
        fallback_indices = torch.nonzero(fallback_mask, as_tuple=False)
        if fallback_indices.numel() > 0:
            row, col = fallback_indices[0].tolist()
            z_proj[row, col] = mask_token_id
            projected_mask[row, col] = True

    p_mask_full = build_p_mask_full(
        masked_indices=masked_indices,
        p_mask=p_mask,
        shape=noisy_input_ids.shape,
        default_p_mask=remask_default_p_mask,
    )
    projected_p_mask = p_mask_full[projected_mask]

    candidate_total = max(int(candidate_mask_flat.sum().item()), 1)
    remask_positive = int(remask_target_flat.sum().item())
    remask_predicted = int(remask_pred_full.sum().item())
    metrics = {
        "candidate_tokens": float(candidate_mask_flat.sum().item()),
        "remask_positive_rate": remask_positive / candidate_total,
        "remask_pred_rate": remask_predicted / candidate_total,
        "projected_mask_tokens": float(projected_mask.sum().item()),
        "remask_loss": float((remask_loss.detach() * remask_loss_weight).item()),
    }

    return GapRemaskOutputs(
        full_candidate_mask=full_candidate_mask,
        remask_target_flat=remask_target_flat,
        remask_pred_full=remask_pred_full,
        z_accept=z_accept,
        z_proj=z_proj,
        projected_mask=projected_mask,
        projected_p_mask=projected_p_mask,
        remask_loss=remask_loss * remask_loss_weight,
        metrics=metrics,
    )
