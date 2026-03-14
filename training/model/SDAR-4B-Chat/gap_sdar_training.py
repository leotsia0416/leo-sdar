from dataclasses import dataclass
import math
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class GapRemaskOutputs:
    full_candidate_mask: torch.BoolTensor
    remask_target_flat: torch.BoolTensor
    remask_pred_full: torch.BoolTensor
    z_raw: torch.LongTensor
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


def select_teacher_forced_rollout_tokens(
    masked_indices: torch.BoolTensor,
    proposal_scores_full: torch.FloatTensor,
    num_tokens,
    block_size: int,
    num_transfer_tokens: int,
    strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.95,
) -> torch.BoolTensor:
    reveal_mask = torch.zeros_like(masked_indices)
    if num_transfer_tokens <= 0:
        return reveal_mask

    for batch_idx, packed_lengths in enumerate(num_tokens):
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
                k = min(num_transfer_tokens, int(masked_local_indices.numel()))

                if strategy == "low_confidence_dynamic":
                    high_conf_mask = block_scores > confidence_threshold
                    if int(high_conf_mask.sum().item()) >= num_transfer_tokens:
                        chosen = masked_local_indices[high_conf_mask]
                    else:
                        topk = torch.topk(block_scores, k=k, sorted=False).indices
                        chosen = masked_local_indices[topk]
                elif strategy == "low_confidence_static":
                    topk = torch.topk(block_scores, k=k, sorted=False).indices
                    chosen = masked_local_indices[topk]
                elif strategy == "sequential":
                    chosen = masked_local_indices[:k]
                else:
                    raise ValueError(f"Unsupported rollout strategy: {strategy}")

                reveal_mask[batch_idx, block_start:block_end][chosen] = True
            cursor = sample_end

    return reveal_mask


def build_rollout_p_mask(
    masked_indices: torch.BoolTensor,
    labels: torch.LongTensor,
    num_tokens,
    eps: float = 1e-3,
) -> torch.FloatTensor:
    p_mask_full = torch.full(masked_indices.shape, eps, dtype=torch.float32, device=masked_indices.device)

    for batch_idx, packed_lengths in enumerate(num_tokens):
        cursor = 0
        for sample_len_tensor in packed_lengths:
            sample_len = int(sample_len_tensor.item())
            sample_end = cursor + sample_len
            sample_target_mask = labels[batch_idx, cursor:sample_end].ne(-100)
            target_count = int(sample_target_mask.sum().item())
            if target_count > 0:
                sample_masked = masked_indices[batch_idx, cursor:sample_end] & sample_target_mask
                sample_p_mask = max(sample_masked.sum().item() / target_count, eps)
                p_mask_full[batch_idx, cursor:sample_end][sample_masked] = sample_p_mask
            cursor = sample_end

    return p_mask_full[masked_indices]


def select_rollout_candidates(
    masked_indices: torch.BoolTensor,
    proposal_scores: torch.FloatTensor,
    reveal_ratio: float,
    min_reveal_tokens: int,
) -> torch.BoolTensor:
    candidate_mask = torch.zeros_like(proposal_scores, dtype=torch.bool)
    offset = 0
    for row in masked_indices:
        row_count = int(row.sum().item())
        if row_count == 0:
            continue

        row_scores = proposal_scores[offset : offset + row_count]
        desired = max(min_reveal_tokens, math.ceil(row_count * reveal_ratio))
        if row_count > 1:
            desired = min(desired, row_count - 1)
        else:
            desired = min(desired, row_count)

        if desired > 0:
            topk = torch.topk(row_scores, k=desired, sorted=False).indices
            candidate_mask[offset + topk] = True
        offset += row_count

    return candidate_mask


def apply_gap_remask(
    noisy_input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    masked_indices: torch.BoolTensor,
    p_mask: torch.FloatTensor,
    proposal_ids: torch.LongTensor,
    proposal_scores: torch.FloatTensor,
    remask_logits: torch.FloatTensor,
    mask_token_id: int,
    reveal_ratio: float,
    min_reveal_tokens: int,
    remask_threshold: float,
    remask_loss_weight: float,
    remask_default_p_mask: float,
    ignore_index: int = -100,
) -> GapRemaskOutputs:
    candidate_mask_flat = select_rollout_candidates(
        masked_indices=masked_indices,
        proposal_scores=proposal_scores,
        reveal_ratio=reveal_ratio,
        min_reveal_tokens=min_reveal_tokens,
    )
    full_candidate_mask = _scatter_flat_mask(masked_indices, candidate_mask_flat)

    labels_flat = labels[masked_indices]
    remask_target_flat = candidate_mask_flat & proposal_ids.ne(labels_flat)

    z_raw = noisy_input_ids.clone()
    if candidate_mask_flat.any():
        z_raw[full_candidate_mask] = proposal_ids[candidate_mask_flat]

    candidate_logits = remask_logits[candidate_mask_flat]
    candidate_targets = remask_target_flat[candidate_mask_flat].float()
    remask_pred_flat = torch.zeros_like(candidate_mask_flat)
    if candidate_logits.numel() > 0:
        remask_loss = F.binary_cross_entropy_with_logits(candidate_logits, candidate_targets)
        remask_pred_flat[candidate_mask_flat] = torch.sigmoid(candidate_logits) >= remask_threshold
    else:
        remask_loss = remask_logits.sum() * 0.0

    remask_pred_full = _scatter_flat_mask(masked_indices, remask_pred_flat)
    z_proj = z_raw.clone()
    z_proj[remask_pred_full] = mask_token_id

    projected_mask = z_proj.eq(mask_token_id) & labels.ne(ignore_index)
    if not projected_mask.any():
        fallback_mask = full_candidate_mask if full_candidate_mask.any() else masked_indices
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
        z_raw=z_raw,
        z_proj=z_proj,
        projected_mask=projected_mask,
        projected_p_mask=projected_p_mask,
        remask_loss=remask_loss * remask_loss_weight,
        metrics=metrics,
    )
