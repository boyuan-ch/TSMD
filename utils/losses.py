import torch
import torch.nn.functional as F


def peak_biased_pairs(gt, num_pairs, top_frac=0.15, generator=None):
    """Draw i from top-top_frac GT frames, j uniformly from all frames."""
    N = gt.size(0)
    k = max(1, int(N * top_frac))
    top_idx = gt.topk(k).indices
    i = top_idx[torch.randint(
        0, k, (num_pairs,), device=gt.device, generator=generator
    )]
    j = torch.randint(
        0, N, (num_pairs,), device=gt.device, generator=generator
    )
    return i, j


def boundary_biased_pairs(gt, num_pairs, top_frac=0.15, generator=None):
    """Draw anchors from GT top-k and negatives from the adjacent k ranks."""
    num_frames = gt.size(0)
    top_count = max(1, int(num_frames * top_frac))
    boundary_stop = min(num_frames, max(top_count + 1, 2 * top_count))
    if top_count >= num_frames or boundary_stop <= top_count:
        empty = torch.empty(0, dtype=torch.long, device=gt.device)
        return empty, empty
    order = torch.argsort(gt, descending=True, stable=True)
    anchors = order[:top_count]
    negatives = order[top_count:boundary_stop]
    i = anchors[torch.randint(
        0, len(anchors), (num_pairs,), device=gt.device, generator=generator
    )]
    j = negatives[torch.randint(
        0, len(negatives), (num_pairs,), device=gt.device,
        generator=generator,
    )]
    return i, j


def boundary_ranknet_loss(
    pred_logits,
    gt,
    num_pairs=512,
    min_gap=0.1,
    top_frac=0.15,
    generator=None,
    error_conditioned=False,
    error_floor=0.25,
    error_temperature=1.0,
    return_details=False,
):
    """RankNet on GT top-k versus adjacent-k boundary pairs."""
    i, j = boundary_biased_pairs(
        gt, num_pairs, top_frac, generator=generator
    )
    sampled_count = i.numel()
    if sampled_count == 0:
        loss = pred_logits.sum() * 0.0
        if not return_details:
            return loss
        return loss, {
            'i': i,
            'j': j,
            'target': pred_logits[:0],
            'pred_diff': pred_logits[:0],
            'pair_weight': pred_logits[:0],
            'sampled_count': 0,
            'accepted_count': 0,
        }

    gt_diff = gt[i] - gt[j]
    keep = gt_diff.abs() >= min_gap
    i, j = i[keep], j[keep]
    if i.numel() == 0:
        loss = pred_logits.sum() * 0.0
        if not return_details:
            return loss
        return loss, {
            'i': i,
            'j': j,
            'target': pred_logits[:0],
            'pred_diff': pred_logits[:0],
            'pair_weight': pred_logits[:0],
            'sampled_count': sampled_count,
            'accepted_count': 0,
        }

    target = (gt[i] > gt[j]).to(pred_logits.dtype)
    pred_diff = pred_logits[i] - pred_logits[j]
    pair_loss = F.binary_cross_entropy_with_logits(
        pred_diff, target, reduction='none'
    )
    pair_weight = torch.ones_like(pair_loss)
    if error_conditioned:
        if not 0.0 < error_floor <= 1.0:
            raise ValueError('error_floor must be in (0, 1]')
        if error_temperature <= 0.0:
            raise ValueError('error_temperature must be positive')
        signed_margin = pred_diff.detach() * (2.0 * target - 1.0)
        pair_weight = error_floor + (1.0 - error_floor) * torch.sigmoid(
            -signed_margin / error_temperature
        )
        pair_weight = pair_weight / pair_weight.mean().clamp_min(1e-8)
    loss = (pair_weight * pair_loss).mean()
    if not return_details:
        return loss
    return loss, {
        'i': i,
        'j': j,
        'target': target,
        'pred_diff': pred_diff,
        'pair_weight': pair_weight,
        'sampled_count': sampled_count,
        'accepted_count': i.numel(),
    }


def batched_boundary_ranknet_loss(
    pred_logits,
    gt_score,
    mask,
    num_pairs=512,
    min_gap=0.1,
    top_frac=0.15,
    generator=None,
    error_conditioned=False,
    error_floor=0.25,
    error_temperature=1.0,
    return_details=False,
    weights=None,
):
    """Compute boundary RankNet over a padded batch."""
    losses = []
    batch_details = []
    for batch_index in range(pred_logits.size(0)):
        valid = mask[batch_index]
        result = boundary_ranknet_loss(
            pred_logits[batch_index][valid],
            gt_score[batch_index][valid],
            num_pairs=num_pairs,
            min_gap=min_gap,
            top_frac=top_frac,
            generator=generator,
            error_conditioned=error_conditioned,
            error_floor=error_floor,
            error_temperature=error_temperature,
            return_details=return_details,
        )
        if return_details:
            loss, details = result
            losses.append(loss)
            batch_details.append(details)
        else:
            losses.append(result)
    losses = torch.stack(losses)
    loss = (
        (losses * weights).sum() / weights.sum()
        if weights is not None else losses.mean()
    )
    if return_details:
        return loss, batch_details
    return loss


def ranknet_loss(
    pred_logits,
    gt,
    num_pairs=512,
    min_gap=0.1,
    top_frac=0.15,
    generator=None,
    return_details=False,
):
    """
    Peak-biased RankNet pairwise loss for a single video.

    pred_logits: (N,) pre-sigmoid logits
    gt:          (N,) ground-truth scores in [0, 1]
    """
    i, j = peak_biased_pairs(gt, num_pairs, top_frac, generator=generator)

    gt_diff = gt[i] - gt[j]
    mask = gt_diff.abs() >= min_gap
    if mask.sum() == 0:
        loss = pred_logits.sum() * 0.0  # zero loss; keeps computation graph
        if not return_details:
            return loss
        empty_index = i[:0]
        return loss, {
            'i': empty_index,
            'j': empty_index,
            'target': pred_logits[:0],
            'pred_diff': pred_logits[:0],
            'sampled_count': num_pairs,
            'accepted_count': 0,
        }

    i, j = i[mask], j[mask]
    target = (gt[i] > gt[j]).float()
    pred_diff = pred_logits[i] - pred_logits[j]
    loss = F.binary_cross_entropy_with_logits(pred_diff, target)
    if not return_details:
        return loss
    return loss, {
        'i': i,
        'j': j,
        'target': target,
        'pred_diff': pred_diff,
        'sampled_count': num_pairs,
        'accepted_count': i.numel(),
    }


def batched_ranknet_loss(
    pred_logits,
    gt_score,
    mask,
    num_pairs=512,
    min_gap=0.1,
    top_frac=0.15,
    generator=None,
    return_details=False,
    weights=None,
):
    """
    Compute RankNet loss over a padded batch.

    pred_logits: (B, T_max) pre-sigmoid logits
    gt_score:    (B, T_max) ground-truth scores, zero-padded
    mask:        (B, T_max) bool, True for valid frames
    """
    losses = []
    batch_details = []
    for b in range(pred_logits.size(0)):
        valid = mask[b]          # (T_max,) bool
        p = pred_logits[b][valid]
        g = gt_score[b][valid]
        result = ranknet_loss(
            p,
            g,
            num_pairs,
            min_gap,
            top_frac,
            generator=generator,
            return_details=return_details,
        )
        if return_details:
            loss, details = result
            losses.append(loss)
            batch_details.append(details)
        else:
            losses.append(result)
    losses = torch.stack(losses)
    loss = (losses * weights).sum() / weights.sum() if weights is not None else losses.mean()
    if return_details:
        return loss, batch_details
    return loss


def pearson_loss(pred, gt, eps=1e-8):
    """
    1 - Pearson r for a single (unpadded) sequence.
    pred, gt: (N,) float tensors
    Returns a scalar in [0, 2] (0 when perfectly correlated).
    """
    pred_c = pred - pred.mean()
    gt_c   = gt   - gt.mean()
    num    = (pred_c * gt_c).sum()
    denom  = pred_c.norm() * gt_c.norm() + eps
    return 1.0 - num / denom


def batched_pearson_loss(pred, gt_score, mask, eps=1e-8, weights=None):
    """
    Compute Pearson loss over a padded batch.

    pred:     (B, T_max) sigmoid scores
    gt_score: (B, T_max) ground-truth scores, zero-padded
    mask:     (B, T_max) bool, True for valid frames
    weights:  optional (B,) per-video loss weight; None = plain mean
              (matches every existing caller, which never passes this)
    """
    losses = []
    for b in range(pred.size(0)):
        valid = mask[b]
        p = pred[b][valid]
        g = gt_score[b][valid]
        losses.append(pearson_loss(p, g, eps))
    losses = torch.stack(losses)
    if weights is not None:
        return (losses * weights).sum() / weights.sum()
    return losses.mean()


def batched_cosine_loss(pred, gt_score, mask, eps=1e-8):
    """Compute per-video cosine distance over valid frames in a padded batch."""
    losses = []
    for batch_idx in range(pred.size(0)):
        valid = mask[batch_idx]
        pred_valid = pred[batch_idx][valid]
        gt_valid = gt_score[batch_idx][valid]
        similarity = F.cosine_similarity(
            pred_valid.unsqueeze(0), gt_valid.unsqueeze(0), dim=1, eps=eps
        )
        losses.append(1.0 - similarity.squeeze(0))
    return torch.stack(losses).mean()


def batched_weighted_cross_entropy(logits, targets, mask, weight=None):
    """Average weighted cross-entropy within each video, then across videos."""
    losses = []
    for batch_idx in range(logits.size(0)):
        valid = mask[batch_idx]
        losses.append(F.cross_entropy(
            logits[batch_idx][valid], targets[batch_idx][valid], weight=weight
        ))
    return torch.stack(losses).mean()


def batched_focal_cls_loss(cls_logits, gt_score, mask, alpha, gamma=2.0, num_bins=10):
    """
    Per-frame multi-class focal loss for ordinal bin classification.

    cls_logits : (B, T_max, num_bins) raw logits
    gt_score   : (B, T_max) continuous scores in [0, 1]
    mask       : (B, T_max) bool, True for valid frames
    alpha      : (num_bins,) per-class inverse-frequency weights (on same device)
    gamma      : focusing parameter (default 2)
    num_bins   : number of ordinal bins
    """
    valid_logits = cls_logits[mask]
    valid_scores = gt_score[mask]
    target_bins  = (valid_scores * num_bins).long().clamp(0, num_bins - 1)

    log_prob = F.log_softmax(valid_logits, dim=-1)
    log_pt   = log_prob.gather(1, target_bins.unsqueeze(1)).squeeze(1)
    pt       = log_pt.exp()
    alpha_t  = alpha[target_bins]
    loss     = -alpha_t * (1.0 - pt) ** gamma * log_pt
    return loss.mean()


def batched_emd_cls_loss(cls_logits, gt_score, mask, num_bins=10):
    """
    Earth Mover's Distance loss for ordinal bin prediction.
    Penalises L1 distance between predicted CDF and target CDF,
    summed over the first K-1 bin positions.

    cls_logits : (B, T_max, num_bins) raw logits
    gt_score   : (B, T_max) continuous scores in [0, 1]
    mask       : (B, T_max) bool, True for valid frames
    """
    valid_logits = cls_logits[mask]
    valid_scores = gt_score[mask]
    target_bins  = (valid_scores * num_bins).long().clamp(0, num_bins - 1)

    pred_probs = F.softmax(valid_logits, dim=-1)
    pred_cdf   = torch.cumsum(pred_probs, dim=-1)

    target_oh  = torch.zeros_like(valid_logits)
    target_oh.scatter_(1, target_bins.unsqueeze(1), 1.0)
    target_cdf = torch.cumsum(target_oh, dim=-1)

    return (pred_cdf[:, :-1] - target_cdf[:, :-1]).abs().mean()


# ---------------------------------------------------------------------------
# Rank-variant losses (RankW / SoftNDCG / ApproxAP)
#
# Alternatives to the uniform peak-biased pairwise RankNet loss above, all
# aimed at the top-K decision boundary that mAP@15 actually scores, rather
# than treating every sampled pair equally. All operate on pre-sigmoid
# logits, matching ranknet_loss.
# ---------------------------------------------------------------------------

def _soft_rank_matrix(logits, temperature):
    """
    Pairwise soft 'b outranks a' matrix.

    S[a, b] = sigmoid((logits[b] - logits[a]) / temperature), diagonal zeroed.
    """
    diff = logits.unsqueeze(0) - logits.unsqueeze(1)  # diff[a, b] = logits[b] - logits[a]
    S = torch.sigmoid(diff / temperature)
    eye = torch.eye(logits.size(0), device=logits.device, dtype=S.dtype)
    return S * (1.0 - eye)


def _soft_rank(S):
    """1-indexed soft rank per item (rank 1 = top) from a soft-comparison matrix."""
    return 1.0 + S.sum(dim=1)


def _subsample_indices(n, max_frames, generator=None, device=None):
    """Uniform without-replacement subsample of frame indices, capped at max_frames."""
    if n <= max_frames:
        return torch.arange(n, device=device)
    return torch.randperm(n, generator=generator, device=device)[:max_frames]


def _true_ranks(gt):
    """0-indexed rank per item (0 = highest gt)."""
    order = torch.argsort(gt, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(gt.size(0), device=gt.device)
    return ranks


def _ideal_dcg_top_k(gt, k):
    """Maximum achievable DCG@K (0-indexed discount 1/log2(r+2)) for this gt vector."""
    top_vals, _ = torch.sort(gt, descending=True)
    top_vals = top_vals[:k]
    discounts = 1.0 / torch.log2(
        torch.arange(2, k + 2, device=gt.device, dtype=gt.dtype)
    )
    return (top_vals * discounts).sum()


def rankw_loss(
    pred_logits,
    gt,
    num_pairs=512,
    min_gap=0.1,
    top_frac=0.15,
    generator=None,
    eps=1e-8,
):
    """
    LambdaRank-style position-weighted pairwise loss for a single video.

    Reuses the same peak-biased pair sampling as ranknet_loss, but weights
    each pair's logistic loss by the |Delta DCG@K| that swapping the pair
    would cause, so pairs straddling the top-K cutoff dominate the loss.
    Weights are renormalized to mean 1 across accepted pairs so the overall
    gradient scale stays comparable to the unweighted RankNet loss under the
    same loss coefficient.

    pred_logits: (N,) pre-sigmoid logits
    gt:          (N,) ground-truth scores in [0, 1]
    """
    N = gt.size(0)
    i, j = peak_biased_pairs(gt, num_pairs, top_frac, generator=generator)

    gt_diff = gt[i] - gt[j]
    mask = gt_diff.abs() >= min_gap
    if mask.sum() == 0:
        return pred_logits.sum() * 0.0  # zero loss; keeps computation graph

    i, j = i[mask], j[mask]
    target = (gt[i] > gt[j]).float()
    pred_diff = pred_logits[i] - pred_logits[j]
    bce = F.binary_cross_entropy_with_logits(pred_diff, target, reduction='none')

    k = max(1, int(round(N * top_frac)))
    ranks = _true_ranks(gt)
    discount = 1.0 / torch.log2(ranks.to(gt.dtype) + 2.0)
    ideal_dcg = _ideal_dcg_top_k(gt, k).clamp_min(eps)

    weight = (gt[i] - gt[j]).abs() * (discount[i] - discount[j]).abs() / ideal_dcg
    weight = weight / weight.mean().clamp_min(eps)
    return (weight * bce).mean()


def batched_rankw_loss(
    pred_logits,
    gt_score,
    mask,
    num_pairs=512,
    min_gap=0.1,
    top_frac=0.15,
    generator=None,
):
    """Compute RankW loss over a padded batch. See rankw_loss for one video."""
    losses = []
    for b in range(pred_logits.size(0)):
        valid = mask[b]
        losses.append(rankw_loss(
            pred_logits[b][valid], gt_score[b][valid],
            num_pairs=num_pairs, min_gap=min_gap, top_frac=top_frac,
            generator=generator,
        ))
    return torch.stack(losses).mean()


def softndcg_loss(
    pred_logits,
    gt,
    top_frac=0.15,
    temperature=1.0,
    max_frames=600,
    generator=None,
    eps=1e-8,
):
    """
    ApproxNDCG@K (Qin et al., 2010) for a single video.

    Soft ranks come from pairwise sigmoid comparisons on logits; graded gain
    is the raw ground-truth score. For videos longer than max_frames, a
    uniform random subset of that size stands in for the full frame set (the
    O(N^2) soft-rank matrix is otherwise too expensive) -- an approximation,
    not an exact NDCG@K, for long videos.

    pred_logits: (N,) pre-sigmoid logits
    gt:          (N,) ground-truth scores in [0, 1]
    """
    N = gt.size(0)
    idx = _subsample_indices(N, max_frames, generator=generator, device=gt.device)
    logits_sub = pred_logits[idx]
    gt_sub = gt[idx]
    M = gt_sub.size(0)

    k = max(1, int(round(M * top_frac)))
    S = _soft_rank_matrix(logits_sub, temperature)
    soft_rank = _soft_rank(S)

    top_vals, top_idx = torch.topk(gt_sub, k)
    dcg = (top_vals / torch.log2(soft_rank[top_idx] + 1.0)).sum()
    ideal_dcg = _ideal_dcg_top_k(gt_sub, k).clamp_min(eps)
    return 1.0 - dcg / ideal_dcg


def batched_softndcg_loss(
    pred_logits,
    gt_score,
    mask,
    top_frac=0.15,
    temperature=1.0,
    max_frames=600,
    generator=None,
):
    """Compute SoftNDCG loss over a padded batch. See softndcg_loss for one video."""
    losses = []
    for b in range(pred_logits.size(0)):
        valid = mask[b]
        losses.append(softndcg_loss(
            pred_logits[b][valid], gt_score[b][valid],
            top_frac=top_frac, temperature=temperature,
            max_frames=max_frames, generator=generator,
        ))
    return torch.stack(losses).mean()


def approxap_loss(
    pred_logits,
    rel,
    temperature=1.0,
    max_frames=600,
    generator=None,
):
    """
    ApproxAP (Qin et al., 2010) for a single video, using binary relevance
    (e.g. gt_summary) rather than graded gt_score. The closest direct
    training-time surrogate for the mAP evaluation metric.

    pred_logits: (N,) pre-sigmoid logits
    rel:         (N,) binary relevance (0/1)
    """
    N = pred_logits.size(0)
    idx = _subsample_indices(N, max_frames, generator=generator, device=pred_logits.device)
    logits_sub = pred_logits[idx]
    rel_sub = rel[idx].to(pred_logits.dtype)

    R = rel_sub.sum()
    if R.item() == 0:
        return pred_logits.sum() * 0.0  # zero loss; keeps computation graph

    S = _soft_rank_matrix(logits_sub, temperature)
    soft_rank = _soft_rank(S)
    soft_pos_above = S @ rel_sub  # sum_b S[a, b] * rel[b]; diagonal already 0

    positive_mask = rel_sub > 0.5
    soft_precision = (1.0 + soft_pos_above[positive_mask]) / soft_rank[positive_mask]
    return 1.0 - soft_precision.sum() / R


def batched_approxap_loss(
    pred_logits,
    gt_summary,
    mask,
    temperature=1.0,
    max_frames=600,
    generator=None,
):
    """
    Compute ApproxAP loss over a padded batch.

    pred_logits: (B, T_max) pre-sigmoid logits
    gt_summary:  list of B numpy/tensor arrays, each (n_frames_b,) binary
    mask:        (B, T_max) bool, True for valid frames
    """
    losses = []
    for b in range(pred_logits.size(0)):
        valid = mask[b]
        p = pred_logits[b][valid]
        rel = torch.as_tensor(
            gt_summary[b], dtype=torch.float32, device=pred_logits.device
        )
        losses.append(approxap_loss(
            p, rel, temperature=temperature, max_frames=max_frames,
            generator=generator,
        ))
    return torch.stack(losses).mean()


def compute_rank_loss(
    pred_logits,
    gt_score,
    mask,
    variant='pairwise',
    num_pairs=512,
    min_gap=0.1,
    top_frac=0.15,
    temperature=1.0,
    max_frames=600,
    error_floor=0.25,
    error_temperature=1.0,
    gt_summary=None,
    generator=None,
    return_details=False,
    weights=None,
):
    """
    Single entry point for the pearson_ranknet rank term.

    variant='pairwise' (default) is exactly batched_ranknet_loss, so every
    existing config -- which never sets rank_variant -- is unaffected.

    weights: optional (B,) per-video loss weight, supported for pairwise and
    boundary variants. None preserves the unweighted historical behavior.
    """
    if variant == 'pairwise':
        return batched_ranknet_loss(
            pred_logits, gt_score, mask,
            num_pairs=num_pairs, min_gap=min_gap, top_frac=top_frac,
            generator=generator, return_details=return_details, weights=weights,
        )
    if variant in ('boundary', 'error_boundary'):
        return batched_boundary_ranknet_loss(
            pred_logits, gt_score, mask,
            num_pairs=num_pairs, min_gap=min_gap, top_frac=top_frac,
            generator=generator,
            error_conditioned=variant == 'error_boundary',
            error_floor=error_floor,
            error_temperature=error_temperature,
            return_details=return_details,
            weights=weights,
        )
    if return_details:
        raise ValueError(
            f"return_details is only supported for variant='pairwise', got '{variant}'"
        )
    if weights is not None:
        raise ValueError(
            f'weights are not supported for rank_variant={variant!r}'
        )
    if variant == 'rankw':
        return batched_rankw_loss(
            pred_logits, gt_score, mask,
            num_pairs=num_pairs, min_gap=min_gap, top_frac=top_frac,
            generator=generator,
        )
    if variant == 'softndcg':
        return batched_softndcg_loss(
            pred_logits, gt_score, mask,
            top_frac=top_frac, temperature=temperature,
            max_frames=max_frames, generator=generator,
        )
    if variant == 'approxap':
        if gt_summary is None:
            raise ValueError("variant='approxap' requires gt_summary")
        return batched_approxap_loss(
            pred_logits, gt_summary, mask,
            temperature=temperature, max_frames=max_frames,
            generator=generator,
        )
    raise ValueError(f'Unknown rank_variant: {variant}')
