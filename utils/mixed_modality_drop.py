from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch

from utils.frame_drop import MODALITIES, training_video_drop_masks


MIXED_MODALITY_DROP_VERSION = 'sha256-mixed-modality-drop-v1'
MIXED_DROP_POLICIES = ('none', 'mixed')
MIXED_DROP_TEMPORAL_PATTERNS = ('independent', 'synchronized')


@dataclass(frozen=True)
class MixedDropState:
    regime: str
    ratio: float = 0.0
    modality: str | None = None


def validate_mixed_drop_compatibility(
    policy,
    frame_drop_pattern,
    modality_drop_policy,
):
    if policy not in MIXED_DROP_POLICIES:
        raise ValueError(
            f'unknown mixed drop policy {policy!r}; expected one of '
            f'{MIXED_DROP_POLICIES}'
        )
    if policy != 'none' and (
        frame_drop_pattern != 'none' or modality_drop_policy != 'none'
    ):
        raise ValueError(
            'mixed drop requires legacy frame and whole-modality drop policies '
            'to be disabled'
        )


def _stable_rng(*parts):
    payload = json.dumps(parts, ensure_ascii=True, separators=(',', ':'))
    digest = hashlib.sha256(payload.encode('utf-8')).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'big'))


def validate_mixed_drop_config(probabilities, temporal_ratios):
    probabilities = tuple(float(value) for value in probabilities)
    temporal_ratios = tuple(float(value) for value in temporal_ratios)
    if len(probabilities) != 3:
        raise ValueError('mixed drop requires clean, temporal, and whole probabilities')
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError('mixed drop probabilities must be finite and non-negative')
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-8):
        raise ValueError('mixed drop probabilities must sum to 1')
    if not temporal_ratios or len(set(temporal_ratios)) != len(temporal_ratios):
        raise ValueError('mixed temporal ratios must be non-empty and unique')
    if any(
        not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0
        for ratio in temporal_ratios
    ):
        raise ValueError('mixed temporal ratios must be in (0, 1]')
    return probabilities, temporal_ratios


def validate_mixed_drop_temporal_pattern(temporal_pattern):
    if temporal_pattern not in MIXED_DROP_TEMPORAL_PATTERNS:
        raise ValueError(
            f'unknown mixed temporal pattern {temporal_pattern!r}; expected one of '
            f'{MIXED_DROP_TEMPORAL_PATTERNS}'
        )
    return temporal_pattern


def sample_mixed_drop_state(
    probabilities,
    temporal_ratios,
    drop_seed,
    epoch,
    video_id,
):
    probabilities, temporal_ratios = validate_mixed_drop_config(
        probabilities, temporal_ratios
    )
    regime = ('clean', 'temporal', 'whole')[int(_stable_rng(
        MIXED_MODALITY_DROP_VERSION,
        drop_seed,
        epoch,
        video_id,
        'regime',
    ).choice(3, p=probabilities))]
    if regime == 'clean':
        return MixedDropState('clean')
    if regime == 'temporal':
        index = int(_stable_rng(
            MIXED_MODALITY_DROP_VERSION,
            drop_seed,
            epoch,
            video_id,
            'temporal-ratio',
        ).integers(len(temporal_ratios)))
        return MixedDropState('temporal', ratio=temporal_ratios[index])
    index = int(_stable_rng(
        MIXED_MODALITY_DROP_VERSION,
        drop_seed,
        epoch,
        video_id,
        'whole-modality',
    ).integers(len(MODALITIES)))
    return MixedDropState('whole', modality=MODALITIES[index])


def apply_training_mixed_drop(
    features,
    valid_mask,
    video_ids,
    probabilities,
    temporal_ratios,
    drop_seed,
    epoch,
    temporal_pattern='independent',
):
    probabilities, temporal_ratios = validate_mixed_drop_config(
        probabilities, temporal_ratios
    )
    temporal_pattern = validate_mixed_drop_temporal_pattern(temporal_pattern)
    batch_size, max_length = valid_mask.shape
    if len(video_ids) != batch_size:
        raise ValueError('video_ids and valid_mask batch sizes differ')
    states = [
        sample_mixed_drop_state(
            probabilities, temporal_ratios, drop_seed, epoch, video_id
        )
        for video_id in video_ids
    ]
    drop_mask = torch.zeros(
        batch_size,
        max_length,
        len(MODALITIES),
        dtype=torch.bool,
        device=valid_mask.device,
    )
    if all(state.regime == 'clean' for state in states):
        return features, drop_mask, states

    corrupted = {modality: features[modality].clone() for modality in MODALITIES}
    lengths = valid_mask.sum(dim=1).tolist()
    for batch_index, (video_id, length, state) in enumerate(
        zip(video_ids, lengths, states)
    ):
        length = int(length)
        if state.regime == 'temporal':
            masks = training_video_drop_masks(
                length,
                state.ratio,
                drop_seed,
                epoch,
                video_id,
                temporal_pattern,
            )
            for modality_index, modality in enumerate(MODALITIES):
                modality_mask = torch.from_numpy(masks[modality]).to(
                    valid_mask.device
                )
                drop_mask[batch_index, :length, modality_index] = modality_mask
                corrupted[modality][batch_index, :length][modality_mask] = 0.0
        elif state.regime == 'whole':
            modality_index = MODALITIES.index(state.modality)
            drop_mask[batch_index, :length, modality_index] = True
            corrupted[state.modality][batch_index, :length] = 0.0
    return corrupted, drop_mask, states