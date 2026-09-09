import hashlib
import json

import numpy as np
import torch


MODALITIES = ('visual', 'text', 'audio')
WHOLE_MODALITY_DROP_POLICIES = (
    'none',
    'bernoulli',
    'categorical',
)
WHOLE_MODALITY_DROP_VERSION = 'sha256-whole-modality-drop-v1'


def _stable_rng(*parts):
    payload = json.dumps(parts, ensure_ascii=True, separators=(',', ':'))
    digest = hashlib.sha256(payload.encode('utf-8')).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'big'))


def validate_whole_modality_drop_config(policy, probability):
    if policy not in WHOLE_MODALITY_DROP_POLICIES:
        raise ValueError(
            f'Unknown whole-modality drop policy {policy!r}; expected one of '
            f'{WHOLE_MODALITY_DROP_POLICIES}'
        )
    probability = float(probability)
    if probability < 0.0 or probability > 1.0:
        raise ValueError('whole-modality drop probability must be in [0, 1]')
    if policy == 'none' and probability != 0.0:
        raise ValueError(
            'whole-modality drop policy none requires probability 0'
        )
    return probability


def validate_drop_policy_compatibility(frame_drop_pattern, modality_drop_policy):
    if frame_drop_pattern != 'none' and modality_drop_policy != 'none':
        raise ValueError(
            'training frame drop and whole-modality drop are mutually exclusive'
        )


def sample_whole_modality_state(
    policy,
    probability,
    drop_seed,
    epoch,
    video_id,
):
    probability = validate_whole_modality_drop_config(policy, probability)
    if policy == 'none':
        return ()
    if policy == 'categorical':
        generator = _stable_rng(
            WHOLE_MODALITY_DROP_VERSION,
            drop_seed,
            epoch,
            video_id,
            policy,
            'state',
        )
        state = int(generator.integers(len(MODALITIES) + 1))
        return () if state == 0 else (MODALITIES[state - 1],)

    dropped = []
    for modality in MODALITIES:
        generator = _stable_rng(
            WHOLE_MODALITY_DROP_VERSION,
            drop_seed,
            epoch,
            video_id,
            policy,
            modality,
        )
        if generator.random() < probability:
            dropped.append(modality)
    return tuple(dropped)


def apply_training_whole_modality_drop(
    features,
    valid_mask,
    video_ids,
    policy,
    probability,
    drop_seed,
    epoch,
):
    probability = validate_whole_modality_drop_config(policy, probability)
    batch_size, max_length = valid_mask.shape
    if len(video_ids) != batch_size:
        raise ValueError('video_ids and valid_mask batch sizes differ')
    if policy == 'none':
        empty_mask = torch.zeros(
            batch_size,
            max_length,
            len(MODALITIES),
            dtype=torch.bool,
            device=valid_mask.device,
        )
        return features, empty_mask, [()] * batch_size

    states = [
        sample_whole_modality_state(
            policy,
            probability,
            drop_seed,
            epoch,
            video_id,
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
    corrupted = {modality: features[modality].clone() for modality in MODALITIES}
    for batch_index, dropped_modalities in enumerate(states):
        for modality in dropped_modalities:
            modality_index = MODALITIES.index(modality)
            drop_mask[batch_index, :, modality_index] = valid_mask[batch_index]
            corrupted[modality][batch_index][valid_mask[batch_index]] = 0.0
    return corrupted, drop_mask, states