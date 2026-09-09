import hashlib
import json

import numpy as np
import torch


MODALITIES = ('visual', 'text', 'audio')
FRAME_DROP_PATTERNS = (*MODALITIES, 'independent', 'synchronized')
TRAIN_FRAME_DROP_PATTERNS = ('none', *FRAME_DROP_PATTERNS)
MASK_VERSION = 'sha256-exact-k-train-v1'


def _stable_rng(*parts):
    payload = json.dumps(parts, ensure_ascii=True, separators=(',', ':'))
    digest = hashlib.sha256(payload.encode('utf-8')).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'big'))


def validate_frame_drop_config(pattern, ratios):
    if pattern not in TRAIN_FRAME_DROP_PATTERNS:
        raise ValueError(
            f'Unknown train_frame_drop_pattern {pattern!r}; expected one of '
            f'{TRAIN_FRAME_DROP_PATTERNS}'
        )
    ratios = tuple(float(ratio) for ratio in ratios)
    if not ratios:
        raise ValueError('train_frame_drop_ratios must not be empty')
    if len(set(ratios)) != len(ratios):
        raise ValueError('train_frame_drop_ratios must be unique')
    if any(ratio < 0.0 or ratio > 1.0 for ratio in ratios):
        raise ValueError('train_frame_drop_ratios must be in [0, 1]')
    if pattern == 'none' and ratios != (0.0,):
        raise ValueError(
            'train_frame_drop_pattern=none requires train_frame_drop_ratios 0'
        )
    return ratios


def sample_training_ratio(ratios, drop_seed, epoch, video_id, pattern):
    if len(ratios) == 1:
        return ratios[0]
    generator = _stable_rng(
        MASK_VERSION,
        drop_seed,
        epoch,
        video_id,
        pattern,
        'ratio',
    )
    return ratios[int(generator.integers(len(ratios)))]


def curriculum_training_ratios(
    ratios,
    epoch,
    stage1_end=3,
    stage2_end=8,
):
    positive = tuple(ratio for ratio in ratios if ratio > 0.0)
    if not positive:
        return (0.0,)
    limits = (0.1, 0.3, float('inf'))
    stage = 0 if epoch <= stage1_end else 1 if epoch <= stage2_end else 2
    active = tuple(ratio for ratio in positive if ratio <= limits[stage])
    if not active:
        raise ValueError(
            f'No positive frame-drop ratio is active at curriculum stage {stage + 1}'
        )
    return active


def sample_curriculum_ratio(
    ratios,
    drop_seed,
    epoch,
    video_id,
    pattern,
    clean_probability=0.0,
    stage1_end=3,
    stage2_end=8,
):
    if clean_probability < 0.0 or clean_probability > 1.0:
        raise ValueError('clean_probability must be in [0, 1]')
    clean_generator = _stable_rng(
        MASK_VERSION,
        drop_seed,
        epoch,
        video_id,
        pattern,
        'clean-view',
    )
    if clean_generator.random() < clean_probability:
        return 0.0
    active = curriculum_training_ratios(
        ratios,
        epoch,
        stage1_end=stage1_end,
        stage2_end=stage2_end,
    )
    return sample_training_ratio(active, drop_seed, epoch, video_id, pattern)


def _exact_temporal_mask(length, ratio, *key_parts):
    count = int(round(ratio * length))
    mask = np.zeros(length, dtype=bool)
    if count:
        selected = _stable_rng(MASK_VERSION, *key_parts).choice(
            length, size=count, replace=False
        )
        mask[selected] = True
    return mask


def training_video_drop_masks(
    length,
    ratio,
    drop_seed,
    epoch,
    video_id,
    pattern,
):
    if pattern not in FRAME_DROP_PATTERNS:
        raise ValueError(
            f'Unknown frame-drop pattern {pattern!r}; expected one of '
            f'{FRAME_DROP_PATTERNS}'
        )
    ratio_key = f'{ratio:.8f}'
    masks = {
        modality: np.zeros(length, dtype=bool) for modality in MODALITIES
    }
    key = (drop_seed, epoch, video_id, pattern, ratio_key)
    if pattern in MODALITIES:
        masks[pattern] = _exact_temporal_mask(
            length, ratio, *key, pattern
        )
    elif pattern == 'independent':
        for modality in MODALITIES:
            masks[modality] = _exact_temporal_mask(
                length, ratio, *key, modality
            )
    else:
        shared = _exact_temporal_mask(length, ratio, *key, 'shared')
        masks = {modality: shared.copy() for modality in MODALITIES}
    return masks


def apply_training_frame_drop(
    features,
    valid_mask,
    video_ids,
    pattern,
    ratios,
    drop_seed,
    epoch,
    curriculum=False,
    clean_probability=0.0,
    curriculum_stage1_end=3,
    curriculum_stage2_end=8,
):
    ratios = validate_frame_drop_config(pattern, ratios)
    batch_size, max_length = valid_mask.shape
    if len(video_ids) != batch_size:
        raise ValueError('video_ids and valid_mask batch sizes differ')

    drop_mask = torch.zeros(
        batch_size,
        max_length,
        len(MODALITIES),
        dtype=torch.bool,
        device=valid_mask.device,
    )
    if pattern == 'none':
        return features, drop_mask, [0.0] * batch_size

    if curriculum:
        sampled_ratios = [
            sample_curriculum_ratio(
                ratios,
                drop_seed,
                epoch,
                video_id,
                pattern,
                clean_probability=clean_probability,
                stage1_end=curriculum_stage1_end,
                stage2_end=curriculum_stage2_end,
            )
            for video_id in video_ids
        ]
    else:
        sampled_ratios = [
            sample_training_ratio(ratios, drop_seed, epoch, video_id, pattern)
            for video_id in video_ids
        ]
    if not any(sampled_ratios):
        return features, drop_mask, sampled_ratios

    corrupted = {modality: features[modality].clone() for modality in MODALITIES}
    lengths = valid_mask.sum(dim=1).tolist()
    for batch_index, (video_id, length, ratio) in enumerate(
        zip(video_ids, lengths, sampled_ratios)
    ):
        length = int(length)
        masks = training_video_drop_masks(
            length,
            ratio,
            drop_seed,
            epoch,
            video_id,
            pattern,
        )
        for modality_index, modality in enumerate(MODALITIES):
            modality_mask = torch.from_numpy(masks[modality]).to(
                valid_mask.device
            )
            drop_mask[batch_index, :length, modality_index] = modality_mask
            corrupted[modality][batch_index, :length][modality_mask] = 0.0
    return corrupted, drop_mask, sampled_ratios