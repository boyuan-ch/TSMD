import numpy as np
import torch


TREND_DECREASING = 0
TREND_FLAT = 1
TREND_INCREASING = 2


def running_average(scores, window_size=5):
    """Return a centered running average without zero-padding edge bias."""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError('window_size must be a positive odd integer')
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores.copy()
    radius = window_size // 2
    padded = np.pad(scores, (radius, radius), mode='edge')
    kernel = np.full(window_size, 1.0 / window_size, dtype=np.float32)
    return np.convolve(padded, kernel, mode='valid').astype(np.float32)


def smooth_salience_scores(scores, window_size=5):
    """Smooth salience scores while preserving the first score exactly."""
    scores = np.asarray(scores, dtype=np.float32)
    smoothed = running_average(scores, window_size)
    if smoothed.size:
        smoothed[0] = scores[0]
    return smoothed


def make_trend_labels(scores, window_size=5):
    """Smooth scores and return forward deltas aligned to the current frame."""
    smoothed = running_average(scores, window_size)
    deltas = np.zeros_like(smoothed)
    if smoothed.size > 1:
        deltas[:-1] = smoothed[1:] - smoothed[:-1]
    return deltas


def calibrate_flat_threshold(deltas, flat_quantile=0.5):
    """Choose a deadband threshold from absolute valid forward deltas."""
    if not 0.0 <= flat_quantile <= 1.0:
        raise ValueError('flat_quantile must be between 0 and 1')
    deltas = np.asarray(deltas, dtype=np.float32)
    if deltas.size == 0:
        raise ValueError('cannot calibrate a threshold from empty deltas')
    return float(np.quantile(np.abs(deltas), flat_quantile))


def make_trend_class_labels(scores, threshold, window_size=5):
    """Map smoothed forward deltas to decreasing, flat, and increasing classes."""
    if threshold < 0:
        raise ValueError('threshold must be non-negative')
    deltas = make_trend_labels(scores, window_size)
    labels = np.full(deltas.shape, TREND_FLAT, dtype=np.int8)
    labels[deltas < -threshold] = TREND_DECREASING
    labels[deltas > threshold] = TREND_INCREASING
    return labels


def reconstruct_trend(deltas, mask, eps=1e-8):
    """Integrate forward deltas from zero, then min-max normalize each video."""
    reconstructed = torch.zeros_like(deltas)
    for batch_idx in range(deltas.size(0)):
        length = int(mask[batch_idx].sum().item())
        if length <= 1:
            continue
        reconstructed[batch_idx, 1:length] = torch.cumsum(
            deltas[batch_idx, :length - 1], dim=0
        )
        valid_scores = reconstructed[batch_idx, :length]
        score_min = valid_scores.min()
        score_range = valid_scores.max() - score_min
        reconstructed[batch_idx, :length] = torch.where(
            score_range > eps,
            (valid_scores - score_min) / score_range.clamp_min(eps),
            torch.zeros_like(valid_scores),
        )
    return reconstructed