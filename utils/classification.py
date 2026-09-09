import numpy as np


TREND_CLASS_NAMES = ('decreasing', 'flat', 'increasing')


def classification_metrics(confusion_matrix, class_names=TREND_CLASS_NAMES):
    """Compute aggregate and per-class metrics from a rows=true confusion matrix."""
    confusion = np.asarray(confusion_matrix, dtype=np.int64)
    num_classes = len(class_names)
    if confusion.shape != (num_classes, num_classes):
        raise ValueError(
            f'expected a {num_classes}x{num_classes} confusion matrix, '
            f'got {confusion.shape}'
        )

    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive, predicted,
        out=np.zeros_like(true_positive), where=predicted > 0,
    )
    recall = np.divide(
        true_positive, support,
        out=np.zeros_like(true_positive), where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive), where=(precision + recall) > 0,
    )
    total = support.sum()
    metrics = {
        'accuracy': float(true_positive.sum() / total) if total > 0 else 0.0,
        'balanced_accuracy': float(recall.mean()),
        'macro_precision': float(precision.mean()),
        'macro_recall': float(recall.mean()),
        'macro_f1': float(f1.mean()),
        'confusion_matrix': confusion.tolist(),
    }
    for index, name in enumerate(class_names):
        metrics[f'{name}_precision'] = float(precision[index])
        metrics[f'{name}_recall'] = float(recall[index])
        metrics[f'{name}_f1'] = float(f1[index])
        metrics[f'{name}_support'] = int(support[index])
    return metrics


def video_classification_metrics(confusion_matrices, class_names):
    """Average classification metrics over videos, ignoring absent true classes."""
    confusions = np.asarray(confusion_matrices, dtype=np.int64)
    num_classes = len(class_names)
    if confusions.ndim != 3 or confusions.shape[1:] != (num_classes, num_classes):
        raise ValueError(
            f'expected (N, {num_classes}, {num_classes}) confusion matrices, '
            f'got {confusions.shape}'
        )

    video_accuracy = []
    video_balanced_accuracy = []
    video_macro_f1 = []
    for confusion in confusions:
        true_positive = np.diag(confusion).astype(np.float64)
        support = confusion.sum(axis=1).astype(np.float64)
        predicted = confusion.sum(axis=0).astype(np.float64)
        present = support > 0
        precision = np.divide(
            true_positive, predicted,
            out=np.zeros_like(true_positive), where=predicted > 0,
        )
        recall = np.divide(
            true_positive, support,
            out=np.zeros_like(true_positive), where=present,
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(true_positive), where=(precision + recall) > 0,
        )
        total = support.sum()
        video_accuracy.append(true_positive.sum() / total if total > 0 else 0.0)
        video_balanced_accuracy.append(recall[present].mean() if present.any() else 0.0)
        video_macro_f1.append(f1[present].mean() if present.any() else 0.0)

    global_metrics = classification_metrics(confusions.sum(axis=0), class_names)
    return {
        'video_accuracy': float(np.mean(video_accuracy)),
        'video_balanced_accuracy': float(np.mean(video_balanced_accuracy)),
        'video_macro_f1': float(np.mean(video_macro_f1)),
        **{f'global_{key}': value for key, value in global_metrics.items()},
    }