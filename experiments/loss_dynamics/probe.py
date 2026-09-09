import json
import math
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr

from dataset import CollateFn, Dataset
from utils.losses import pearson_loss, ranknet_loss


def _close_dataset(dataset):
    for name in ('gt_data', 'trend_data', 'trend_class_data', 'visual_data',
                 'text_data', 'audio_data'):
        handle = getattr(dataset, name, None)
        if handle is not None:
            handle.close()
            setattr(dataset, name, None)


def build_probe_batches(cfg, size=64, batch_size=8, seed=314159):
    dataset = Dataset(cfg, split='val')
    lengths = []
    with h5py.File(dataset._gt_path, 'r') as ground_truth:
        for index, video_id in enumerate(dataset.video_ids):
            try:
                length = int(ground_truth[video_id]['gt_score'].shape[0])
            except (KeyError, RuntimeError, OSError):
                continue
            lengths.append((index, video_id, length))

    sorted_entries = sorted(lengths, key=lambda item: (item[2], item[1]))
    strata = np.array_split(np.arange(len(sorted_entries)), 4)
    rng = np.random.default_rng(seed)
    quota = math.ceil(size / len(strata))
    selected = []
    items = []
    try:
        for stratum_index, stratum in enumerate(strata):
            accepted = 0
            for position in rng.permutation(stratum):
                index, video_id, length = sorted_entries[int(position)]
                try:
                    item = dataset[index]
                except (KeyError, RuntimeError, OSError):
                    item = None
                if item is None:
                    continue
                selected.append({
                    'video_id': video_id,
                    'length': length,
                    'length_stratum': stratum_index,
                })
                items.append(item)
                accepted += 1
                if accepted >= quota or len(items) >= size:
                    break
            if len(items) >= size:
                break
    finally:
        _close_dataset(dataset)

    if len(items) < size:
        raise RuntimeError(
            f'Only {len(items)} readable validation videos found for '
            f'a requested probe of {size}'
        )
    items = items[:size]
    selected = selected[:size]
    collate = CollateFn()
    batches = [
        collate(items[start:start + batch_size])
        for start in range(0, len(items), batch_size)
    ]
    return batches, {
        'seed': seed,
        'requested_size': size,
        'batch_size': batch_size,
        'videos': selected,
    }


class ValidationProbe:
    def __init__(
        self,
        cfg,
        output_dir,
        size=64,
        batch_size=8,
        seed=314159,
        flush_interval=20,
    ):
        self.device = cfg.device
        self.batches, manifest = build_probe_batches(
            cfg, size=size, batch_size=batch_size, seed=seed
        )
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, 'probe_manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding='utf-8') as existing:
                if json.load(existing) != manifest:
                    raise ValueError('Existing validation probe manifest differs')
        else:
            with open(manifest_path, 'w', encoding='utf-8') as output:
                json.dump(manifest, output, indent=2, sort_keys=True)
        self.path = os.path.join(output_dir, 'loss_dynamics_probe.jsonl')
        self.snapshot_path = os.path.join(
            output_dir, 'loss_dynamics_snapshots.h5'
        )
        self.last_step = self._last_recorded_step()
        self.file = open(self.path, 'a', encoding='utf-8')
        self.flush_interval = flush_interval
        self.records_since_flush = 0

    def has_snapshot(self, snapshot_name):
        if not os.path.exists(self.snapshot_path):
            return False
        with h5py.File(self.snapshot_path, 'r') as snapshots:
            return snapshot_name in snapshots

    def snapshot(
        self,
        model,
        snapshot_name,
        epoch,
        global_step,
        weights,
        num_pairs,
        min_gap,
        top_frac,
        seed,
    ):
        if self.has_snapshot(snapshot_name):
            return
        was_training = model.training
        model.eval()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with h5py.File(self.snapshot_path, 'a') as snapshots:
            snapshot_group = snapshots.create_group(snapshot_name)
            snapshot_group.attrs['epoch'] = epoch
            snapshot_group.attrs['global_step'] = global_step
            for batch in self.batches:
                visual = batch['visual_feat'].to(self.device, non_blocking=True)
                text = batch['text_feat'].to(self.device, non_blocking=True)
                audio = batch['audio_feat'].to(self.device, non_blocking=True)
                target = batch['gt_score'].to(self.device, non_blocking=True)
                mask = batch['mask'].to(self.device, non_blocking=True)
                output, logits, _ = model(visual, text, audio, mask=mask)
                for batch_index, video_id in enumerate(batch['video_id']):
                    valid = mask[batch_index]
                    prediction = output[batch_index][valid]
                    prediction_logits = logits[batch_index][valid]
                    ground_truth = target[batch_index][valid]
                    mse = F.mse_loss(prediction, ground_truth)
                    pearson = pearson_loss(prediction, ground_truth)
                    ranknet, details = ranknet_loss(
                        prediction_logits,
                        ground_truth,
                        num_pairs=num_pairs,
                        min_gap=min_gap,
                        top_frac=top_frac,
                        generator=generator,
                        return_details=True,
                    )
                    mse_gradient = torch.autograd.grad(
                        mse, prediction, retain_graph=True
                    )[0]
                    pearson_gradient = torch.autograd.grad(
                        pearson, prediction, retain_graph=True
                    )[0]
                    ranknet_gradient = torch.autograd.grad(
                        ranknet, prediction_logits, retain_graph=True
                    )[0]
                    if details['accepted_count']:
                        score_difference = (
                            prediction[details['i']]
                            - prediction[details['j']]
                        )
                        counterfactual_loss = F.binary_cross_entropy_with_logits(
                            score_difference, details['target']
                        )
                    else:
                        counterfactual_loss = prediction.sum() * 0.0
                    counterfactual_gradient = torch.autograd.grad(
                        counterfactual_loss, prediction, retain_graph=True
                    )[0]

                    video_group = snapshot_group.create_group(video_id)
                    arrays = {
                        'prediction': prediction,
                        'logits': prediction_logits,
                        'target': ground_truth,
                        'gradient_score_mse': mse_gradient,
                        'gradient_score_mse_weighted': mse_gradient * weights[0],
                        'gradient_score_pearson': pearson_gradient,
                        'gradient_score_pearson_weighted': (
                            pearson_gradient * weights[1]
                        ),
                        'gradient_logit_ranknet': ranknet_gradient,
                        'gradient_logit_ranknet_weighted': (
                            ranknet_gradient * weights[2]
                        ),
                        'gradient_score_ranknet_counterfactual': (
                            counterfactual_gradient
                        ),
                        'gradient_score_ranknet_counterfactual_weighted': (
                            counterfactual_gradient * weights[2]
                        ),
                    }
                    for dataset_name, tensor in arrays.items():
                        video_group.create_dataset(
                            dataset_name,
                            data=tensor.detach().float().cpu().numpy(),
                            compression='gzip',
                        )
                    video_group.attrs['ranknet_sampled_count'] = (
                        details['sampled_count']
                    )
                    video_group.attrs['ranknet_accepted_count'] = (
                        details['accepted_count']
                    )
                del visual, text, audio, target, mask, output, logits
        if was_training:
            model.train()

    def _last_recorded_step(self):
        if not os.path.exists(self.path):
            return -1
        last_step = -1
        with open(self.path, encoding='utf-8') as existing:
            for line in existing:
                last_step = max(last_step, json.loads(line)['global_step'])
        return last_step

    def evaluate(self, model, epoch, global_step):
        if global_step <= self.last_step:
            raise ValueError(
                f'Probe step {global_step} was already recorded '
                f'(last={self.last_step})'
            )
        was_training = model.training
        model.eval()
        per_video = {}
        with torch.no_grad():
            for batch in self.batches:
                visual = batch['visual_feat'].to(self.device, non_blocking=True)
                text = batch['text_feat'].to(self.device, non_blocking=True)
                audio = batch['audio_feat'].to(self.device, non_blocking=True)
                target = batch['gt_score'].to(self.device, non_blocking=True)
                mask = batch['mask'].to(self.device, non_blocking=True)
                output, _, _ = model(visual, text, audio, mask=mask)
                for batch_index, video_id in enumerate(batch['video_id']):
                    valid = mask[batch_index]
                    prediction = output[batch_index][valid].float()
                    ground_truth = target[batch_index][valid].float()
                    prediction_cpu = prediction.cpu().numpy()
                    target_cpu = ground_truth.cpu().numpy()
                    derivative = prediction * (1.0 - prediction)
                    per_video[video_id] = {
                        'mse': torch.mean(
                            (prediction - ground_truth).square()
                        ).item(),
                        'pearson': float(np.corrcoef(
                            prediction_cpu, target_cpu
                        )[0, 1]),
                        'spearman': float(spearmanr(
                            prediction_cpu, target_cpu
                        ).statistic),
                        'kendall': float(kendalltau(
                            prediction_cpu, target_cpu
                        ).statistic),
                        'prediction_mean': prediction.mean().item(),
                        'prediction_std': prediction.std(unbiased=False).item(),
                        'target_std': ground_truth.std(unbiased=False).item(),
                        'sigmoid_saturated_fraction': (
                            (derivative < 0.05).float().mean().item()
                        ),
                        'frames': prediction.numel(),
                    }
                del visual, text, audio, target, mask, output
        if was_training:
            model.train()

        metric_names = (
            'mse', 'pearson', 'spearman', 'kendall', 'prediction_mean',
            'prediction_std', 'target_std', 'sigmoid_saturated_fraction',
        )
        aggregate = {}
        for metric_name in metric_names:
            values = np.array([
                metrics[metric_name] for metrics in per_video.values()
            ])
            aggregate[metric_name] = float(np.nanmean(values))
        record = {
            'epoch': epoch,
            'global_step': global_step,
            'aggregate': aggregate,
            'per_video': per_video,
        }
        self.file.write(json.dumps(record, sort_keys=True) + '\n')
        self.records_since_flush += 1
        if self.records_since_flush >= self.flush_interval:
            self.file.flush()
            self.records_since_flush = 0
        self.last_step = global_step
        return record

    def close(self):
        self.file.flush()
        self.file.close()