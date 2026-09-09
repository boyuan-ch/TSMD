import json
import math
import os

import torch
import torch.nn.functional as F


SCHEMA_VERSION = 1
LOSS_NAMES = ('mse', 'pearson', 'ranknet')


def _tensor_norm(tensor):
    return torch.linalg.vector_norm(tensor.float())


def _gradient_norm(gradients):
    squared = sum(
        _tensor_norm(gradient).square()
        for gradient in gradients
        if gradient is not None
    )
    if isinstance(squared, int):
        return 0.0
    return math.sqrt(squared.item())


def _gradient_dot(first, second):
    value = sum(
        (left.float() * right.float()).sum()
        for left, right in zip(first, second)
        if left is not None and right is not None
    )
    return 0.0 if isinstance(value, int) else value.item()


def _gradient_cosine(first, second):
    denominator = _gradient_norm(first) * _gradient_norm(second)
    if denominator == 0.0:
        return 0.0
    return _gradient_dot(first, second) / denominator


def _parameter_group(name):
    if name.startswith('head.'):
        return 'head'
    if '_proj.' in name:
        return 'projections'
    if name.startswith('temporal_block.'):
        return 'temporal_blocks'
    if name.startswith('modality_block.'):
        return 'modality_blocks'
    return 'other'


def _counterfactual_ranknet_score_loss(output, mask, rank_details):
    losses = []
    for batch_index, details in enumerate(rank_details):
        valid_output = output[batch_index][mask[batch_index]]
        if details['accepted_count'] == 0:
            losses.append(valid_output.sum() * 0.0)
            continue
        score_difference = (
            valid_output[details['i']] - valid_output[details['j']]
        )
        losses.append(F.binary_cross_entropy_with_logits(
            score_difference, details['target']
        ))
    return torch.stack(losses).mean()


class LossDynamicsRecorder:
    def __init__(self, output_dir, flush_interval=20, metadata=None):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, 'loss_dynamics_steps.jsonl')
        self.epoch_path = os.path.join(
            output_dir, 'loss_dynamics_epochs.jsonl'
        )
        self.flush_interval = flush_interval
        self.pending = None
        self.records_since_flush = 0
        self.last_step = self._last_recorded_step()
        self.file = open(self.path, 'a', encoding='utf-8')
        if os.path.getsize(self.path) == 0:
            self._write({
                'record_type': 'metadata',
                'schema_version': SCHEMA_VERSION,
                **(metadata or {}),
            })

    def record_epoch(self, epoch, global_step, train_results, val_results,
                     weights):
        record = {
            'schema_version': SCHEMA_VERSION,
            'epoch': epoch,
            'global_step': global_step,
            'weights': list(weights),
            'train': {
                key: float(value) for key, value in train_results.items()
            },
            'val': {
                key: float(value) for key, value in val_results.items()
            },
        }
        existing_epochs = set()
        if os.path.exists(self.epoch_path):
            with open(self.epoch_path, encoding='utf-8') as existing:
                existing_epochs = {
                    json.loads(line)['epoch'] for line in existing
                }
        if epoch in existing_epochs:
            raise ValueError(f'Epoch {epoch} was already recorded')
        with open(self.epoch_path, 'a', encoding='utf-8') as output:
            output.write(json.dumps(record, sort_keys=True) + '\n')
        return record

    def _last_recorded_step(self):
        if not os.path.exists(self.path):
            return -1
        last_step = -1
        with open(self.path, encoding='utf-8') as existing:
            for line in existing:
                record = json.loads(line)
                if record.get('record_type') == 'step':
                    last_step = max(last_step, record['global_step'])
        return last_step

    def _write(self, record):
        self.file.write(json.dumps(record, sort_keys=True) + '\n')
        self.records_since_flush += 1
        if self.records_since_flush >= self.flush_interval:
            self.file.flush()
            self.records_since_flush = 0

    def measure_step(
        self,
        *,
        losses,
        output,
        logits,
        gt_score,
        mask,
        model,
        weights,
        rank_details,
        epoch,
        global_step,
        learning_rate,
        amp_scale,
    ):
        if self.pending is not None:
            raise RuntimeError('finish_step must be called before measure_step')
        if global_step <= self.last_step:
            raise ValueError(
                f'global_step {global_step} was already recorded '
                f'(last={self.last_step})'
            )

        parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        parameter_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        term_parameter_gradients = {}
        term_logit_gradients = {}
        for name in LOSS_NAMES:
            term_parameter_gradients[name] = torch.autograd.grad(
                losses[name], parameters, retain_graph=True, allow_unused=True
            )
            term_logit_gradients[name] = torch.autograd.grad(
                losses[name], logits, retain_graph=True, allow_unused=True
            )[0]

        mse_score_gradient = torch.autograd.grad(
            losses['mse'], output, retain_graph=True
        )[0]
        pearson_score_gradient = torch.autograd.grad(
            losses['pearson'], output, retain_graph=True
        )[0]
        counterfactual_rank_loss = _counterfactual_ranknet_score_loss(
            output, mask, rank_details
        )
        ranknet_score_gradient = torch.autograd.grad(
            counterfactual_rank_loss, output, retain_graph=True
        )[0]

        combined_gradients = []
        for parameter_index in range(len(parameters)):
            combined = None
            for name, weight in zip(LOSS_NAMES, weights):
                gradient = term_parameter_gradients[name][parameter_index]
                if gradient is None:
                    continue
                contribution = gradient.detach() * weight
                combined = (
                    contribution if combined is None
                    else combined + contribution
                )
            combined_gradients.append(combined)

        valid_output = output[mask].detach().float()
        valid_target = gt_score[mask].detach().float()
        sigmoid_derivative = valid_output * (1.0 - valid_output)
        record = {
            'record_type': 'step',
            'schema_version': SCHEMA_VERSION,
            'epoch': epoch,
            'global_step': global_step,
            'learning_rate': learning_rate,
            'amp_scale': amp_scale,
            'batch_videos': output.size(0),
            'batch_frames': int(mask.sum().item()),
            'sequence_length': output.size(1),
            'prediction_mean': valid_output.mean().item(),
            'prediction_std': valid_output.std(unbiased=False).item(),
            'target_mean': valid_target.mean().item(),
            'target_std': valid_target.std(unbiased=False).item(),
            'sigmoid_derivative_mean': sigmoid_derivative.mean().item(),
            'sigmoid_saturated_fraction': (
                (sigmoid_derivative < 0.05).float().mean().item()
            ),
            'counterfactual_ranknet_score_loss': counterfactual_rank_loss.item(),
        }
        for name, weight in zip(LOSS_NAMES, weights):
            record[f'loss/{name}'] = losses[name].detach().item()
            record[f'weight/{name}'] = weight
            record[f'weighted_loss/{name}'] = (
                losses[name].detach().item() * weight
            )
            logit_norm = _tensor_norm(term_logit_gradients[name]).item()
            parameter_norm = _gradient_norm(term_parameter_gradients[name])
            record[f'gradient/logit/{name}'] = logit_norm
            record[f'gradient/logit_weighted/{name}'] = abs(weight) * logit_norm
            record[f'gradient/parameter/{name}'] = parameter_norm
            record[f'gradient/parameter_weighted/{name}'] = (
                abs(weight) * parameter_norm
            )
            for group in ('head', 'projections', 'temporal_blocks',
                          'modality_blocks', 'other'):
                grouped = [
                    gradient
                    for parameter_name, gradient in zip(
                        parameter_names, term_parameter_gradients[name]
                    )
                    if _parameter_group(parameter_name) == group
                ]
                record[f'gradient/group/{group}/{name}'] = _gradient_norm(grouped)

        record['gradient/score/mse'] = _tensor_norm(mse_score_gradient).item()
        record['gradient/score/pearson'] = _tensor_norm(
            pearson_score_gradient
        ).item()
        record['gradient/score/ranknet_counterfactual'] = _tensor_norm(
            ranknet_score_gradient
        ).item()
        for first_index, first in enumerate(LOSS_NAMES):
            for second in LOSS_NAMES[first_index + 1:]:
                record[f'gradient/cosine/{first}_{second}'] = _gradient_cosine(
                    term_parameter_gradients[first],
                    term_parameter_gradients[second],
                )

        sampled_count = sum(item['sampled_count'] for item in rank_details)
        accepted_count = sum(item['accepted_count'] for item in rank_details)
        margins = []
        pair_gradient_magnitudes = []
        incorrect = []
        for details in rank_details:
            if details['accepted_count'] == 0:
                continue
            signed_margin = details['pred_diff'].detach() * (
                details['target'].detach() * 2.0 - 1.0
            )
            margins.append(signed_margin)
            pair_gradient_magnitudes.append(torch.sigmoid(-signed_margin))
            incorrect.append((signed_margin <= 0.0).float())
        record['ranknet/sampled_count'] = sampled_count
        record['ranknet/accepted_count'] = accepted_count
        record['ranknet/accepted_fraction'] = (
            accepted_count / sampled_count if sampled_count else 0.0
        )
        if margins:
            record['ranknet/signed_margin_mean'] = torch.cat(margins).mean().item()
            record['ranknet/pair_gradient_mean'] = torch.cat(
                pair_gradient_magnitudes
            ).mean().item()
            record['ranknet/incorrect_fraction'] = torch.cat(
                incorrect
            ).mean().item()
        else:
            record['ranknet/signed_margin_mean'] = 0.0
            record['ranknet/pair_gradient_mean'] = 0.0
            record['ranknet/incorrect_fraction'] = 0.0

        record['gradient/parameter_weighted_sum'] = _gradient_norm(
            combined_gradients
        )
        self.pending = (record, combined_gradients, parameters)
        return record

    def finish_step(self):
        if self.pending is None:
            raise RuntimeError('measure_step must be called before finish_step')
        record, expected_gradients, parameters = self.pending
        actual_gradients = [parameter.grad for parameter in parameters]
        record['gradient/optimizer_actual'] = _gradient_norm(actual_gradients)
        differences = []
        for expected, actual in zip(expected_gradients, actual_gradients):
            if expected is None and actual is None:
                differences.append(None)
            elif expected is None:
                differences.append(actual.detach())
            elif actual is None:
                differences.append(expected)
            else:
                differences.append(actual.detach() - expected)
        difference_norm = _gradient_norm(differences)
        expected_norm = record['gradient/parameter_weighted_sum']
        record['gradient/optimizer_difference'] = difference_norm
        record['gradient/optimizer_relative_error'] = (
            difference_norm / max(expected_norm, 1e-12)
        )
        self._write(record)
        self.last_step = record['global_step']
        self.pending = None
        return record

    def close(self):
        if self.pending is not None:
            raise RuntimeError('cannot close recorder with an unfinished step')
        self.file.flush()
        self.file.close()