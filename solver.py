import os
import time
import h5py
import json
import torch
import torch.nn.functional as F
import itertools
import numpy as np
from tqdm import tqdm

from utils.wandb import wandb_training
from utils.logger import setup_logger, log_training
from utils.compute_metrics import evaluate_summary, evaluate_highlight
from utils.losses import batched_ranknet_loss, batched_pearson_loss, batched_cosine_loss, batched_weighted_cross_entropy, batched_focal_cls_loss, batched_emd_cls_loss, compute_rank_loss
from utils.trend import reconstruct_trend
from utils.classification import classification_metrics, video_classification_metrics
from models import build_model, build_optimizer, build_scheduler
from experiments.loss_dynamics import LossDynamicsRecorder
from experiments.loss_dynamics.probe import ValidationProbe
from utils.frame_drop import (
    MASK_VERSION,
    MODALITIES,
    apply_training_frame_drop,
    curriculum_training_ratios,
)
from utils.whole_modality_drop import (
    WHOLE_MODALITY_DROP_VERSION,
    apply_training_whole_modality_drop,
)
from utils.mixed_modality_drop import (
    MIXED_MODALITY_DROP_VERSION,
    apply_training_mixed_drop,
    validate_mixed_drop_compatibility,
    validate_mixed_drop_config,
    validate_mixed_drop_temporal_pattern,
)

CLASSIFICATION_LOSS_TYPES = ('trend_cls', 'trend_cls_mse', 'salience_cls')
SALIENCE_CLASS_NAMES = ('bin_0', 'bin_1', 'bin_2', 'bin_3')


def ctm_warmup_factor(epoch, start_epoch, end_epoch):
    if epoch <= start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return 1.0
    return (epoch - start_epoch) / max(1, end_epoch - start_epoch)


def compute_optional_ctm(model, cfg, epoch, visual, text, audio, mask):
    weight = getattr(cfg, 'ctm_weight', 0.0)
    if weight <= 0.0 or not hasattr(model, 'compute_ctm_loss'):
        return None
    factor = ctm_warmup_factor(
        epoch,
        getattr(cfg, 'ctm_warmup_start', 3),
        getattr(cfg, 'ctm_warmup_end', 5),
    )
    if factor <= 0.0:
        return None
    radius = getattr(cfg, 'ctm_radius', -1)
    ctm_loss, diagnostics = model.compute_ctm_loss(
        visual,
        text,
        audio,
        mask,
        temperature=getattr(cfg, 'ctm_temperature', 0.07),
        sigma=getattr(cfg, 'ctm_sigma', 1.0),
        radius=None if radius < 0 else radius,
        num_negatives=getattr(cfg, 'ctm_num_negatives', 16),
    )
    return ctm_loss, weight * factor, diagnostics


def _linear_progress(epoch, start_epoch, end_epoch):
    if epoch <= start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return 1.0
    return (epoch - start_epoch) / (end_epoch - start_epoch)


def combined_loss_weights(cfg, epoch):
    """Return active (MSE, Pearson, RankNet) coefficients for an epoch."""
    schedule = cfg.combined_schedule
    final_weights = (
        cfg.combined_mse_weight,
        cfg.combined_pearson_weight,
        cfg.combined_ranknet_weight,
    )
    if schedule in ('static', 'anchor'):
        return final_weights

    first_end = cfg.combined_stage1_end
    second_end = cfg.combined_stage2_end
    if schedule == 'mse_to_pr':
        progress = _linear_progress(epoch, first_end, second_end)
        return (
            1.0 + progress * (final_weights[0] - 1.0),
            progress * final_weights[1],
            progress * final_weights[2],
        )

    mse_to_pearson = _linear_progress(epoch, first_end, second_end)
    if schedule == 'rank_decay' and epoch <= second_end:
        rank_target = cfg.combined_ranknet_start_weight
        return (
            1.0 - mse_to_pearson,
            mse_to_pearson * (1.0 - rank_target),
            mse_to_pearson * rank_target,
        )
    if epoch <= second_end:
        return (1.0 - mse_to_pearson, mse_to_pearson, 0.0)

    third_end = cfg.combined_stage3_end
    if schedule == 'mse_to_p_to_pr':
        progress = _linear_progress(epoch, second_end, third_end)
        return (
            progress * final_weights[0],
            1.0 + progress * (final_weights[1] - 1.0),
            progress * final_weights[2],
        )
    if schedule == 'rank_decay':
        progress = _linear_progress(epoch, second_end, third_end)
        rank_weight = (
            cfg.combined_ranknet_start_weight
            + progress * (
                cfg.combined_ranknet_end_weight
                - cfg.combined_ranknet_start_weight
            )
        )
        return (0.0, 1.0 - rank_weight, rank_weight)
    if schedule == 'rank_ramp':
        progress = _linear_progress(epoch, second_end, third_end)
        rank_weight = progress * final_weights[2]
        return (0.0, 1.0 - rank_weight, rank_weight)
    raise ValueError(f'Unknown combined loss schedule: {schedule}')


def ranknet_seed_offset(cfg):
    offset = getattr(cfg, 'ranknet_seed_offset', None)
    if offset is None and cfg.loss_dynamics:
        return cfg.loss_dynamics_ranknet_seed_offset
    return offset


def masked_logit_distillation_loss(student_logits, teacher_logits, mask):
    if not mask.any():
        return student_logits.sum() * 0.0
    return F.mse_loss(student_logits[mask], teacher_logits[mask])


def observable_consistency_mask(valid_mask, *drop_masks):
    combined_drop_mask = torch.zeros(
        (*valid_mask.shape, len(MODALITIES)),
        dtype=torch.bool,
        device=valid_mask.device,
    )
    for drop_mask in drop_masks:
        combined_drop_mask |= drop_mask
    return valid_mask & ~combined_drop_mask.all(dim=2)

class Solver:
    def __init__(self, cfg, train_loader, val_loader, test_loader):
        self.cfg = cfg
        modality_drop_defaults = {
            'train_frame_drop_teacher_mode': 'frozen',
            'train_modality_drop_policy': 'none',
            'train_modality_drop_probability': 0.0,
            'train_modality_drop_seed': 42,
            'train_mixed_drop_policy': 'none',
            'train_mixed_drop_probabilities': [0.25, 0.375, 0.375],
            'train_mixed_drop_temporal_ratios': [0.1, 0.3, 0.5],
            'train_mixed_drop_temporal_pattern': 'independent',
            'train_mixed_drop_seed': 42,
        }
        for name, value in modality_drop_defaults.items():
            if not hasattr(self.cfg, name):
                setattr(self.cfg, name, value)
        validate_mixed_drop_compatibility(
            self.cfg.train_mixed_drop_policy,
            self.cfg.train_frame_drop_pattern,
            self.cfg.train_modality_drop_policy,
        )
        mixed_probabilities, mixed_ratios = validate_mixed_drop_config(
            self.cfg.train_mixed_drop_probabilities,
            self.cfg.train_mixed_drop_temporal_ratios,
        )
        self.cfg.train_mixed_drop_probabilities = list(mixed_probabilities)
        self.cfg.train_mixed_drop_temporal_ratios = list(mixed_ratios)
        self.cfg.train_mixed_drop_temporal_pattern = (
            validate_mixed_drop_temporal_pattern(
                self.cfg.train_mixed_drop_temporal_pattern
            )
        )
        is_resume = os.path.exists(os.path.join(self.cfg.output_dir, 'resume_ckpt.pth'))
        self.logger = setup_logger('solver', self.cfg.output_dir, overwrite=not is_resume)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.model = build_model(self.cfg).to(self.cfg.device)
        self.frame_drop_teacher = None
        if self.cfg.train_frame_drop_view_mode == 'dual':
            if self.cfg.loss_dynamics:
                raise ValueError('dual-view frame-drop training does not support loss dynamics')
            if getattr(self.cfg, 'ctm_weight', 0.0) > 0.0:
                raise ValueError('dual-view frame-drop training does not support CTM')
        if (
            self.cfg.train_frame_drop_kd_weight > 0.0
            and self.cfg.train_frame_drop_teacher_mode == 'frozen'
        ):
            self.frame_drop_teacher = build_model(self.cfg).to(self.cfg.device)
            self.frame_drop_teacher.load_state_dict(torch.load(
                self.cfg.train_frame_drop_teacher_ckpt,
                map_location=self.cfg.device,
                weights_only=True,
            ))
            self.frame_drop_teacher.eval()
            self.frame_drop_teacher.requires_grad_(False)
        self.optimizer = build_optimizer(self.cfg, self.model)
        # Pass train-set size for step-based schedulers (None when no loader provided)
        _num_steps = len(train_loader) * self.cfg.num_epochs if train_loader is not None else None
        self.scheduler = build_scheduler(self.cfg, self.optimizer, num_training_steps=_num_steps)
        
        self.criterion = torch.nn.MSELoss()
        if cfg.use_genre:
            self.criterion_genre = torch.nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.cfg.amp)
        self.cls_alpha = None  # set by _compute_cls_alpha() when loss_type in cls_mse_*
        if self.cfg.loss_type in ('trend_cls', 'trend_cls_mse'):
            self.cls_alpha = self._compute_trend_class_alpha()
        elif self.cfg.loss_type == 'salience_cls':
            self.cls_alpha = self._compute_cls_alpha(self.cfg.num_cls_bins)
        self.combined_weights = (
            combined_loss_weights(self.cfg, 0)
            if self.cfg.loss_type == 'pearson_ranknet'
            else None
        )
        self.loss_dynamics_recorder = None
        self.loss_dynamics_probe = None
        self.ranknet_generator = None
        ranknet_offset = ranknet_seed_offset(self.cfg)
        if ranknet_offset is not None:
            if self.cfg.loss_type != 'pearson_ranknet':
                raise ValueError(
                    'ranknet_seed_offset requires loss_type=pearson_ranknet'
                )
            self.ranknet_generator = torch.Generator(device=self.cfg.device)
            self.ranknet_generator.manual_seed(
                self.cfg.seed + ranknet_offset
            )
        self.logger.info(
            f'\t# Training frame drop: pattern={self.cfg.train_frame_drop_pattern} '
            f'ratios={self.cfg.train_frame_drop_ratios} '
            f'seed={self.cfg.train_frame_drop_seed} version={MASK_VERSION} '
            f'view_mode={self.cfg.train_frame_drop_view_mode} '
            f'curriculum={self.cfg.train_frame_drop_curriculum} '
            f'clean_probability={self.cfg.train_frame_drop_clean_probability} '
            f'kd_weight={self.cfg.train_frame_drop_kd_weight} '
            f'teacher_mode={self.cfg.train_frame_drop_teacher_mode}'
        )
        self.logger.info(
            f'\t# Training whole-modality drop: '
            f'policy={self.cfg.train_modality_drop_policy} '
            f'probability={self.cfg.train_modality_drop_probability} '
            f'seed={self.cfg.train_modality_drop_seed} '
            f'version={WHOLE_MODALITY_DROP_VERSION}'
        )
        self.logger.info(
            f'\t# Training mixed drop: policy={self.cfg.train_mixed_drop_policy} '
            f'probabilities={self.cfg.train_mixed_drop_probabilities} '
            f'temporal_ratios={self.cfg.train_mixed_drop_temporal_ratios} '
            f'temporal_pattern={self.cfg.train_mixed_drop_temporal_pattern} '
            f'seed={self.cfg.train_mixed_drop_seed} '
            f'version={MIXED_MODALITY_DROP_VERSION}'
        )
        if self.cfg.loss_dynamics:
            if self.cfg.loss_type != 'pearson_ranknet':
                raise ValueError(
                    'loss_dynamics requires loss_type=pearson_ranknet'
                )
            if self.cfg.use_genre or getattr(self.cfg, 'ctm_weight', 0.0) > 0.0:
                raise ValueError(
                    'loss_dynamics requires genre and CTM auxiliary losses '
                    'to be disabled'
                )
            if getattr(self.cfg, 'rank_variant', 'pairwise') != 'pairwise':
                raise ValueError(
                    'loss_dynamics only instruments rank_variant=pairwise; '
                    'the gradient-diagnostics recorder is built around the '
                    'pairwise formulation\'s per-pair details'
                )
            self.loss_dynamics_recorder = LossDynamicsRecorder(
                self.cfg.output_dir,
                flush_interval=self.cfg.loss_dynamics_flush_interval,
                metadata={
                    'seed': self.cfg.seed,
                    'loss_type': self.cfg.loss_type,
                    'weights': list(self.combined_weights),
                },
            )
            self.loss_dynamics_probe = ValidationProbe(
                self.cfg,
                self.cfg.output_dir,
                size=self.cfg.loss_dynamics_probe_size,
                batch_size=self.cfg.loss_dynamics_probe_batch_size,
                seed=self.cfg.loss_dynamics_probe_seed,
                flush_interval=self.cfg.loss_dynamics_flush_interval,
            )

    def _compute_mpr_loss(
        self,
        visual,
        text,
        audio,
        gt_score,
        mask,
        batch,
    ):
        output, logits, _ = self.model(visual, text, audio, mask=mask)
        sample_weight = None
        if 'sample_weight' in batch:
            sample_weight = batch['sample_weight'].to(
                self.cfg.device, non_blocking=True
            )
        if sample_weight is not None:
            frame_weight = sample_weight.unsqueeze(1).expand_as(mask)[mask]
            squared_error = (output[mask] - gt_score[mask]) ** 2
            mse_loss = (frame_weight * squared_error).sum() / frame_weight.sum()
        else:
            mse_loss = F.mse_loss(output[mask], gt_score[mask])
        pearson_loss = batched_pearson_loss(
            output, gt_score, mask, weights=sample_weight
        )
        rank_variant = getattr(self.cfg, 'rank_variant', 'pairwise')
        rank_result = compute_rank_loss(
            logits,
            gt_score,
            mask,
            variant=rank_variant,
            num_pairs=self.cfg.num_pairs,
            min_gap=self.cfg.min_gap,
            top_frac=self.cfg.peak_top_frac,
            temperature=getattr(self.cfg, 'rank_temperature', 1.0),
            max_frames=getattr(self.cfg, 'listwise_max_frames', 600),
            error_floor=getattr(self.cfg, 'rank_error_floor', 0.25),
            error_temperature=getattr(
                self.cfg, 'rank_error_temperature', 1.0
            ),
            gt_summary=(
                batch['gt_summary'] if rank_variant == 'approxap' else None
            ),
            generator=self.ranknet_generator,
            return_details=self.loss_dynamics_recorder is not None,
            weights=sample_weight,
        )
        if self.loss_dynamics_recorder is not None:
            rank_loss, rank_details = rank_result
        else:
            rank_loss = rank_result
            rank_details = None
        mse_weight, pearson_weight, rank_weight = self.combined_weights
        loss = (
            mse_weight * mse_loss
            + pearson_weight * pearson_loss
            + rank_weight * rank_loss
        )
        return {
            'loss': loss,
            'output': output,
            'logits': logits,
            'mse': mse_loss,
            'pearson': pearson_loss,
            'rank': rank_loss,
            'rank_details': rank_details,
        }
    
    # Training method for training the model and evaluating on train/val sets
    def train(self):
        best_model_score = -np.inf   # task-specific primary validation metric
        best_map15_score = -np.inf   # secondary: mAP@15
        patience_counter = 0
        start_epoch = 1
        global_step = 0

        # Resume from checkpoint if one exists in the output directory
        resume_ckpt_path = os.path.join(self.cfg.output_dir, 'resume_ckpt.pth')
        if os.path.exists(resume_ckpt_path):
            ckpt = torch.load(resume_ckpt_path, map_location=self.cfg.device, weights_only=False)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.scheduler is not None and ckpt.get('scheduler') is not None:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            best_model_score = ckpt['best_model_score']
            best_map15_score = ckpt.get('best_map15_score', -np.inf)
            patience_counter = ckpt['patience_counter']
            global_step = ckpt.get('global_step', len(self.train_loader) * ckpt['epoch'])
            if (self.ranknet_generator is not None
                    and ckpt.get('ranknet_generator_state') is not None):
                self.ranknet_generator.set_state(
                    ckpt['ranknet_generator_state'].cpu()
                )
            self.logger.info(f"\t# Resumed from epoch {ckpt['epoch']} (best_score={best_model_score:.4f}, patience={patience_counter})")

        # Pre-compute per-class alpha weights for focal cls loss
        if self.cfg.loss_type in ('cls_mse_focal', 'cls_mse_emd'):
            self.cls_alpha = self._compute_cls_alpha(self.cfg.num_cls_bins)

        if (self.loss_dynamics_probe is not None
            and start_epoch == 1
            and not self.loss_dynamics_probe.has_snapshot('initial')):
            self.loss_dynamics_probe.snapshot(
                self.model,
                snapshot_name='initial',
                epoch=0,
                global_step=global_step,
                weights=self.combined_weights,
                num_pairs=self.cfg.num_pairs,
                min_gap=self.cfg.min_gap,
                top_frac=self.cfg.peak_top_frac,
                seed=self.cfg.loss_dynamics_probe_seed,
            )

        if not self.cfg.skip_pretrain_eval and start_epoch == 1:
            train_results = self.evaluate(split='train', epoch=0)
            val_results = self.evaluate(split='val', epoch=0)
            if self.cfg.save_epoch_ckpts:
                epoch_ckpt_dir = os.path.join(self.cfg.output_dir, 'epoch_ckpts')
                os.makedirs(epoch_ckpt_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(epoch_ckpt_dir, 'epoch_000_ckpt.pth'),
                )
            if self.loss_dynamics_recorder is not None:
                self.loss_dynamics_recorder.record_epoch(
                    0,
                    global_step,
                    train_results,
                    val_results,
                    self.combined_weights,
                )
            log_training(self.logger, train_results, val_results, epoch=0)
            if self.cfg.loss_type == 'ranknet':
                self.logger.info(
                    f"\t  RankNet losses  "
                    f"Train: MSE={train_results['mse_loss']:.6f}  Rank={train_results['rank_loss']:.6f}  "
                    f"| Val: MSE={val_results['mse_loss']:.6f}  Rank={val_results['rank_loss']:.6f}"
                )
            elif self.cfg.loss_type == 'pearson_ranknet':
                self._log_combined_losses(train_results, val_results)
            if self.cfg.initialize_best_from_pretrain:
                best_model_score = self._primary_validation_score(val_results)
                best_map15_score = val_results['map15']
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.cfg.output_dir, 'best_model_ckpt.pth'),
                )
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.cfg.output_dir, 'best_map15_ckpt.pth'),
                )
                self.logger.info(
                    '\t# Initialized best checkpoints from epoch 0 '
                    f'(primary={best_model_score:.4f}, '
                    f'mAP@15={best_map15_score:.2f})'
                )
            wandb_training(self.cfg, train_results, val_results, epoch=0)
        
        epoch_bar = tqdm(range(start_epoch, self.cfg.num_epochs + 1), desc='Epochs', leave=True, dynamic_ncols=True)
        for epoch in epoch_bar:
            self.model.train()
            t_epoch_start = time.perf_counter()
            if self.cfg.loss_type == 'pearson_ranknet':
                self.combined_weights = combined_loss_weights(self.cfg, epoch)

            # Compute Pearson weight for staged MSE→Pearson schedule
            pearson_weight = 0.0
            if self.cfg.loss_type == 'mse_pearson':
                ws, we = self.cfg.pearson_warmup_start, self.cfg.pearson_warmup_end
                if epoch <= ws:    pearson_weight = 0.0
                elif epoch >= we:  pearson_weight = 1.0
                else:              pearson_weight = (epoch - ws) / (we - ws)

            # Compute effective genre loss weight (supports linear decay schedule)
            genre_weight = self.cfg.genre_loss_weight
            if (self.cfg.use_genre
                    and self.cfg.genre_loss_start_decay > 0
                    and epoch > self.cfg.genre_loss_start_decay):
                decay_range = self.cfg.genre_loss_end_decay - self.cfg.genre_loss_start_decay
                progress = min(1.0, (epoch - self.cfg.genre_loss_start_decay) / decay_range)
                genre_weight = self.cfg.genre_loss_weight * (1.0 - progress)

            # Iterate over training batches
            ctm_loss_values = []
            ctm_margin_values = []
            ctm_retrieval_1_values = []
            ctm_retrieval_2_values = []
            clean_mpr_loss_values = []
            corrupt_mpr_loss_values = []
            kd_loss_values = []
            frame_drop_ratio_counts = {
                ratio: 0
                for ratio in sorted(set(self.cfg.train_frame_drop_ratios) | {0.0})
            }
            frame_drop_modality_counts = np.zeros(len(MODALITIES), dtype=np.int64)
            frame_drop_valid_count = 0
            frame_drop_union_count = 0
            frame_drop_all_count = 0
            modality_drop_state_counts = {
                'clean': 0,
                'drop_1': 0,
                'drop_2': 0,
                'drop_3': 0,
            }
            modality_drop_combination_counts = {}
            mixed_drop_regime_counts = {'clean': 0, 'temporal': 0, 'whole': 0}
            mixed_drop_ratio_counts = {
                ratio: 0 for ratio in self.cfg.train_mixed_drop_temporal_ratios
            }
            mixed_drop_modality_counts = {modality: 0 for modality in MODALITIES}
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.cfg.num_epochs}', leave=False, dynamic_ncols=True)
            t_batch_end = time.perf_counter()
            for batch in pbar:
                t_data_end = time.perf_counter()
                visual = batch['visual_feat'].to(self.cfg.device, non_blocking=True)
                text = batch['text_feat'].to(self.cfg.device, non_blocking=True)
                audio = batch['audio_feat'].to(self.cfg.device, non_blocking=True)
                gt_score = batch['gt_score'].to(self.cfg.device, non_blocking=True)
                mask = batch['mask'].to(self.cfg.device, non_blocking=True)
                clean_features = {
                    'visual': visual,
                    'text': text,
                    'audio': audio,
                }
                features, frame_drop_mask, sampled_ratios = apply_training_frame_drop(
                    clean_features,
                    mask,
                    batch['video_id'],
                    self.cfg.train_frame_drop_pattern,
                    self.cfg.train_frame_drop_ratios,
                    self.cfg.train_frame_drop_seed,
                    epoch,
                    curriculum=self.cfg.train_frame_drop_curriculum,
                    clean_probability=(
                        self.cfg.train_frame_drop_clean_probability
                        if self.cfg.train_frame_drop_view_mode == 'single'
                        else 0.0
                    ),
                    curriculum_stage1_end=(
                        self.cfg.train_frame_drop_curriculum_stage1_end
                    ),
                    curriculum_stage2_end=(
                        self.cfg.train_frame_drop_curriculum_stage2_end
                    ),
                )
                features, modality_drop_mask, modality_drop_states = (
                    apply_training_whole_modality_drop(
                        features,
                        mask,
                        batch['video_id'],
                        self.cfg.train_modality_drop_policy,
                        self.cfg.train_modality_drop_probability,
                        self.cfg.train_modality_drop_seed,
                        epoch,
                    )
                )
                if self.cfg.train_mixed_drop_policy == 'mixed':
                    features, mixed_drop_mask, mixed_drop_states = (
                        apply_training_mixed_drop(
                            features,
                            mask,
                            batch['video_id'],
                            self.cfg.train_mixed_drop_probabilities,
                            self.cfg.train_mixed_drop_temporal_ratios,
                            self.cfg.train_mixed_drop_seed,
                            epoch,
                            self.cfg.train_mixed_drop_temporal_pattern,
                        )
                    )
                else:
                    mixed_drop_mask = torch.zeros_like(frame_drop_mask)
                    mixed_drop_states = []
                visual = features['visual']
                text = features['text']
                audio = features['audio']
                for ratio in sampled_ratios:
                    frame_drop_ratio_counts[ratio] += 1
                frame_drop_modality_counts += frame_drop_mask.sum(
                    dim=(0, 1)
                ).cpu().numpy()
                frame_drop_valid_count += int(mask.sum().item())
                frame_drop_union_count += int(
                    frame_drop_mask.any(dim=2).sum().item()
                )
                frame_drop_all_count += int(
                    frame_drop_mask.all(dim=2).sum().item()
                )
                for state in modality_drop_states:
                    state_key = 'clean' if not state else f'drop_{len(state)}'
                    modality_drop_state_counts[state_key] += 1
                    combination = 'clean' if not state else '+'.join(state)
                    modality_drop_combination_counts[combination] = (
                        modality_drop_combination_counts.get(combination, 0) + 1
                    )
                for state in mixed_drop_states:
                    mixed_drop_regime_counts[state.regime] += 1
                    if state.regime == 'temporal':
                        mixed_drop_ratio_counts[state.ratio] += 1
                    elif state.regime == 'whole':
                        mixed_drop_modality_counts[state.modality] += 1
                loss_mask = batch.get('trend_mask', batch['mask']).to(self.cfg.device, non_blocking=True)
                if self.cfg.loss_type in ('trend_cls', 'trend_cls_mse'):
                    trend_class = batch['trend_class'].to(self.cfg.device, non_blocking=True)
                elif self.cfg.loss_type == 'salience_cls':
                    salience_class = batch['salience_class'].to(self.cfg.device, non_blocking=True)
                if self.cfg.use_genre:
                    cluster_ids = batch['cluster_id'].to(self.cfg.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=self.cfg.amp):
                    if self.cfg.use_genre:
                        output, genre_logits, _ = self.model(visual, text, audio, mask=mask)
                        mse_loss = self.criterion(output[mask], gt_score[mask])
                        cls_loss = self.criterion_genre(genre_logits, cluster_ids)
                        loss = (1 - genre_weight) * mse_loss + genre_weight * cls_loss
                    elif self.cfg.loss_type == 'ranknet':
                        output, logits, _ = self.model(visual, text, audio, mask=mask)
                        mse_loss = F.mse_loss(output[mask], gt_score[mask])
                        rank_loss = batched_ranknet_loss(
                            logits, gt_score, mask,
                            num_pairs=self.cfg.num_pairs,
                            min_gap=self.cfg.min_gap,
                            top_frac=self.cfg.peak_top_frac,
                        )
                        loss = self.cfg.mse_weight * mse_loss + self.cfg.ranknet_weight * rank_loss
                    elif self.cfg.loss_type == 'pearson':
                        output, _ = self.model(visual, text, audio, mask=mask)
                        loss = batched_pearson_loss(output, gt_score, mask)
                    elif self.cfg.loss_type == 'mse_pearson':
                        output, _ = self.model(visual, text, audio, mask=mask)
                        mse_loss = F.mse_loss(output[mask], gt_score[mask])
                        p_loss   = batched_pearson_loss(output, gt_score, mask)
                        loss     = (1.0 - pearson_weight) * mse_loss + pearson_weight * p_loss
                    elif self.cfg.loss_type == 'pearson_ranknet':
                        corrupt_result = self._compute_mpr_loss(
                            visual,
                            text,
                            audio,
                            gt_score,
                            mask,
                            batch,
                        )
                        output = corrupt_result['output']
                        logits = corrupt_result['logits']
                        mse_loss = corrupt_result['mse']
                        p_loss = corrupt_result['pearson']
                        rank_loss = corrupt_result['rank']
                        rank_details = corrupt_result['rank_details']
                        if self.cfg.train_frame_drop_view_mode == 'dual':
                            clean_result = self._compute_mpr_loss(
                                clean_features['visual'],
                                clean_features['text'],
                                clean_features['audio'],
                                gt_score,
                                mask,
                                batch,
                            )
                            kd_loss = output.new_zeros(())
                            consistency_mask = observable_consistency_mask(
                                mask,
                                frame_drop_mask,
                                modality_drop_mask,
                                mixed_drop_mask,
                            )
                            if self.cfg.train_frame_drop_teacher_mode == 'online':
                                kd_loss = masked_logit_distillation_loss(
                                    logits,
                                    clean_result['logits'].detach(),
                                    consistency_mask,
                                )
                            elif self.frame_drop_teacher is not None:
                                with torch.no_grad():
                                    _, teacher_logits, _ = self.frame_drop_teacher(
                                        clean_features['visual'],
                                        clean_features['text'],
                                        clean_features['audio'],
                                        mask=mask,
                                    )
                                kd_loss = masked_logit_distillation_loss(
                                    logits, teacher_logits, consistency_mask
                                )
                            loss = (
                                self.cfg.train_frame_drop_clean_weight
                                * clean_result['loss']
                                + self.cfg.train_frame_drop_corrupt_weight
                                * corrupt_result['loss']
                                + self.cfg.train_frame_drop_kd_weight * kd_loss
                            )
                            clean_mpr_loss_values.append(
                                clean_result['loss'].item()
                            )
                            corrupt_mpr_loss_values.append(
                                corrupt_result['loss'].item()
                            )
                            kd_loss_values.append(kd_loss.item())
                        else:
                            loss = corrupt_result['loss']
                    elif self.cfg.loss_type == 'trend':
                        output, _ = self.model(visual, text, audio, mask=mask)
                        pred_delta = output * 2.0 - 1.0
                        loss = batched_cosine_loss(pred_delta, gt_score, loss_mask)
                    elif self.cfg.loss_type in ('trend_cls', 'trend_cls_mse'):
                        output, cls_logits, _ = self.model(
                            visual, text, audio, mask=mask
                        )
                        cls_loss = F.cross_entropy(
                            cls_logits[loss_mask], trend_class[loss_mask],
                            weight=self.cls_alpha,
                        )
                        if self.cfg.loss_type == 'trend_cls_mse':
                            mse_loss = F.mse_loss(output[mask], gt_score[mask])
                            loss = (
                                self.cfg.cls_loss_weight * cls_loss
                                + (1.0 - self.cfg.cls_loss_weight) * mse_loss
                            )
                        else:
                            loss = cls_loss
                    elif self.cfg.loss_type == 'salience_cls':
                        output, cls_logits, _ = self.model(
                            visual, text, audio, mask=mask
                        )
                        cls_loss = batched_weighted_cross_entropy(
                            cls_logits, salience_class, mask, weight=self.cls_alpha
                        )
                        loss = cls_loss
                    elif self.cfg.loss_type in ('cls_mse_focal', 'cls_mse_emd'):
                        output, cls_logits, _ = self.model(visual, text, audio, mask=mask)
                        mse_loss = F.mse_loss(output[mask], gt_score[mask])
                        if self.cfg.loss_type == 'cls_mse_focal':
                            cls_loss = batched_focal_cls_loss(
                                cls_logits, gt_score, mask,
                                alpha=self.cls_alpha, gamma=2.0,
                                num_bins=self.cfg.num_cls_bins,
                            )
                        else:
                            cls_loss = batched_emd_cls_loss(
                                cls_logits, gt_score, mask,
                                num_bins=self.cfg.num_cls_bins,
                            )
                        loss = (1.0 - self.cfg.cls_loss_weight) * mse_loss + self.cfg.cls_loss_weight * cls_loss
                    else:
                        output, _ = self.model(visual, text, audio, mask=mask)
                        if 'sample_weight' in batch:
                            sample_weight = batch['sample_weight'].to(
                                self.cfg.device, non_blocking=True
                            )
                            frame_weight = sample_weight.unsqueeze(1).expand_as(mask)[mask]
                            squared_error = (output[mask] - gt_score[mask]) ** 2
                            loss = (frame_weight * squared_error).sum() / frame_weight.sum()
                        else:
                            loss = self.criterion(output[mask], gt_score[mask])

                    ctm_result = compute_optional_ctm(
                        self.model,
                        self.cfg,
                        epoch,
                        visual,
                        text,
                        audio,
                        mask,
                    )
                    if ctm_result is not None:
                        ctm_loss, ctm_coefficient, ctm_diagnostics = ctm_result
                        loss = loss + ctm_coefficient * ctm_loss

                self.optimizer.zero_grad()
                if self.loss_dynamics_recorder is not None:
                    self.loss_dynamics_recorder.measure_step(
                        losses={
                            'mse': mse_loss,
                            'pearson': p_loss,
                            'ranknet': rank_loss,
                        },
                        output=output,
                        logits=logits,
                        gt_score=gt_score,
                        mask=mask,
                        model=self.model,
                        weights=self.combined_weights,
                        rank_details=rank_details,
                        epoch=epoch,
                        global_step=global_step,
                        learning_rate=self.optimizer.param_groups[0]['lr'],
                        amp_scale=self.scaler.get_scale(),
                    )
                self.scaler.scale(loss).backward()
                if self.loss_dynamics_recorder is not None:
                    self.scaler.unscale_(self.optimizer)
                    self.loss_dynamics_recorder.finish_step()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if (self.loss_dynamics_probe is not None
                        and global_step % self.cfg.loss_dynamics_probe_interval == 0):
                    self.loss_dynamics_probe.evaluate(
                        self.model, epoch=epoch, global_step=global_step
                    )
                global_step += 1
                # Step-based scheduler (cosine_warmup) is called per optimizer step
                if self.scheduler is not None and self.cfg.scheduler == 'cosine_warmup':
                    self.scheduler.step()

                loss_value = loss.item()
                t_step_end = time.perf_counter()
                postfix = {
                    'loss': f'{loss_value:.4f}',
                    'seq': batch['visual_feat'].shape[1],
                    'data_ms': f'{(t_data_end - t_batch_end)*1000:.0f}',
                    'step_ms': f'{(t_step_end - t_data_end)*1000:.0f}',
                }
                if self.cfg.use_genre:
                    postfix['mse'] = f'{mse_loss.item():.4f}'
                    postfix['cls'] = f'{cls_loss.item():.4f}'
                elif self.cfg.loss_type == 'ranknet':
                    postfix['mse'] = f'{mse_loss.item():.4f}'
                    postfix['rank'] = f'{rank_loss.item():.4f}'
                elif self.cfg.loss_type == 'mse_pearson':
                    postfix['mse'] = f'{mse_loss.item():.4f}'
                    postfix['p_w'] = f'{pearson_weight:.2f}'
                elif self.cfg.loss_type == 'pearson_ranknet':
                    mse_weight, pearson_weight_active, rank_weight = (
                        self.combined_weights
                    )
                    postfix['mse'] = f'{mse_loss.item():.4f}'
                    postfix['pearson'] = f'{p_loss.item():.4f}'
                    postfix['rank'] = f'{rank_loss.item():.4f}'
                    postfix['weights'] = (
                        f'{mse_weight:.2f}/'
                        f'{pearson_weight_active:.2f}/{rank_weight:.2f}'
                    )
                elif self.cfg.loss_type in ('cls_mse_focal', 'cls_mse_emd'):
                    postfix['mse'] = f'{mse_loss.item():.4f}'
                    postfix['cls'] = f'{cls_loss.item():.4f}'
                elif self.cfg.loss_type in ('trend_cls', 'trend_cls_mse'):
                    postfix['cls'] = f'{cls_loss.item():.4f}'
                    if self.cfg.loss_type == 'trend_cls_mse':
                        postfix['mse'] = f'{mse_loss.item():.4f}'
                elif self.cfg.loss_type == 'salience_cls':
                    postfix['cls'] = f'{cls_loss.item():.4f}'
                if ctm_result is not None:
                    ctm_loss_values.append(ctm_loss.item())
                    ctm_margin_values.append(ctm_diagnostics['similarity_margin'])
                    ctm_retrieval_1_values.append(ctm_diagnostics['retrieval_at_1'])
                    ctm_retrieval_2_values.append(ctm_diagnostics['retrieval_at_2'])
                    postfix['ctm'] = f'{ctm_loss.item():.4f}'
                    postfix['ctm_w'] = f'{ctm_coefficient:.3f}'
                if self.cfg.train_frame_drop_view_mode == 'dual':
                    postfix['clean_mpr'] = f'{clean_result["loss"].item():.4f}'
                    postfix['corrupt_mpr'] = f'{corrupt_result["loss"].item():.4f}'
                    if self.cfg.train_frame_drop_kd_weight > 0.0:
                        postfix['kd'] = f'{kd_loss.item():.4f}'
                pbar.set_postfix(**postfix)
                t_batch_end = time.perf_counter()

            # Epoch-level scheduler step (skipped for step-based cosine_warmup)
            if self.scheduler is not None and self.cfg.scheduler != 'cosine_warmup':
                self.scheduler.step()

            if (self.loss_dynamics_probe is not None
                    and epoch % self.cfg.loss_dynamics_snapshot_interval == 0):
                self.loss_dynamics_probe.snapshot(
                    self.model,
                    snapshot_name=f'epoch_{epoch:03d}',
                    epoch=epoch,
                    global_step=global_step,
                    weights=self.combined_weights,
                    num_pairs=self.cfg.num_pairs,
                    min_gap=self.cfg.min_gap,
                    top_frac=self.cfg.peak_top_frac,
                    seed=self.cfg.loss_dynamics_probe_seed + epoch,
                )

            # Evaluation on train and val sets
            t_train_end = time.perf_counter()
            train_results = self.evaluate(split='train', epoch=epoch)
            val_results = self.evaluate(split='val', epoch=epoch)
            if self.loss_dynamics_recorder is not None:
                self.loss_dynamics_recorder.record_epoch(
                    epoch,
                    global_step,
                    train_results,
                    val_results,
                    self.combined_weights,
                )
            log_training(self.logger, train_results, val_results, epoch)
            if self.cfg.use_genre:
                self.logger.info(
                    f"\t  Genre losses  "
                    f"Train: MSE={train_results['mse_loss']:.6f}  CE={train_results['cls_loss']:.6f}  "
                    f"| Val: MSE={val_results['mse_loss']:.6f}  CE={val_results['cls_loss']:.6f}  "
                    f"| genre_weight={genre_weight:.4f}"
                )
            elif self.cfg.loss_type == 'ranknet':
                self.logger.info(
                    f"\t  RankNet losses  "
                    f"Train: MSE={train_results['mse_loss']:.6f}  Rank={train_results['rank_loss']:.6f}  "
                    f"| Val: MSE={val_results['mse_loss']:.6f}  Rank={val_results['rank_loss']:.6f}"
                )
            elif self.cfg.loss_type == 'pearson_ranknet':
                self._log_combined_losses(train_results, val_results)
            elif self.cfg.loss_type in ('cls_mse_focal', 'cls_mse_emd'):
                self.logger.info(
                    f"\t  ClsMSE losses  "
                    f"Train: MSE={train_results['mse_loss']:.6f}  CLS={train_results['cls_loss']:.6f}  "
                    f"| Val: MSE={val_results['mse_loss']:.6f}  CLS={val_results['cls_loss']:.6f}"
                )
            elif self.cfg.loss_type in ('pearson', 'mse_pearson'):
                msg = (
                    f"\t  Pearson losses  "
                    f"Train: Pearson={train_results.get('pearson_loss', 0):.6f}  "
                    f"| Val: Pearson={val_results.get('pearson_loss', 0):.6f}"
                )
                if self.cfg.loss_type == 'mse_pearson':
                    msg += f"  | pearson_weight={pearson_weight:.4f}"
                    if 'mse_loss' in train_results:
                        msg += (f"  Train MSE={train_results['mse_loss']:.6f}"
                                f"  Val MSE={val_results['mse_loss']:.6f}")
                self.logger.info(msg)
            elif self.cfg.loss_type == 'trend':
                self.logger.info(
                    f"\t  Trend cosine loss  Train: {train_results['cosine_loss']:.6f}  "
                    f"| Val: {val_results['cosine_loss']:.6f}"
                )
            elif self.cfg.loss_type in ('trend_cls', 'trend_cls_mse'):
                msg = (
                    f"\t  Trend classification  "
                    f"Train CE={train_results['cls_loss']:.6f}  "
                    f"Val CE={val_results['cls_loss']:.6f}"
                )
                if self.cfg.loss_type == 'trend_cls_mse':
                    msg += (
                        f"  Train MSE={train_results['mse_loss']:.6f}"
                        f"  Val MSE={val_results['mse_loss']:.6f}"
                    )
                self.logger.info(msg)
            elif self.cfg.loss_type == 'salience_cls':
                self.logger.info(
                    f"\t  Salience classification  "
                    f"Train CE={train_results['cls_loss']:.6f}  "
                    f"Val CE={val_results['cls_loss']:.6f}"
                )
            if ctm_loss_values:
                self.logger.info(
                    f"\t  CTM train diagnostics  "
                    f"loss={np.mean(ctm_loss_values):.6f}  "
                    f"weight={ctm_coefficient:.4f}  "
                    f"margin={np.mean(ctm_margin_values):.4f}  "
                    f"retrieval@1={np.mean(ctm_retrieval_1_values):.4f}  "
                    f"retrieval@2={np.mean(ctm_retrieval_2_values):.4f}"
                )
            if self.cfg.train_frame_drop_pattern != 'none':
                denominator = max(1, frame_drop_valid_count)
                active_ratios = (
                    curriculum_training_ratios(
                        self.cfg.train_frame_drop_ratios,
                        epoch,
                        self.cfg.train_frame_drop_curriculum_stage1_end,
                        self.cfg.train_frame_drop_curriculum_stage2_end,
                    )
                    if self.cfg.train_frame_drop_curriculum
                    else tuple(self.cfg.train_frame_drop_ratios)
                )
                modality_fractions = {
                    modality: float(count / denominator)
                    for modality, count in zip(
                        MODALITIES, frame_drop_modality_counts
                    )
                }
                self.logger.info(
                    f'\t  Frame drop epoch {epoch}: '
                    f'active_ratios={active_ratios} '
                    f'ratio_videos={frame_drop_ratio_counts} '
                    f'modality_fractions={modality_fractions} '
                    f'union_fraction={frame_drop_union_count / denominator:.4f} '
                    f'all_fraction={frame_drop_all_count / denominator:.4f}'
                )
            if self.cfg.train_modality_drop_policy != 'none':
                self.logger.info(
                    f'\t  Whole-modality drop epoch {epoch}: '
                    f'states={modality_drop_state_counts} '
                    f'combinations={modality_drop_combination_counts}'
                )
            if self.cfg.train_mixed_drop_policy == 'mixed':
                self.logger.info(
                    f'\t  Mixed drop epoch {epoch}: '
                    f'regimes={mixed_drop_regime_counts} '
                    f'temporal_ratios={mixed_drop_ratio_counts} '
                    f'whole_modalities={mixed_drop_modality_counts}'
                )
            if clean_mpr_loss_values:
                self.logger.info(
                    f'\t  Dual-view train epoch {epoch}: '
                    f'clean_mpr={np.mean(clean_mpr_loss_values):.6f} '
                    f'corrupt_mpr={np.mean(corrupt_mpr_loss_values):.6f} '
                    f'kd={np.mean(kd_loss_values):.6f}'
                )
            epoch_time = time.perf_counter() - t_epoch_start
            self.logger.info(f"\t# Epoch {epoch} wall time: train={t_train_end-t_epoch_start:.1f}s  eval={time.perf_counter()-t_train_end:.1f}s")
            wandb_training(self.cfg, train_results, val_results, epoch)

            # Update outer progress bar with key metrics
            if self.cfg.loss_type in CLASSIFICATION_LOSS_TYPES:
                primary_metric = (
                    val_results['video_macro_f1']
                    if self.cfg.loss_type == 'salience_cls'
                    else val_results['macro_f1']
                )
                epoch_bar.set_postfix(
                    val_macro_f1=f"{primary_metric:.3f}",
                    val_acc=f"{val_results.get('video_accuracy', val_results.get('accuracy')):.3f}",
                    loss=f"{train_results['loss']:.4f}",
                    t=f"{epoch_time:.0f}s",
                    best=f"{best_model_score:.3f}" if best_model_score > -np.inf else "—",
                )
            else:
                epoch_bar.set_postfix(
                    val_kTau=f"{val_results['ktau']:.3f}",
                    val_sRho=f"{val_results['srho']:.3f}",
                    loss=f"{train_results['loss']:.4f}",
                    t=f"{epoch_time:.0f}s",
                    best=f"{best_model_score:.3f}" if best_model_score > -np.inf else "—",
                )

            # Best model checkpointing (task-specific primary metric)
            model_score = self._primary_validation_score(val_results)
            model_improved = model_score > best_model_score
            if model_improved:
                best_model_score = model_score
                best_model_ckpt = os.path.join(self.cfg.output_dir, 'best_model_ckpt.pth')
                torch.save(self.model.state_dict(), best_model_ckpt)
                self.logger.info(f"\t# New best model at epoch {epoch}")

            # Secondary checkpoint: best mAP@15
            map15_improved = (
                self.cfg.loss_type not in CLASSIFICATION_LOSS_TYPES
                and val_results['map15'] > best_map15_score
            )
            if map15_improved:
                best_map15_score = val_results['map15']
                torch.save(self.model.state_dict(),
                           os.path.join(self.cfg.output_dir, 'best_map15_ckpt.pth'))
                self.logger.info(f"\t# New best mAP@15 model at epoch {epoch}  (mAP@15={best_map15_score:.2f})")

            stopping_improved = (
                map15_improved
                if self.cfg.early_stop_metric == 'map15'
                else model_improved
            )
            patience_counter = 0 if stopping_improved else patience_counter + 1

            # Save every epoch's weights, uncompressed history (opt-in)
            if self.cfg.save_epoch_ckpts:
                epoch_ckpt_dir = os.path.join(self.cfg.output_dir, 'epoch_ckpts')
                os.makedirs(epoch_ckpt_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(epoch_ckpt_dir, f'epoch_{epoch:03d}_ckpt.pth'),
                )

            # Save full resume checkpoint every epoch (overwrites previous)
            torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                'best_model_score': best_model_score,
                'best_map15_score': best_map15_score,
                'patience_counter': patience_counter,
                'global_step': global_step,
                'ranknet_generator_state': (
                    self.ranknet_generator.get_state()
                    if self.ranknet_generator is not None else None
                ),
            }, resume_ckpt_path)

            # Early stopping
            if patience_counter >= self.cfg.patience:
                self.logger.info(f"\t# Early stopping at epoch {epoch}")
                break

        if self.cfg.loss_type in CLASSIFICATION_LOSS_TYPES:
            best_model_ckpt = os.path.join(
                self.cfg.output_dir, 'best_model_ckpt.pth'
            )
            self.model.load_state_dict(torch.load(
                best_model_ckpt,
                map_location=self.cfg.device,
                weights_only=True,
            ))
        if self.loss_dynamics_recorder is not None:
            self.loss_dynamics_recorder.close()
        if self.loss_dynamics_probe is not None:
            self.loss_dynamics_probe.close()

    def _log_combined_losses(self, train_results, val_results):
        mse_weight, pearson_weight, rank_weight = self.combined_weights
        self.logger.info(
            f"\t  Combined losses  "
            f"Train: MSE={train_results['mse_loss']:.6f}  "
            f"Pearson={train_results['pearson_loss']:.6f}  "
            f"Rank={train_results['rank_loss']:.6f}  "
            f"| Val: MSE={val_results['mse_loss']:.6f}  "
            f"Pearson={val_results['pearson_loss']:.6f}  "
            f"Rank={val_results['rank_loss']:.6f}  "
            f"| weights={mse_weight:.3f}/{pearson_weight:.3f}/{rank_weight:.3f}"
        )

    def _primary_validation_score(self, val_results):
        if self.cfg.loss_type == 'salience_cls':
            return val_results['video_macro_f1']
        if self.cfg.loss_type in ('trend_cls', 'trend_cls_mse'):
            return val_results['macro_f1']
        return val_results['ktau'] + val_results['srho']

    def _compute_trend_class_alpha(self):
        """Return inverse-frequency weights from valid training trend labels."""
        label_path = os.path.join(
            self.cfg.data_dir,
            self.cfg.dataset,
            self.cfg.trend_class_label_file,
        )
        with h5py.File(label_path, 'r') as label_file:
            split_counts = json.loads(label_file.attrs['split_class_counts'])
        counts = torch.tensor(split_counts['train'], dtype=torch.float32)
        alpha = 1.0 / counts.clamp_min(1.0)
        alpha = alpha / alpha.mean()
        self.logger.info(
            f"\t  Trend class counts: {counts.long().tolist()}  "
            f"alpha: {[f'{value:.3f}' for value in alpha.tolist()]}"
        )
        return alpha.to(self.cfg.device)

    def _compute_cls_alpha(self, num_bins):
        """
        Scan training gt_scores and return inverse-frequency alpha weights
        for the ordinal classification loss (normalised so mean = 1).
        Only reads the lightweight gt HDF5 file — does NOT load model features.
        """
        import json as _json, h5py as _h5py
        gt_path    = os.path.join(self.cfg.data_dir, 'mosu', 'mosu_gt.h5')
        split_path = os.path.join(self.cfg.data_dir, 'mosu', 'mosu_split.json')
        with open(split_path) as fp:
            train_ids = _json.load(fp)['train_keys']
        counts = torch.zeros(num_bins)
        with _h5py.File(gt_path, 'r') as f:
            for vid in tqdm(train_ids, desc='Computing cls alpha', leave=False):
                try:
                    gt   = torch.from_numpy(f[vid]['gt_score'][...])
                    bins = (gt * num_bins).long().clamp(0, num_bins - 1)
                    for b in range(num_bins):
                        counts[b] += (bins == b).sum()
                except (KeyError, RuntimeError):
                    continue
        alpha = 1.0 / (counts + 1.0)
        alpha = alpha / alpha.mean()   # normalise so mean weight = 1
        self.logger.info(f"\t  Cls alpha: {[f'{a:.3f}' for a in alpha.tolist()]}")
        return alpha.to(self.cfg.device)

    # Evaluation method for evaluating the model on train/val/test sets
    def evaluate(self, split='val', epoch=None):
        self.model.eval()

        if split == 'train':
            loader = list(itertools.islice(self.train_loader, len(self.val_loader)))
        elif split == 'val':
            loader = self.val_loader
        elif split == 'test':
            loader = self.test_loader

        ktau_list, srho_list = [], []
        map50_list, map15_list = [], []
        metric_weight_list = []
        loss_list = []
        per_video_metrics = {} if self.cfg.save_per_video_metrics else None
        mse_loss_list, cls_loss_list, rank_loss_list, pearson_loss_list, cosine_loss_list = [], [], [], [], []
        genre_correct, genre_total = 0, 0
        trend_confusion = torch.zeros((3, 3), dtype=torch.int64)
        salience_confusions = []
        classification_loss_weights = []
        
        h5_file = None
        if self.cfg.get_attn_weights:
            attn_weights_path = os.path.join(os.path.dirname(self.cfg.model_ckpt), f'{split}_attn_weights.h5')
            h5_file = h5py.File(attn_weights_path, 'w')

        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Evaluating {split} set', leave=False):
                video_ids = batch['video_id']
                
                visual = batch['visual_feat'].to(self.cfg.device, non_blocking=True)
                text = batch['text_feat'].to(self.cfg.device, non_blocking=True)
                audio = batch['audio_feat'].to(self.cfg.device, non_blocking=True)
                
                gt_score = batch['gt_score'].to(self.cfg.device, non_blocking=True)
                mask = batch['mask'].to(self.cfg.device, non_blocking=True)
                loss_mask = batch.get('trend_mask', batch['mask']).to(self.cfg.device, non_blocking=True)
                if self.cfg.loss_type in ('trend_cls', 'trend_cls_mse'):
                    trend_class = batch['trend_class'].to(self.cfg.device, non_blocking=True)
                elif self.cfg.loss_type == 'salience_cls':
                    salience_class = batch['salience_class'].to(self.cfg.device, non_blocking=True)
                if self.cfg.use_genre:
                    cluster_ids = batch['cluster_id'].to(self.cfg.device, non_blocking=True)

                if self.cfg.use_genre:
                    output, genre_logits, attn_weights = self.model(visual, text, audio, mask=mask)
                    mse_loss = self.criterion(output[mask], gt_score[mask])
                    cls_loss = self.criterion_genre(genre_logits, cluster_ids)
                    loss = (1 - self.cfg.genre_loss_weight) * mse_loss + self.cfg.genre_loss_weight * cls_loss
                    preds = genre_logits.argmax(dim=1)
                    genre_correct += (preds == cluster_ids).sum().item()
                    genre_total += cluster_ids.size(0)
                    mse_loss_list.append(mse_loss.item())
                    cls_loss_list.append(cls_loss.item())
                elif self.cfg.loss_type == 'ranknet':
                    output, logits, attn_weights = self.model(visual, text, audio, mask=mask)
                    mse_loss_val = F.mse_loss(output[mask], gt_score[mask])
                    rank_loss_val = batched_ranknet_loss(
                        logits, gt_score, mask,
                        num_pairs=self.cfg.num_pairs,
                        min_gap=self.cfg.min_gap,
                        top_frac=self.cfg.peak_top_frac,
                    )
                    loss = self.cfg.mse_weight * mse_loss_val + self.cfg.ranknet_weight * rank_loss_val
                    mse_loss_list.append(mse_loss_val.item())
                    rank_loss_list.append(rank_loss_val.item())
                elif self.cfg.loss_type == 'pearson':
                    output, attn_weights = self.model(visual, text, audio, mask=mask)
                    p_val = batched_pearson_loss(output, gt_score, mask)
                    loss  = p_val
                    pearson_loss_list.append(p_val.item())
                elif self.cfg.loss_type == 'mse_pearson':
                    output, attn_weights = self.model(visual, text, audio, mask=mask)
                    mse_val = F.mse_loss(output[mask], gt_score[mask])
                    p_val   = batched_pearson_loss(output, gt_score, mask)
                    loss    = mse_val + p_val
                    mse_loss_list.append(mse_val.item())
                    pearson_loss_list.append(p_val.item())
                elif self.cfg.loss_type == 'pearson_ranknet':
                    output, logits, attn_weights = self.model(
                        visual, text, audio, mask=mask
                    )
                    mse_val = F.mse_loss(output[mask], gt_score[mask])
                    p_val = batched_pearson_loss(output, gt_score, mask)
                    rank_variant = getattr(self.cfg, 'rank_variant', 'pairwise')
                    rank_loss_val = compute_rank_loss(
                        logits, gt_score, mask,
                        variant=rank_variant,
                        num_pairs=self.cfg.num_pairs,
                        min_gap=self.cfg.min_gap,
                        top_frac=self.cfg.peak_top_frac,
                        temperature=getattr(self.cfg, 'rank_temperature', 1.0),
                        max_frames=getattr(self.cfg, 'listwise_max_frames', 600),
                        gt_summary=(
                            batch['gt_summary']
                            if rank_variant == 'approxap' else None
                        ),
                    )
                    mse_weight, pearson_weight, rank_weight = (
                        self.combined_weights
                    )
                    loss = (
                        mse_weight * mse_val
                        + pearson_weight * p_val
                        + rank_weight * rank_loss_val
                    )
                    mse_loss_list.append(mse_val.item())
                    pearson_loss_list.append(p_val.item())
                    rank_loss_list.append(rank_loss_val.item())
                elif self.cfg.loss_type == 'trend':
                    output, attn_weights = self.model(visual, text, audio, mask=mask)
                    pred_delta = output * 2.0 - 1.0
                    loss = batched_cosine_loss(pred_delta, gt_score, loss_mask)
                    cosine_loss_list.append(loss.item())
                elif self.cfg.loss_type in ('trend_cls', 'trend_cls_mse'):
                    output, cls_logits_val, attn_weights = self.model(
                        visual, text, audio, mask=mask
                    )
                    cls_val = F.cross_entropy(
                        cls_logits_val[loss_mask], trend_class[loss_mask],
                        weight=self.cls_alpha,
                    )
                    cls_loss_list.append(cls_val.item())
                    if self.cfg.loss_type == 'trend_cls_mse':
                        mse_val = F.mse_loss(output[mask], gt_score[mask])
                        loss = (
                            self.cfg.cls_loss_weight * cls_val
                            + (1.0 - self.cfg.cls_loss_weight) * mse_val
                        )
                        mse_loss_list.append(mse_val.item())
                    else:
                        loss = cls_val
                    predictions = cls_logits_val.argmax(dim=-1)
                    encoded = (
                        trend_class[loss_mask] * 3 + predictions[loss_mask]
                    )
                    trend_confusion += torch.bincount(
                        encoded, minlength=9
                    ).reshape(3, 3).cpu()
                elif self.cfg.loss_type == 'salience_cls':
                    output, cls_logits_val, attn_weights = self.model(
                        visual, text, audio, mask=mask
                    )
                    cls_val = batched_weighted_cross_entropy(
                        cls_logits_val, salience_class, mask, weight=self.cls_alpha
                    )
                    loss = cls_val
                    cls_loss_list.append(cls_val.item())
                    classification_loss_weights.append(len(video_ids))
                    predictions = cls_logits_val.argmax(dim=-1)
                    for sample_idx in range(len(video_ids)):
                        valid = mask[sample_idx]
                        encoded = (
                            salience_class[sample_idx][valid] * 4
                            + predictions[sample_idx][valid]
                        )
                        salience_confusions.append(torch.bincount(
                            encoded, minlength=16
                        ).reshape(4, 4).cpu().numpy())
                elif self.cfg.loss_type in ('cls_mse_focal', 'cls_mse_emd'):
                    output, cls_logits_val, attn_weights = self.model(visual, text, audio, mask=mask)
                    mse_val = F.mse_loss(output[mask], gt_score[mask])
                    if self.cfg.loss_type == 'cls_mse_focal':
                        cls_val = batched_focal_cls_loss(
                            cls_logits_val, gt_score, mask,
                            alpha=self.cls_alpha, gamma=2.0,
                            num_bins=self.cfg.num_cls_bins,
                        )
                    else:
                        cls_val = batched_emd_cls_loss(
                            cls_logits_val, gt_score, mask,
                            num_bins=self.cfg.num_cls_bins,
                        )
                    loss = (1.0 - self.cfg.cls_loss_weight) * mse_val + self.cfg.cls_loss_weight * cls_val
                    mse_loss_list.append(mse_val.item())
                    cls_loss_list.append(cls_val.item())
                else:
                    output, attn_weights = self.model(visual, text, audio, mask=mask)
                    loss = self.criterion(output[mask], gt_score[mask])
                
                if output is not None and output.dim() == 3:
                    output = output.squeeze(-1)

                if self.cfg.loss_type not in CLASSIFICATION_LOSS_TYPES:
                    metric_gt_score = batch.get('eval_gt_score', batch['gt_score'])
                    if self.cfg.loss_type == 'trend':
                        output = reconstruct_trend(output * 2.0 - 1.0, mask)

                    pred_score = output.detach().cpu().numpy().tolist()
                    metric_gt_score = metric_gt_score.detach().cpu().numpy().tolist()
                    metric_mask = mask.detach().cpu().numpy()

                    if per_video_metrics is not None:
                        ktau, srho, ktau_per_video, srho_per_video = evaluate_summary(
                            pred_score, metric_gt_score, metric_mask, return_per_video=True
                        )
                        map50, map15, map50_per_video, map15_per_video = evaluate_highlight(
                            pred_score, metric_gt_score, metric_mask, return_per_video=True
                        )
                        for i, vid in enumerate(video_ids):
                            per_video_metrics[vid] = {
                                'ktau': float(ktau_per_video[i]),
                                'srho': float(srho_per_video[i]),
                                'map50': float(map50_per_video[i]) * 100,
                                'map15': float(map15_per_video[i]) * 100,
                                'n_frames': len(pred_score[i]),
                            }
                    else:
                        ktau, srho = evaluate_summary(
                            pred_score, metric_gt_score, metric_mask
                        )
                        map50, map15 = evaluate_highlight(
                            pred_score, metric_gt_score, metric_mask
                        )

                    ktau_list.append(ktau)
                    srho_list.append(srho)
                    map50_list.append(map50)
                    map15_list.append(map15)
                    metric_weight_list.append(len(video_ids))
                loss_list.append(loss.item())
                
                if self.cfg.get_attn_weights:
                    batch_size = visual.size(0)
                    for i in range(batch_size):
                        video_id = video_ids[i]
                        if video_id not in h5_file:
                            video_group = h5_file.create_group(video_id)
                            for layer_idx, layer_weights_tensor in enumerate(attn_weights):
                                sample_layer_weights = layer_weights_tensor[i].detach().cpu().numpy()
                                video_group.create_dataset(f'layer_{layer_idx}', data=sample_layer_weights)
        
        if self.cfg.get_attn_weights:
            h5_file.close()
        
        if self.cfg.loss_type in ('trend_cls', 'trend_cls_mse'):
            results = classification_metrics(trend_confusion.numpy())
            results['loss'] = np.mean(loss_list)
            results['cls_loss'] = np.mean(cls_loss_list)
            if self.cfg.loss_type == 'trend_cls_mse':
                results['mse_loss'] = np.mean(mse_loss_list)
            return results

        if self.cfg.loss_type == 'salience_cls':
            results = video_classification_metrics(
                salience_confusions, SALIENCE_CLASS_NAMES
            )
            results['loss'] = np.average(
                cls_loss_list, weights=classification_loss_weights
            )
            results['cls_loss'] = results['loss']
            return results

        results = {
            'ktau': np.average(ktau_list, weights=metric_weight_list),
            'srho': np.average(srho_list, weights=metric_weight_list),
            'map50': np.average(map50_list, weights=metric_weight_list),
            'map15': np.average(map15_list, weights=metric_weight_list),
            'loss': np.mean(loss_list),
        }
        if self.cfg.use_genre:
            results['genre_acc'] = genre_correct / genre_total if genre_total > 0 else 0.0
            results['mse_loss'] = np.mean(mse_loss_list)
            results['cls_loss'] = np.mean(cls_loss_list)
        if self.cfg.loss_type == 'ranknet':
            results['mse_loss'] = np.mean(mse_loss_list)
            results['rank_loss'] = np.mean(rank_loss_list)
        if self.cfg.loss_type in ('pearson', 'mse_pearson') and pearson_loss_list:
            results['pearson_loss'] = np.mean(pearson_loss_list)
        if self.cfg.loss_type == 'mse_pearson' and mse_loss_list:
            results['mse_loss'] = np.mean(mse_loss_list)
        if self.cfg.loss_type == 'pearson_ranknet':
            results['mse_loss'] = np.mean(mse_loss_list)
            results['pearson_loss'] = np.mean(pearson_loss_list)
            results['rank_loss'] = np.mean(rank_loss_list)
        if self.cfg.loss_type == 'trend':
            results['cosine_loss'] = np.mean(cosine_loss_list)
        if self.cfg.loss_type in ('cls_mse_focal', 'cls_mse_emd'):
            results['mse_loss'] = np.mean(mse_loss_list)
            results['cls_loss'] = np.mean(cls_loss_list)
        if per_video_metrics is not None:
            record = {'epoch': epoch, 'split': split, 'per_video': per_video_metrics}
            path = os.path.join(self.cfg.output_dir, 'per_video_metrics.jsonl')
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        return results