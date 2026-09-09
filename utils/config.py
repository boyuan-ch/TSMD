import os
import yaml
import argparse

from utils.frame_drop import TRAIN_FRAME_DROP_PATTERNS, validate_frame_drop_config
from utils.whole_modality_drop import (
    WHOLE_MODALITY_DROP_POLICIES,
    validate_drop_policy_compatibility,
    validate_whole_modality_drop_config,
)
from utils.mixed_modality_drop import (
    MIXED_DROP_POLICIES,
    MIXED_DROP_TEMPORAL_PATTERNS,
    validate_mixed_drop_compatibility,
    validate_mixed_drop_config,
    validate_mixed_drop_temporal_pattern,
)

# Function to parse configuration
def get_config():
    parser = argparse.ArgumentParser(description="Configuration Parser")
    
    # General settings
    parser.add_argument('--exp_name', type=str, default='exp', help='Name of the experiment')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode of operation')
    parser.add_argument('--cfg', type=str, default=None, help='Config file name override (loads configs/{cfg}.yaml instead of configs/{dataset}.yaml)')
    
    # Data settings
    parser.add_argument('--dataset', type=str, default='mosu', choices=['mosu', 'mrhisum', 'summe', 'tvsum'], help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory where datasets are stored')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4], help='5-fold split index for summe/tvsum (3 train/1 val/1 test rotation); unused for mosu/mrhisum')
    parser.add_argument('--split_protocol', type=str, default='tvt', choices=['tvt', 'tv'], help='SumMe/TVSum five-fold protocol; TVT has a separate test fold, TV reuses its held-out validation fold for evaluation')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--input_modalities', nargs='+', default=['visual', 'text', 'audio'], choices=['visual', 'text', 'audio'], help='Fixed modalities retained during both training and evaluation')
    parser.add_argument('--salience_smoothing_window', type=int, default=1, help='Odd running-average window for training salience labels; 1 disables smoothing')
    
    # Model settings
    parser.add_argument('--model', type=str, default='triplesumm', help='Model architecture to use')
    parser.add_argument('--visual_dim', type=int, default=768, help='Dimension of visual features')
    parser.add_argument('--text_dim', type=int, default=768, help='Dimension of text features')
    parser.add_argument('--audio_dim', type=int, default=768, help='Dimension of audio features')
    parser.add_argument('--input_dim', type=int, default=512, help='Dimension of input features')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of hidden layers')
    parser.add_argument('--num_model_layers', type=int, default=2, help='Number of model layers')
    parser.add_argument('--num_mst_layers', type=int, default=2, help='Number of multi-scale temporal layers per model layer')
    parser.add_argument('--num_cmf_layers', type=int, default=2, help='Number of cross-modal fusion layers per model layer')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--window_size', type=int, nargs='+', default=[5,15,45,0], help='Window sizes for multi-scale temporal blocks')
    parser.add_argument('--max_seq_len', type=int, default=10000, help='Maximum sequence length')
    parser.add_argument('--get_attn_weights', action='store_true', help='Whether to return attention weights from the model')
    parser.add_argument('--use_genre', action='store_true', help='Enable genre classification multitask learning')
    parser.add_argument('--genre_loss_weight', type=float, default=0.2, help='Weight for genre classification loss (1-weight goes to MSE)')
    parser.add_argument('--genre_loss_start_decay', type=int, default=0, help='Epoch after which genre loss weight linearly decays (0 = no decay)')
    parser.add_argument('--genre_loss_end_decay', type=int, default=0, help='Epoch at which genre loss weight reaches 0')
    parser.add_argument('--num_genre_classes', type=int, default=10, help='Number of classes for the genre/mode auxiliary classification head')
    parser.add_argument('--genre_label_source', type=str, default='metadata', choices=['metadata', 'mode'], help='"metadata" reads cluster_id from mosu_metadata.csv (semantic genre); "mode" reads mode_cluster from mosu_mode_clusters.csv (data-driven gt_score behavior cluster)')
    
    # Training settings
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Learning rate scheduler to use')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for cosine scheduler with warmup')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--early_stop_metric', type=str, default='primary', choices=['primary', 'map15'], help='Validation metric used to reset early-stopping patience')
    parser.add_argument('--initialize_best_from_pretrain', action='store_true', help='Save epoch-0 pre-trained weights as initial best A/B checkpoints')
    
    # Miscellaneous settings
    parser.add_argument('--model_ckpt', type=str, default=None, help='Path to a pre-trained model checkpoint')
    parser.add_argument('--results_json', type=str, default=None, help='Optional path for structured test metrics')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--wandb', action='store_true', help='Whether to use Weights & Biases for experiment tracking')
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision (FP16) training')
    parser.add_argument('--skip_pretrain_eval', action='store_true', help='Skip the epoch-0 evaluation before training starts')
    parser.add_argument('--train_frame_drop_pattern', type=str, default='none', choices=TRAIN_FRAME_DROP_PATTERNS, help='Training-only raw-feature frame-drop pattern')
    parser.add_argument('--train_frame_drop_ratios', type=float, nargs='+', default=[0.0], help='Per-video frame-drop ratios sampled during training')
    parser.add_argument('--train_frame_drop_seed', type=int, default=42, help='Dedicated seed for deterministic training frame dropping')
    parser.add_argument('--train_frame_drop_curriculum', action='store_true', help='Progressively enable frame-drop ratios up to 0.1, 0.3, then all configured ratios')
    parser.add_argument('--train_frame_drop_curriculum_stage1_end', type=int, default=3, help='Last epoch of the ratio-at-most-0.1 curriculum stage')
    parser.add_argument('--train_frame_drop_curriculum_stage2_end', type=int, default=8, help='Last epoch of the ratio-at-most-0.3 curriculum stage')
    parser.add_argument('--train_frame_drop_clean_probability', type=float, default=0.0, help='Deterministic probability of a clean sample in single-view curriculum training')
    parser.add_argument('--train_frame_drop_view_mode', type=str, default='single', choices=['single', 'dual'], help='Use one sampled view or paired clean/corrupt supervised views')
    parser.add_argument('--train_frame_drop_clean_weight', type=float, default=0.5, help='Clean MPR loss coefficient in dual-view mode')
    parser.add_argument('--train_frame_drop_corrupt_weight', type=float, default=0.5, help='Corrupt MPR loss coefficient in dual-view mode')
    parser.add_argument('--train_frame_drop_kd_weight', type=float, default=0.0, help='Frozen clean-teacher logit MSE coefficient in dual-view mode')
    parser.add_argument('--train_frame_drop_teacher_ckpt', type=str, default=None, help='Clean MPR checkpoint used by the frozen teacher')
    parser.add_argument('--train_frame_drop_teacher_mode', type=str, default='frozen', choices=['frozen', 'online'], help='Use a frozen checkpoint or the detached online clean branch as the consistency teacher')
    parser.add_argument('--train_modality_drop_policy', type=str, default='none', choices=WHOLE_MODALITY_DROP_POLICIES, help='Training-only whole-modality drop policy')
    parser.add_argument('--train_modality_drop_probability', type=float, default=0.0, help='Per-modality drop probability for Bernoulli ModDrop; categorical sampling is uniform over clean/V/T/A')
    parser.add_argument('--train_modality_drop_seed', type=int, default=42, help='Dedicated seed for deterministic whole-modality dropping')
    parser.add_argument('--train_mixed_drop_policy', type=str, default='none', choices=MIXED_DROP_POLICIES, help='Training-only mutually exclusive clean/temporal/whole corruption mixture')
    parser.add_argument('--train_mixed_drop_probabilities', type=float, nargs=3, default=[0.25, 0.375, 0.375], metavar=('CLEAN', 'TEMPORAL', 'WHOLE'), help='Mixed-drop probabilities for clean, temporal, and one whole-modality outage')
    parser.add_argument('--train_mixed_drop_temporal_ratios', type=float, nargs='+', default=[0.1, 0.3, 0.5], help='Positive ratios sampled conditionally within the mixed temporal regime')
    parser.add_argument('--train_mixed_drop_temporal_pattern', type=str, default='independent', choices=MIXED_DROP_TEMPORAL_PATTERNS, help='Mask geometry used within the mixed temporal regime')
    parser.add_argument('--train_mixed_drop_seed', type=int, default=42, help='Dedicated seed for deterministic mixed corruption sampling')

    # RankNet loss settings (only used when loss_type='ranknet')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'ranknet', 'pearson', 'mse_pearson', 'pearson_ranknet', 'trend', 'trend_cls', 'trend_cls_mse', 'salience_cls', 'cls_mse_focal', 'cls_mse_emd'], help='Loss function type')
    parser.add_argument('--trend_label_file', type=str, default='mosu_gt_trend_w5.h5', help='HDF5 file containing smoothed forward-delta labels')
    parser.add_argument('--trend_class_label_file', type=str, default='mosu_gt_trend_cls_w5_q50.h5', help='HDF5 file containing three-class trend labels')
    parser.add_argument('--mse_weight', type=float, default=0.2, help='Weight of MSE term in combined ranknet loss')
    parser.add_argument('--ranknet_weight', type=float, default=0.8, help='Weight of RankNet term in combined ranknet loss')
    parser.add_argument('--num_pairs', type=int, default=512, help='Number of pairs sampled per video per step for RankNet loss')
    parser.add_argument('--min_gap', type=float, default=0.1, help='Minimum GT score gap to consider a pair for ranking loss')
    parser.add_argument('--peak_top_frac', type=float, default=0.15, help='Fraction of top GT frames used as positive anchors in peak-biased sampling')
    parser.add_argument('--combined_schedule', type=str, default='static', choices=['static', 'mse_to_pr', 'mse_to_p_to_pr', 'rank_decay', 'rank_ramp', 'anchor'], help='MSE/Pearson/RankNet weight schedule')
    parser.add_argument('--combined_mse_weight', type=float, default=0.0, help='Final MSE coefficient for pearson_ranknet')
    parser.add_argument('--combined_pearson_weight', type=float, default=0.8, help='Final Pearson coefficient for pearson_ranknet')
    parser.add_argument('--combined_ranknet_weight', type=float, default=0.2, help='Final RankNet coefficient for pearson_ranknet')
    parser.add_argument('--combined_stage1_end', type=int, default=5, help='End of initial MSE stage')
    parser.add_argument('--combined_stage2_end', type=int, default=10, help='End of MSE-to-Pearson/PR transition')
    parser.add_argument('--combined_stage3_end', type=int, default=15, help='End of RankNet ramp/decay stage')
    parser.add_argument('--combined_ranknet_start_weight', type=float, default=0.3, help='Initial RankNet coefficient for rank_decay')
    parser.add_argument('--combined_ranknet_end_weight', type=float, default=0.1, help='Final RankNet coefficient for rank_decay')
    parser.add_argument('--rank_variant', type=str, default='pairwise', choices=['pairwise', 'boundary', 'error_boundary', 'rankw', 'softndcg', 'approxap'], help='Shape of the RankNet term in pearson_ranknet: uniform pairwise (default), boundary pairwise, detached error-conditioned boundary pairwise, position-weighted pairwise, listwise ApproxNDCG@K, or listwise ApproxAP')
    parser.add_argument('--rank_temperature', type=float, default=1.0, help='Soft-rank sigmoid temperature for rank_variant in {softndcg, approxap}')
    parser.add_argument('--rank_error_floor', type=float, default=0.25, help='Minimum pre-normalization pair weight for rank_variant=error_boundary')
    parser.add_argument('--rank_error_temperature', type=float, default=1.0, help='Detached signed-margin temperature for rank_variant=error_boundary')
    parser.add_argument('--listwise_max_frames', type=int, default=600, help='Subsampling cap for the O(N^2) soft-rank matrix in rank_variant in {softndcg, approxap} on long videos')
    parser.add_argument('--ranknet_seed_offset', type=int, default=None, help='Optional offset added to the run seed for a dedicated checkpointed RankNet sampling RNG')
    parser.add_argument('--loss_dynamics', action='store_true', help='Record per-step MSE/Pearson/RankNet gradient diagnostics')
    parser.add_argument('--loss_dynamics_flush_interval', type=int, default=20, help='Number of loss-dynamics records between JSONL flushes')
    parser.add_argument('--loss_dynamics_ranknet_seed_offset', type=int, default=100000, help='Offset added to the run seed for the dedicated RankNet sampling RNG')
    parser.add_argument('--loss_dynamics_probe_size', type=int, default=64, help='Number of fixed validation videos evaluated during loss-dynamics runs')
    parser.add_argument('--loss_dynamics_probe_batch_size', type=int, default=8, help='Batch size used to stream the fixed validation probe')
    parser.add_argument('--loss_dynamics_probe_seed', type=int, default=314159, help='Seed for deterministic validation-probe selection')
    parser.add_argument('--loss_dynamics_probe_interval', type=int, default=1, help='Optimizer-step interval between fixed validation probe evaluations')
    parser.add_argument('--loss_dynamics_snapshot_interval', type=int, default=1, help='Epoch interval between raw fixed-probe gradient snapshots')
    parser.add_argument('--save_per_video_metrics', action='store_true', help='Append per-video kTau/sRho/mAP50/mAP15 to {output_dir}/per_video_metrics.jsonl on every evaluate() call')
    parser.add_argument('--save_epoch_ckpts', action='store_true', help='Save model.state_dict() for every epoch to {output_dir}/epoch_ckpts/epoch_NNN_ckpt.pth')
    parser.add_argument('--exclude_ids_file', type=str, default=None, help='Path to a newline-separated video_id file to exclude from the TRAIN split only (no-op if unset)')
    parser.add_argument('--sample_weights_file', type=str, default=None, help='Path to a "video_id,weight" CSV to down-weight (not exclude) specific TRAIN videos in the training loss (missing ids default to weight 1.0; no-op if unset)')
    parser.add_argument('--pearson_warmup_start', type=int, default=5, help='Epoch to start blending Pearson into MSE (mse_pearson only)')
    parser.add_argument('--pearson_warmup_end',   type=int, default=10, help='Epoch at which Pearson fully replaces MSE (mse_pearson only)')
    parser.add_argument('--num_cls_bins', type=int, default=10, help='Number of ordinal bins for cls_mse_focal / cls_mse_emd loss')
    parser.add_argument('--cls_loss_weight', type=float, default=0.5, help='Weight of classification loss; MSE weight = 1 - cls_loss_weight')

    # Optional Cross-Temporal Matching regularization
    parser.add_argument('--ctm_weight', type=float, default=0.0, help='CTM auxiliary-loss coefficient; 0 disables CTM')
    parser.add_argument('--ctm_temperature', type=float, default=0.07, help='Temperature for CTM similarity distributions')
    parser.add_argument('--ctm_sigma', type=float, default=1.0, help='Gaussian CTM bandwidth in seconds')
    parser.add_argument('--ctm_radius', type=int, default=-1, help='CTM local radius; -1 uses ceil(3*sigma)')
    parser.add_argument('--ctm_num_negatives', type=int, default=16, help='Distant CTM candidates sampled per source token')
    parser.add_argument('--ctm_warmup_start', type=int, default=3, help='Last epoch with CTM disabled')
    parser.add_argument('--ctm_warmup_end', type=int, default=5, help='Epoch where CTM reaches its full coefficient')
    
    # Load YAML config
    temp_args, _ = parser.parse_known_args()
    yaml_name = temp_args.cfg if temp_args.cfg else temp_args.dataset
    yaml_path = f"./configs/{yaml_name}.yaml"
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
        parser.set_defaults(**yaml_config)
    args = parser.parse_args()
    if args.dataset not in ('summe', 'tvsum') and args.split_protocol != 'tvt':
        parser.error('--split_protocol tv is only valid for summe/tvsum')
    if len(set(args.input_modalities)) != len(args.input_modalities):
        parser.error('--input_modalities must not contain duplicates')
    args.train_frame_drop_ratios = list(validate_frame_drop_config(
        args.train_frame_drop_pattern,
        args.train_frame_drop_ratios,
    ))
    try:
        args.train_modality_drop_probability = validate_whole_modality_drop_config(
            args.train_modality_drop_policy,
            args.train_modality_drop_probability,
        )
        validate_drop_policy_compatibility(
            args.train_frame_drop_pattern,
            args.train_modality_drop_policy,
        )
        validate_mixed_drop_compatibility(
            args.train_mixed_drop_policy,
            args.train_frame_drop_pattern,
            args.train_modality_drop_policy,
        )
        mixed_probabilities, mixed_ratios = validate_mixed_drop_config(
            args.train_mixed_drop_probabilities,
            args.train_mixed_drop_temporal_ratios,
        )
        args.train_mixed_drop_temporal_pattern = validate_mixed_drop_temporal_pattern(
            args.train_mixed_drop_temporal_pattern
        )
        args.train_mixed_drop_probabilities = list(mixed_probabilities)
        args.train_mixed_drop_temporal_ratios = list(mixed_ratios)
    except ValueError as error:
        parser.error(str(error))
    if not 0.0 <= args.train_frame_drop_clean_probability <= 1.0:
        parser.error('--train_frame_drop_clean_probability must be in [0, 1]')
    if not 0.0 < args.rank_error_floor <= 1.0:
        parser.error('--rank_error_floor must be in (0, 1]')
    if args.rank_error_temperature <= 0.0:
        parser.error('--rank_error_temperature must be positive')
    if args.train_frame_drop_curriculum_stage1_end < 1:
        parser.error('--train_frame_drop_curriculum_stage1_end must be at least 1')
    if (args.train_frame_drop_curriculum_stage2_end
            <= args.train_frame_drop_curriculum_stage1_end):
        parser.error('frame-drop curriculum stage 2 must end after stage 1')
    if args.train_frame_drop_view_mode == 'dual':
        if args.loss_type != 'pearson_ranknet':
            parser.error('dual-view frame-drop training requires loss_type=pearson_ranknet')
        has_corruption = (
            args.train_frame_drop_pattern != 'none'
            or args.train_modality_drop_policy != 'none'
            or args.train_mixed_drop_policy != 'none'
        )
        if not has_corruption:
            parser.error('dual-view training requires a corruption policy')
        if args.train_frame_drop_clean_probability != 0.0:
            parser.error('dual-view training uses an explicit clean view; clean probability must be 0')
        weights = (
            args.train_frame_drop_clean_weight,
            args.train_frame_drop_corrupt_weight,
            args.train_frame_drop_kd_weight,
        )
        if any(weight < 0.0 for weight in weights):
            parser.error('dual-view loss weights must be non-negative')
        if abs(sum(weights) - 1.0) > 1e-8:
            parser.error('dual-view clean, corrupt, and KD weights must sum to 1')
    elif args.train_frame_drop_kd_weight != 0.0:
        parser.error('frame-drop KD requires --train_frame_drop_view_mode dual')
    if args.train_frame_drop_kd_weight > 0.0:
        if (
            args.train_frame_drop_teacher_mode == 'frozen'
            and args.train_frame_drop_teacher_ckpt is None
        ):
            parser.error('frame-drop KD requires --train_frame_drop_teacher_ckpt')
        if (
            args.train_frame_drop_teacher_mode == 'frozen'
            and not os.path.isfile(args.train_frame_drop_teacher_ckpt)
        ):
            parser.error(
                f'frame-drop teacher checkpoint not found: '
                f'{args.train_frame_drop_teacher_ckpt}'
            )
    elif args.train_frame_drop_teacher_mode == 'online':
        parser.error('online teacher mode requires a positive frame-drop KD weight')
    
    # Create output directory
    args.output_dir = os.path.join('./outputs', args.dataset, args.model, args.exp_name)
    
    return args