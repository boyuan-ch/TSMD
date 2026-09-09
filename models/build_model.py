from torch import optim
import math
from models.model import TripleSumm
from models.genre_model import TripleSummGenreV1, TripleSummGenreV2, TripleSummGenreV3
from models.ranknet_model import TripleSummRankNet
from models.cls_model import TripleSummClsMSE, TripleSummSalienceCls
from models.mode_model import TripleSummModeCls

_MODEL_REGISTRY = {
    'triplesumm': TripleSumm,
    'triplesumm-genrev1': TripleSummGenreV1,
    'triplesumm-genrev2': TripleSummGenreV2,
    'triplesumm-genrev3': TripleSummGenreV3,
    'triplesumm-ranknet': TripleSummRankNet,
    'triplesumm-trend-cls': TripleSummClsMSE,
    'triplesumm-salience-cls': TripleSummSalienceCls,
    'triplesumm-clsmse': TripleSummClsMSE,
    'triplesumm-modecls': TripleSummModeCls,
}

_GENRE_FAMILY = (
    'triplesumm-genrev1', 'triplesumm-genrev2', 'triplesumm-genrev3',
    'triplesumm-modecls',
)

# Build model based on configuration
def build_model(cfg):
    model_cls = _MODEL_REGISTRY.get(cfg.model)
    if model_cls is None:
        raise ValueError(f"Unknown model '{cfg.model}'. Available: {list(_MODEL_REGISTRY.keys())}")
    kwargs = dict(
        visual_dim=cfg.visual_dim,
        text_dim=cfg.text_dim,
        audio_dim=cfg.audio_dim,
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_model_layers=cfg.num_model_layers,
        num_mst_layers=cfg.num_mst_layers,
        num_cmf_layers=cfg.num_cmf_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        window_size=cfg.window_size,
        max_seq_len=cfg.max_seq_len,
        get_attn_weights=cfg.get_attn_weights,
    )
    if cfg.model in ('triplesumm-trend-cls', 'triplesumm-salience-cls', 'triplesumm-clsmse'):
        kwargs['num_cls_bins'] = cfg.num_cls_bins
        kwargs['use_regression_head'] = cfg.model == 'triplesumm-clsmse'
    if cfg.model in _GENRE_FAMILY:
        kwargs['num_genre_classes'] = cfg.num_genre_classes
    model = model_cls(**kwargs)
    return model

# Build optimizer
def build_optimizer(cfg, model):
    if cfg.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    return optimizer

# Build learning rate scheduler
def build_scheduler(cfg, optimizer, num_training_steps=None):
    if cfg.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cfg.num_epochs,
            eta_min=0
        )
    elif cfg.scheduler == 'cosine_warmup':
        assert num_training_steps is not None, \
            "cosine_warmup requires num_training_steps (pass len(train_loader)*num_epochs)"
        warmup_steps = int(cfg.warmup_ratio * num_training_steps)
        def _lr_lambda(current_step, _w=warmup_steps, _t=num_training_steps):
            if current_step < _w:
                return float(current_step) / max(1, _w)
            progress = float(current_step - _w) / max(1, _t - _w)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    else:
        scheduler = None
    return scheduler
    return scheduler