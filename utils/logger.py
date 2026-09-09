import os
import copy
import torch
import logging
from thop import profile

# Set up logger
def setup_logger(name, output_dir=None, overwrite=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter('%(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
        # File handler
        if overwrite:
            file_handler = logging.FileHandler(os.path.join(output_dir, f'{name}.log'), mode='w')
        else:
            file_handler = logging.FileHandler(os.path.join(output_dir, f'{name}.log'), mode='a')
        file_handler.setFormatter(stream_formatter)
        logger.addHandler(file_handler)
    return logger

# Log configuration details
def log_config(logger, cfg):
    logger.info("[Configuration]")
    for key, value in vars(cfg).items():
        logger.info(f"    - {key}: {value}")
    logger.info("")

# Log dataset statistics
def log_dataset(logger, train_dataset, val_dataset, test_dataset):
    logger.info("[Dataset]")
    if train_dataset is not None and val_dataset is not None:
        logger.info(f"    - Train samples: {len(train_dataset)}")
        logger.info(f"    -   Val samples: {len(val_dataset)}")
    logger.info(f"    -  Test samples: {len(test_dataset)}")
    logger.info("")

# Log model architecture and complexity
def log_model(logger, cfg, model):
    # Create dummy inputs for FLOPs calculation
    dummy_visual = torch.randn(1, 300, cfg.visual_dim).to(cfg.device)
    dummy_text = torch.randn(1, 300, cfg.text_dim).to(cfg.device)
    dummy_audio = torch.randn(1, 300, cfg.audio_dim).to(cfg.device)
    dummy_mask = torch.ones(1, 300).bool().to(cfg.device)
    inputs = (dummy_visual, dummy_text, dummy_audio, dummy_mask)
    # Compute FLOPs and parameter count
    model_for_profile = copy.deepcopy(model).to(cfg.device)
    model_for_profile.eval()
    with torch.no_grad():
        flops, _ = profile(model_for_profile, inputs=inputs, verbose=False)
    del model_for_profile
    gflops = flops / 1e9
    params = sum(p.numel() for p in model.parameters())
    # Log model details
    logger.info(f"[Model Parameters]")
    logger.info(f"\t- Params: {params:,}")
    logger.info(f"\t- GFLOPs: {gflops:.2f} G")
    logger.info("")
    logger.info(f"[Model Structure]")
    logger.info(f"{model}")
    logger.info("")

# Log training progress and metrics
def log_training(logger, train_results, val_results, epoch):
    if 'video_macro_f1' in train_results:
        logger.info(
            f"[Epoch {epoch:03d}] "
            f"Train VideoAcc: {train_results['video_accuracy']:.3f} "
            f"Train VideoBalAcc: {train_results['video_balanced_accuracy']:.3f} "
            f"Train VideoMacroF1: {train_results['video_macro_f1']:.3f} "
            f"Train Loss: {train_results['loss']:.6f} "
            f"Val VideoAcc: {val_results['video_accuracy']:.3f} "
            f"Val VideoBalAcc: {val_results['video_balanced_accuracy']:.3f} "
            f"Val VideoMacroF1: {val_results['video_macro_f1']:.3f} "
            f"Val GlobalMacroF1: {val_results['global_macro_f1']:.3f} "
            f"Val Loss: {val_results['loss']:.6f}"
        )
        logger.info(
            f"\t  Val global per-bin F1: "
            f"bin0={val_results['global_bin_0_f1']:.3f} "
            f"bin1={val_results['global_bin_1_f1']:.3f} "
            f"bin2={val_results['global_bin_2_f1']:.3f} "
            f"bin3={val_results['global_bin_3_f1']:.3f} "
            f"confusion={val_results['global_confusion_matrix']}"
        )
        return
    if 'macro_f1' in train_results:
        logger.info(
            f"[Epoch {epoch:03d}] "
            f"Train Acc: {train_results['accuracy']:.3f} "
            f"Train BalAcc: {train_results['balanced_accuracy']:.3f} "
            f"Train MacroF1: {train_results['macro_f1']:.3f} "
            f"Train Loss: {train_results['loss']:.6f} "
            f"Val Acc: {val_results['accuracy']:.3f} "
            f"Val BalAcc: {val_results['balanced_accuracy']:.3f} "
            f"Val MacroF1: {val_results['macro_f1']:.3f} "
            f"Val Loss: {val_results['loss']:.6f}"
        )
        logger.info(
            f"\t  Val per-class F1: "
            f"decreasing={val_results['decreasing_f1']:.3f} "
            f"flat={val_results['flat_f1']:.3f} "
            f"increasing={val_results['increasing_f1']:.3f} "
            f"support=({val_results['decreasing_support']}, "
            f"{val_results['flat_support']}, {val_results['increasing_support']}) "
            f"confusion={val_results['confusion_matrix']}"
        )
        return
    logger.info(
        f"[Epoch {epoch:03d}] "
        f"Train kTau: {train_results['ktau']:.3f} "
        f"Train sRho: {train_results['srho']:.3f} "
        f"Train mAP50: {train_results['map50']:.2f} "
        f"Train mAP15: {train_results['map15']:.2f} "
        f"Train Loss: {train_results['loss']:.6f} "
        f"Val kTau: {val_results['ktau']:.3f} "
        f"Val sRho: {val_results['srho']:.3f} "
        f"Val mAP50: {val_results['map50']:.2f} "
        f"Val mAP15: {val_results['map15']:.2f} "
        f"Val Loss: {val_results['loss']:.6f}"
    )

# Log evaluation results
def log_results(logger, results_list, mode_list):
    for results, mode in zip(results_list, mode_list):
        if 'video_macro_f1' in results:
            logger.info(
                f"[{mode.upper():>5} Results] "
                f"VideoAcc: {results['video_accuracy']:.3f} "
                f"VideoBalAcc: {results['video_balanced_accuracy']:.3f} "
                f"VideoMacroF1: {results['video_macro_f1']:.3f} "
                f"GlobalAcc: {results['global_accuracy']:.3f} "
                f"GlobalBalAcc: {results['global_balanced_accuracy']:.3f} "
                f"GlobalMacroF1: {results['global_macro_f1']:.3f} "
                f"Confusion: {results['global_confusion_matrix']}"
            )
        elif 'macro_f1' in results:
            logger.info(
                f"[{mode.upper():>5} Results] "
                f"Acc: {results['accuracy']:.3f} "
                f"BalAcc: {results['balanced_accuracy']:.3f} "
                f"MacroF1: {results['macro_f1']:.3f} "
                f"Confusion: {results['confusion_matrix']}"
            )
        else:
            logger.info(
                f"[{mode.upper():>5} Results] "
                f"kTau: {results['ktau']:.3f} "
                f"sRho: {results['srho']:.3f} "
                f"mAP50: {results['map50']:.2f} "
                f"mAP15: {results['map15']:.2f}"
            )
    logger.info("")