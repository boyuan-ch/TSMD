# Set up Weights & Biases (wandb) logging
def setup_wandb(cfg):
    if cfg.wandb:
        import wandb
        project_name = f"{cfg.dataset}"
        # Initialize wandb run with the specified project name and configuration
        wandb.init(
            entity='TripleSumm',
            project=project_name,
            name=f"{cfg.model}-{cfg.exp_name}",
            config=vars(cfg),
            reinit=True
        )
        return wandb.run.url
    return None

# Log training and validation metrics to wandb
def wandb_training(cfg, train_results, val_results, epoch):
    if cfg.wandb:
        if 'video_macro_f1' in train_results:
            wandb.log({
                "train/video_accuracy": train_results['video_accuracy'],
                "train/video_balanced_accuracy": train_results['video_balanced_accuracy'],
                "train/video_macro_f1": train_results['video_macro_f1'],
                "train/global_macro_f1": train_results['global_macro_f1'],
                "train/loss": train_results['loss'],
                "val/video_accuracy": val_results['video_accuracy'],
                "val/video_balanced_accuracy": val_results['video_balanced_accuracy'],
                "val/video_macro_f1": val_results['video_macro_f1'],
                "val/global_macro_f1": val_results['global_macro_f1'],
                "val/loss": val_results['loss'],
            }, step=epoch)
        elif 'macro_f1' in train_results:
            wandb.log({
                "train/accuracy": train_results['accuracy'],
                "train/balanced_accuracy": train_results['balanced_accuracy'],
                "train/macro_f1": train_results['macro_f1'],
                "train/loss": train_results['loss'],
                "val/accuracy": val_results['accuracy'],
                "val/balanced_accuracy": val_results['balanced_accuracy'],
                "val/macro_f1": val_results['macro_f1'],
                "val/loss": val_results['loss'],
                "val/decreasing_f1": val_results['decreasing_f1'],
                "val/flat_f1": val_results['flat_f1'],
                "val/increasing_f1": val_results['increasing_f1'],
            }, step=epoch)
        else:
            wandb.log({
                f"train/kTau": train_results['ktau'],
                f"train/sRho": train_results['srho'],
                f"train/mAP50": train_results['map50'],
                f"train/mAP15": train_results['map15'],
                f"train/Loss": train_results['loss'],
                f"val/kTau": val_results['ktau'],
                f"val/sRho": val_results['srho'],
                f"val/mAP50": val_results['map50'],
                f"val/mAP15": val_results['map15'],
                f"val/Loss": val_results['loss']
            }, step=epoch)

# Log summary metrics to wandb
def wandb_summary(cfg, train_results, val_results, test_results, fold=None):
    if cfg.wandb:
        if 'video_macro_f1' in train_results:
            wandb.summary.update({
                "train/video_accuracy": train_results['video_accuracy'],
                "train/video_balanced_accuracy": train_results['video_balanced_accuracy'],
                "train/video_macro_f1": train_results['video_macro_f1'],
                "val/video_accuracy": val_results['video_accuracy'],
                "val/video_balanced_accuracy": val_results['video_balanced_accuracy'],
                "val/video_macro_f1": val_results['video_macro_f1'],
            })
            wandb.log({
                "test/video_accuracy": test_results['video_accuracy'],
                "test/video_balanced_accuracy": test_results['video_balanced_accuracy'],
                "test/video_macro_f1": test_results['video_macro_f1'],
                "test/global_accuracy": test_results['global_accuracy'],
                "test/global_balanced_accuracy": test_results['global_balanced_accuracy'],
                "test/global_macro_f1": test_results['global_macro_f1'],
                "test/global_bin_0_f1": test_results['global_bin_0_f1'],
                "test/global_bin_1_f1": test_results['global_bin_1_f1'],
                "test/global_bin_2_f1": test_results['global_bin_2_f1'],
                "test/global_bin_3_f1": test_results['global_bin_3_f1'],
            })
        elif 'macro_f1' in train_results:
            wandb.summary.update({
                "train/accuracy": train_results['accuracy'],
                "train/balanced_accuracy": train_results['balanced_accuracy'],
                "train/macro_f1": train_results['macro_f1'],
                "val/accuracy": val_results['accuracy'],
                "val/balanced_accuracy": val_results['balanced_accuracy'],
                "val/macro_f1": val_results['macro_f1'],
            })
            wandb.log({
                "test/accuracy": test_results['accuracy'],
                "test/balanced_accuracy": test_results['balanced_accuracy'],
                "test/macro_f1": test_results['macro_f1'],
                "test/decreasing_f1": test_results['decreasing_f1'],
                "test/flat_f1": test_results['flat_f1'],
                "test/increasing_f1": test_results['increasing_f1'],
            })
        else:
            wandb.summary.update({
                "train/kTau": train_results['ktau'],
                "train/sRho": train_results['srho'],
                "train/mAP50": train_results['map50'],
                "train/mAP15": train_results['map15'],
                "val/kTau": val_results['ktau'],
                "val/sRho": val_results['srho'],
                "val/mAP50": val_results['map50'],
                "val/mAP15": val_results['map15']
            })
            wandb.log({
                "test/kTau": test_results['ktau'],
                "test/sRho": test_results['srho'],
                "test/mAP50": test_results['map50'],
                "test/mAP15": test_results['map15']
            })
        wandb.finish()