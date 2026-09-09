"""Evaluate a TSMD checkpoint under deterministic temporal or stream missingness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import CollateFn, Dataset
from models import build_model
from utils.compute_metrics import evaluate_highlight, evaluate_summary

MODALITIES = ("visual", "text", "audio")
PATTERNS = ("clean", "independent", "synchronized", "missing-visual", "missing-text", "missing-audio")


def stable_rng(*parts: object) -> np.random.Generator:
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def missing_indices(length: int, ratio: float, *key: object) -> np.ndarray:
    count = int(round(length * ratio))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    return stable_rng("tsmd-eval-v1", *key).choice(length, count, replace=False)


def apply_missingness(
    features: dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
    video_ids: list[str],
    pattern: str,
    ratio: float,
    seed: int,
) -> None:
    """Zero selected valid feature rows in place without changing padding."""
    for sample_index, video_id in enumerate(video_ids):
        length = int(valid_mask[sample_index].sum().item())
        if pattern.startswith("missing-"):
            modality = pattern.removeprefix("missing-")
            features[modality][sample_index, :length] = 0
            continue
        if pattern == "clean":
            continue
        if pattern == "synchronized":
            indices = missing_indices(length, ratio, seed, video_id, pattern)
            for modality in MODALITIES:
                features[modality][sample_index, indices] = 0
            continue
        if pattern == "independent":
            for modality in MODALITIES:
                indices = missing_indices(length, ratio, seed, video_id, pattern, modality)
                features[modality][sample_index, indices] = 0
            continue
        raise ValueError(f"Unknown missingness pattern: {pattern}")


def load_config(path: Path, dataset: str, data_dir: Path) -> SimpleNamespace:
    with path.open() as config_file:
        values = yaml.safe_load(config_file)
    values.update(
        dataset=dataset,
        data_dir=str(data_dir),
        input_modalities=["visual", "text", "audio"],
        salience_smoothing_window=1,
        fold=0,
        split_protocol="tvt",
        use_genre=False,
        max_seq_len=10000,
        get_attn_weights=False,
        num_cls_bins=10,
    )
    return SimpleNamespace(**values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("mosu", "mrhisum"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--pattern", required=True, choices=PATTERNS)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.ratio <= 1.0:
        parser.error("--ratio must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    config_path = args.config or Path("configs") / f"{args.dataset}_mpr.yaml"
    cfg = load_config(config_path, args.dataset, args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Dataset(cfg, split="test")
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=CollateFn(),
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    ktau_values: list[float] = []
    srho_values: list[float] = []
    map50_values: list[float] = []
    map15_values: list[float] = []
    evaluated = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{args.dataset}: {args.pattern}"):
            mask = batch["mask"].to(device)
            features = {
                modality: batch[f"{modality}_feat"].to(device)
                for modality in MODALITIES
            }
            apply_missingness(features, mask, batch["video_id"], args.pattern, args.ratio, args.mask_seed)
            output = model(
                features["visual"], features["text"], features["audio"], mask=mask
            )[0]
            if output.ndim == 3:
                output = output.squeeze(-1)

            predictions = output.cpu().numpy().tolist()
            targets = batch.get("eval_gt_score", batch["gt_score"]).numpy().tolist()
            metric_mask = mask.cpu().numpy()
            _, _, batch_ktau, batch_srho = evaluate_summary(
                predictions, targets, metric_mask, return_per_video=True
            )
            _, _, batch_map50, batch_map15 = evaluate_highlight(
                predictions, targets, metric_mask, return_per_video=True
            )
            ktau_values.extend(batch_ktau)
            srho_values.extend(batch_srho)
            map50_values.extend(batch_map50)
            map15_values.extend(batch_map15)
            evaluated += len(batch["video_id"])

    results = {
        "dataset": args.dataset,
        "pattern": args.pattern,
        "ratio": args.ratio,
        "mask_seed": args.mask_seed,
        "videos": evaluated,
        "ktau": float(np.nanmean(ktau_values)),
        "srho": float(np.nanmean(srho_values)),
        "map50": float(np.mean(map50_values) * 100),
        "map15": float(np.mean(map15_values) * 100),
    }
    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
