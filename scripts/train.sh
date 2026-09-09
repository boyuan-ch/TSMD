#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <mosu|mrhisum> <clean|temporal|stream|mix> [seed]" >&2
    exit 2
fi

dataset=$1
policy=$2
seed=${3:-42}

case "$dataset" in
    mosu|mrhisum) ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac
case "$policy" in
    clean|temporal|stream|mix) ;;
    *) echo "Unknown policy: $policy" >&2; exit 2 ;;
esac

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

command=(
    python main.py
    --mode train
    --dataset "$dataset"
    --cfg "${dataset}_mpr"
    --exp_name "mpr-${policy}-s${seed}"
    --data_dir "${DATA_DIR:-./data}"
    --seed "$seed"
    --amp
)

case "$policy" in
    clean) ;;
    temporal)
        command+=(
            --train_frame_drop_pattern independent
            --train_frame_drop_ratios 0.0 0.1 0.3 0.5
            --train_frame_drop_seed "$seed"
        )
        ;;
    stream)
        command+=(
            --train_modality_drop_policy categorical
            --train_modality_drop_probability 0.0
            --train_modality_drop_seed "$seed"
        )
        ;;
    mix)
        command+=(
            --train_mixed_drop_policy mixed
            --train_mixed_drop_probabilities 0.25 0.375 0.375
            --train_mixed_drop_temporal_ratios 0.1 0.3 0.5
            --train_mixed_drop_temporal_pattern independent
            --train_mixed_drop_seed "$seed"
        )
        ;;
esac

printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'
exec "${command[@]}"
