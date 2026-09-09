#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <mosu|mrhisum> <checkpoint>" >&2
    exit 2
fi

dataset=$1
checkpoint=$2
case "$dataset" in
    mosu|mrhisum) ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

exec python main.py \
    --mode test \
    --dataset "$dataset" \
    --cfg "${dataset}_mpr" \
    --exp_name "eval-$(basename "$checkpoint" .pth)" \
    --data_dir "${DATA_DIR:-./data}" \
    --model_ckpt "$checkpoint" \
    --results_json "${RESULTS_JSON:-./results/${dataset}-clean.json}"
