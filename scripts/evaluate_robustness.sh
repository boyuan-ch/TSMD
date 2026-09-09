#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <mosu|mrhisum> <checkpoint> [output-directory]" >&2
    exit 2
fi

dataset=$1
checkpoint=$2
output_dir=${3:-./results/robustness/$dataset}
case "$dataset" in
    mosu|mrhisum) ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
mkdir -p "$output_dir"

common=(
    python evaluate_robustness.py
    --dataset "$dataset"
    --checkpoint "$checkpoint"
    --data-dir "${DATA_DIR:-./data}"
    --batch-size "${BATCH_SIZE:-64}"
    --num-workers "${NUM_WORKERS:-4}"
)

for seed in 42 123 2026; do
    "${common[@]}" --pattern independent --ratio 0.5 --mask-seed "$seed" \
        --output "$output_dir/independent-r50-seed${seed}.json"
done
for modality in visual audio text; do
    "${common[@]}" --pattern "missing-${modality}" \
        --output "$output_dir/missing-${modality}.json"
done
