#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-session_quickstart}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/landmarks_train_base.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/repro_quickstart/${SESSION}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/repro_quickstart/${SESSION}}"

SEEDS=(123 321 111 666)

for seed in "${SEEDS[@]}"; do
  python scripts/train.py --config "$TRAIN_CONFIG" \
    --seed "$seed" --split-seed "$seed" \
    --save-dir "$CHECKPOINT_ROOT/seed${seed}" \
    --output-dir "$OUTPUT_ROOT/seed${seed}"
done

MODELS=(
  "$CHECKPOINT_ROOT/seed123/final_model.pt"
  "$CHECKPOINT_ROOT/seed321/final_model.pt"
  "$CHECKPOINT_ROOT/seed111/final_model.pt"
  "$CHECKPOINT_ROOT/seed666/final_model.pt"
)

python -m src_v2 evaluate-ensemble "${MODELS[@]}" --tta --clahe

python scripts/extract_predictions.py \
  --models "${MODELS[@]}" \
  --output-dir "$OUTPUT_ROOT/predictions" \
  --split-seed 42 \
  --tta --clahe
