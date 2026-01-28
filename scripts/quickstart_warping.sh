#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-session_warping}"
INPUT_DIR="${INPUT_DIR:-data/dataset/COVID-19_Radiography_Dataset}"
ENSEMBLE_CONFIG="${ENSEMBLE_CONFIG:-configs/ensemble_best.json}"
CANONICAL_DIR="${CANONICAL_DIR:-outputs/shape_analysis}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-outputs/landmark_predictions/${SESSION}/predictions.npz}"
WARPED_OUTPUT_DIR="${WARPED_OUTPUT_DIR:-outputs/warped_lung_best/${SESSION}}"

CANONICAL_SHAPE="$CANONICAL_DIR/canonical_shape_gpa.json"
CANONICAL_TRIANGLES="$CANONICAL_DIR/canonical_delaunay_triangles.json"

if [[ ! -f "$CANONICAL_SHAPE" || ! -f "$CANONICAL_TRIANGLES" ]]; then
  python -m src_v2 compute-canonical data/coordenadas/coordenadas_maestro.csv \
    --output-dir "$CANONICAL_DIR" --visualize
fi

python scripts/predict_landmarks_dataset.py \
  --input-dir "$INPUT_DIR" \
  --output "$PREDICTIONS_PATH" \
  --ensemble-config "$ENSEMBLE_CONFIG" \
  --tta --clahe --clahe-clip 2.0 --clahe-tile 4

python -m src_v2 generate-dataset \
  "$INPUT_DIR" \
  "$WARPED_OUTPUT_DIR" \
  --canonical "$CANONICAL_SHAPE" \
  --triangles "$CANONICAL_TRIANGLES" \
  --margin 1.05 \
  --splits 0.75,0.125,0.125 \
  --seed 42 \
  --clahe --clahe-clip 2.0 --clahe-tile 4 \
  --no-full-coverage \
  --predictions "$PREDICTIONS_PATH"
