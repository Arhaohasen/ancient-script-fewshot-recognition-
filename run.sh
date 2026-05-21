#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
SAIS_DATA_DIR="${SAIS_DATA_DIR:-/saisdata}"
SAIS_RESULT_DIR="${SAIS_RESULT_DIR:-/saisresult}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${APP_DIR}/best.pt}"
PREDICTION_SCHEMA="${PREDICTION_SCHEMA:-sais}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_BOXES="${MAX_BOXES:-64}"
MAX_DETECTION_SIDE="${MAX_DETECTION_SIDE:-800}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-25}"

mkdir -p "${SAIS_RESULT_DIR}"

if [[ ! -d "${SAIS_DATA_DIR}" ]]; then
  echo "Input directory not found: ${SAIS_DATA_DIR}" >&2
  exit 2
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  echo "Set CHECKPOINT_PATH or copy best.pt into /app before submitting the image." >&2
  exit 3
fi

export PYTHONPATH="${APP_DIR}/04-code:${PYTHONPATH:-}"

python -m ancient_ocr.inference.predict_classifier \
  --input-dir "${SAIS_DATA_DIR}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output "${SAIS_RESULT_DIR}/prediction.json" \
  --batch-size "${BATCH_SIZE}" \
  --max-boxes "${MAX_BOXES}" \
  --max-detection-side "${MAX_DETECTION_SIDE}" \
  --progress-interval "${PROGRESS_INTERVAL}" \
  --schema "${PREDICTION_SCHEMA}"
