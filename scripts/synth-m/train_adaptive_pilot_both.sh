#!/usr/bin/env bash
set -euo pipefail

# Environment variables may override these pilot defaults, for example:
# EPOCHS=30 DEVICE=cuda:1 bash scripts/synth-m/train_adaptive_pilot_both.sh
TEACHER_NPZ="${TEACHER_NPZ:-./cache/adaptive/synth-m/run0_teacher_pilot.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./save/adaptive_controller/synth-m}"
LOG_DIR="${LOG_DIR:-./logs}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"

if [[ ! -f "${TEACHER_NPZ}" ]]; then
  echo "Teacher file not found: ${TEACHER_NPZ}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

train_controller() {
  local feature_mode="$1"
  local output_dir="${OUTPUT_ROOT}/pilot_${feature_mode}"
  local log_path="${LOG_DIR}/synth_m_pilot_${feature_mode}_train.log"

  echo "============================================================"
  echo "Training ${feature_mode}"
  echo "Teacher: ${TEACHER_NPZ}"
  echo "Output:  ${output_dir}"
  echo "Log:     ${log_path}"
  echo "============================================================"

  python tools/train_strength_controller.py \
    --teacher-npz "${TEACHER_NPZ}" \
    --output-dir "${output_dir}" \
    --feature-mode "${feature_mode}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    2>&1 | tee "${log_path}"
}

train_controller score_only
train_controller score_plus_pair

echo "============================================================"
echo "Both pilot controllers finished successfully."
echo "Score-only:      ${OUTPUT_ROOT}/pilot_score_only/best.pt"
echo "Score-plus-pair: ${OUTPUT_ROOT}/pilot_score_plus_pair/best.pt"
echo "============================================================"
