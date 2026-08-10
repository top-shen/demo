#!/usr/bin/env bash
set -euo pipefail

# One-command Synth-M adaptive-controller smoke pipeline.
# This deliberately consumes an existing pilot teacher and never starts a full
# teacher build. Environment variables can override the defaults below.
TEACHER_NPZ="${TEACHER_NPZ:-./cache/adaptive/synth-m/run0_teacher_pilot.npz}"
TEACHER_MANIFEST="${TEACHER_MANIFEST:-./cache/adaptive/synth-m/run0_teacher_pilot.json}"
FIXED_SWEEP="${FIXED_SWEEP:-./cache/adaptive/synth-m/run0_teacher_pilot_fixed_sweep.json}"
CONTROLLER_ROOT="${CONTROLLER_ROOT:-./save/adaptive_controller/synth-m}"
EVAL_ROOT="${EVAL_ROOT:-./save/adaptive_pilot_eval/synth-m}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-./save/adaptive_pilot_analysis/synth-m}"
LOG_DIR="${LOG_DIR:-./logs}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-1}"
SEED="${SEED:-42}"
COPY_THRESHOLD="${COPY_THRESHOLD:-0.05}"

for required_file in "${TEACHER_NPZ}" "${TEACHER_MANIFEST}" "${FIXED_SWEEP}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required pilot artifact not found: ${required_file}" >&2
    exit 2
  fi
done

mkdir -p "${CONTROLLER_ROOT}" "${EVAL_ROOT}" "${ANALYSIS_ROOT}" "${LOG_DIR}"

echo "[1/6] Validate the pilot teacher"
TEACHER_NPZ="${TEACHER_NPZ}" python - <<'PY'
import os
import numpy as np
from retrieval.strength_teacher import load_teacher_dataset

path = os.environ["TEACHER_NPZ"]
data, manifest = load_teacher_dataset(path, for_training=True)
same_sample = data["query_sample_ids"] == data["reference_sample_ids"]
if same_sample.any():
    raise RuntimeError("Teacher contains same-sample retrieval leakage")
print("teacher rows:", len(data["query_sample_ids"]))
print("gate positive/negative:", int(data["gate_targets"].sum()), int((data["gate_targets"] == 0).sum()))
print("controller train/validation:", int((data["controller_split_ids"] == 0).sum()), int((data["controller_split_ids"] == 1).sum()))
print("teacher validation: PASS")
PY

echo "[2/6] Train score_only and score_plus_pair"
TEACHER_NPZ="${TEACHER_NPZ}" \
OUTPUT_ROOT="${CONTROLLER_ROOT}" \
LOG_DIR="${LOG_DIR}" \
DEVICE="${DEVICE}" \
EPOCHS="${EPOCHS}" \
PATIENCE="${PATIENCE}" \
BATCH_SIZE="${BATCH_SIZE}" \
SEED="${SEED}" \
bash scripts/synth-m/train_adaptive_pilot_both.sh

run_evaluation() {
  local feature_mode="$1"
  local checkpoint="${CONTROLLER_ROOT}/pilot_${feature_mode}/best.pt"
  local save_folder="${EVAL_ROOT}/${feature_mode}"
  local log_path="${LOG_DIR}/synth_m_pilot_${feature_mode}_eval.log"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Controller checkpoint not found: ${checkpoint}" >&2
    exit 3
  fi

  bash scripts/synth-m/eval_adaptive.sh \
    --start_runid 0 \
    --n_runs 1 \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --eval_max_batches "${EVAL_MAX_BATCHES}" \
    --save_folder "${save_folder}" \
    --rag_controller_checkpoint_path "${checkpoint}" \
    --rag_controller_feature_mode "${feature_mode}" \
    2>&1 | tee "${log_path}"
}

echo "[3/6] Evaluate score_only"
run_evaluation score_only

echo "[4/6] Evaluate score_plus_pair"
run_evaluation score_plus_pair

run_analysis() {
  local feature_mode="$1"
  local controller_dir="${CONTROLLER_ROOT}/pilot_${feature_mode}"
  local evaluation_dir="${EVAL_ROOT}/${feature_mode}/0"
  local output_dir="${ANALYSIS_ROOT}/${feature_mode}"

  python tools/analyze_strength_controller.py \
    --teacher-npz "${TEACHER_NPZ}" \
    --teacher-manifest "${TEACHER_MANIFEST}" \
    --controller-predictions "${controller_dir}/validation_predictions.npz" \
    --evaluation-npz "${evaluation_dir}/rag_predictions.npz" \
    --retrieval-trace "${evaluation_dir}/retrieval_trace.jsonl" \
    --fixed-sweep "${FIXED_SWEEP}" \
    --copy-threshold "${COPY_THRESHOLD}" \
    --output-dir "${output_dir}"
}

echo "[5/6] Analyze score_only"
run_analysis score_only

echo "[6/6] Analyze score_plus_pair"
run_analysis score_plus_pair

echo "============================================================"
echo "Synth-M adaptive pilot pipeline completed successfully."
echo "Controllers: ${CONTROLLER_ROOT}/pilot_{score_only,score_plus_pair}"
echo "Evaluations: ${EVAL_ROOT}/{score_only,score_plus_pair}"
echo "Analyses:    ${ANALYSIS_ROOT}/{score_only,score_plus_pair}"
echo "Logs:        ${LOG_DIR}/synth_m_pilot_*"
echo "============================================================"
