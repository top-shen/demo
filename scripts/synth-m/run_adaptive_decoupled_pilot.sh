#!/usr/bin/env bash
set -euo pipefail

# Cheap architecture diagnostic on an already relabeled train-only Teacher.
# This trains independent gate/strength towers and never runs diffusion or the
# dataset validation/test evaluator.
TEACHER_NPZ="${TEACHER_NPZ:-./cache/adaptive/synth-m/run0_teacher_medium_q256_seed42_q05_eps1pct.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./save/adaptive_controller/synth-m/decoupled_q05_eps1pct}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-./save/adaptive_pilot_analysis/synth-m/decoupled_q05_eps1pct}"
LOG_DIR="${LOG_DIR:-./logs/decoupled_q05_eps1pct}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
LAMBDA_GATE="${LAMBDA_GATE:-5.0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

if [[ ! -f "${TEACHER_NPZ}" ]]; then
  echo "Relabeled Teacher not found: ${TEACHER_NPZ}" >&2
  echo "Run scripts/synth-m/run_adaptive_repair_pilot.sh first." >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}" "${ANALYSIS_ROOT}" "${LOG_DIR}"

GATE_POS_WEIGHT="${GATE_POS_WEIGHT:-$(TEACHER_NPZ="${TEACHER_NPZ}" python - <<'PY'
import os
from retrieval.strength_teacher import load_teacher_dataset

data, _ = load_teacher_dataset(os.environ["TEACHER_NPZ"], for_training=True)
train = data["controller_split_ids"] == 0
positive = float(data["gate_targets"][train].sum())
negative = float(train.sum() - positive)
if positive <= 0 or negative <= 0:
    raise RuntimeError("Decoupled gate training requires both classes")
print(negative / positive)
PY
)}"
echo "Train-only gate positive weight: ${GATE_POS_WEIGHT}"
echo "Gate loss weight: ${LAMBDA_GATE}"

train_variant() {
  local name="$1"
  local feature_mode="$2"
  shift 2
  local output_dir="${OUTPUT_ROOT}/${name}"
  local log_path="${LOG_DIR}/${name}.log"
  if [[ "${FORCE_RETRAIN}" != "1" && -s "${output_dir}/best.pt" ]]; then
    echo "[reuse] ${name}: ${output_dir}/best.pt"
    return
  fi
  echo "[train] ${name}"
  python -u tools/train_strength_controller.py \
    --teacher-npz "${TEACHER_NPZ}" \
    --output-dir "${output_dir}" \
    --feature-mode "${feature_mode}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --gate-pos-weight "${GATE_POS_WEIGHT}" \
    --lambda-gate "${LAMBDA_GATE}" \
    --separate-task-towers \
    "$@" \
    2>&1 | tee "${log_path}"
}

echo "[1/2] Train six decoupled controller variants"
train_variant score_only_prior_bounded score_only \
  --max-residual 0.15 --lambda-monotonic 0.2
train_variant score_plus_pair_prior_bounded score_plus_pair \
  --max-residual 0.15 --lambda-monotonic 0.2
train_variant score_only_prior_wide score_only \
  --max-residual 0.75 --lambda-monotonic 0.0
train_variant score_plus_pair_prior_wide score_plus_pair \
  --max-residual 0.75 --lambda-monotonic 0.0
train_variant score_only_direct score_only \
  --lambda-monotonic 0.0 --lambda-residual 0.0 --direct-strength-head
train_variant score_plus_pair_direct score_plus_pair \
  --lambda-monotonic 0.0 --lambda-residual 0.0 --direct-strength-head

echo "[2/2] Compare decoupled variants"
variants=(
  score_only_prior_bounded
  score_plus_pair_prior_bounded
  score_only_prior_wide
  score_plus_pair_prior_wide
  score_only_direct
  score_plus_pair_direct
)
compare_args=()
for name in "${variants[@]}"; do
  compare_args+=(--controller "${name}=${OUTPUT_ROOT}/${name}")
done
python tools/compare_strength_controllers.py \
  --teacher-npz "${TEACHER_NPZ}" \
  "${compare_args[@]}" \
  --output-dir "${ANALYSIS_ROOT}/comparison" \
  2>&1 | tee "${LOG_DIR}/variant_comparison.log"

echo "============================================================"
echo "Decoupled adaptive pilot completed successfully."
echo "Comparison: ${ANALYSIS_ROOT}/comparison/controller_variant_comparison.json"
echo "No diffusion generation or dataset validation/test evaluation was run."
echo "============================================================"
