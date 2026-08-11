#!/usr/bin/env bash
set -euo pipefail

# Three initialization seeds plus shuffled-feature negative controls for the
# selected decoupled score-plus-pair direct-head controller. No diffusion or
# dataset validation/test evaluation is performed.
TEACHER_NPZ="${TEACHER_NPZ:-./cache/adaptive/synth-m/run0_teacher_medium_q256_seed42_q05_eps1pct.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./save/adaptive_controller/synth-m/robustness_q05_eps1pct}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-./save/adaptive_pilot_analysis/synth-m/robustness_q05_eps1pct}"
LOG_DIR="${LOG_DIR:-./logs/robustness_q05_eps1pct}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LAMBDA_GATE="${LAMBDA_GATE:-5.0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
SEEDS="${SEEDS:-42 43 44}"

if [[ ! -f "${TEACHER_NPZ}" ]]; then
  echo "Relabeled Teacher not found: ${TEACHER_NPZ}" >&2
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
    raise RuntimeError("Robustness pilot requires both gate classes")
print(negative / positive)
PY
)}"
echo "Train-only gate positive weight: ${GATE_POS_WEIGHT}"

train_one() {
  local name="$1"
  local seed="$2"
  local shuffled="$3"
  local output_dir="${OUTPUT_ROOT}/${name}"
  local log_path="${LOG_DIR}/${name}.log"
  if [[ "${FORCE_RETRAIN}" != "1" && -s "${output_dir}/best.pt" ]]; then
    echo "[reuse] ${name}: ${output_dir}/best.pt"
    return
  fi
  extra_args=()
  if [[ "${shuffled}" == "1" ]]; then
    extra_args+=(--shuffle-retrieval-features)
  fi
  echo "[train] ${name}"
  python -u tools/train_strength_controller.py \
    --teacher-npz "${TEACHER_NPZ}" \
    --output-dir "${output_dir}" \
    --feature-mode score_plus_pair \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --seed "${seed}" \
    --gate-pos-weight "${GATE_POS_WEIGHT}" \
    --lambda-gate "${LAMBDA_GATE}" \
    --lambda-monotonic 0.0 \
    --lambda-residual 0.0 \
    --direct-strength-head \
    --separate-task-towers \
    ${extra_args[@]+"${extra_args[@]}"} \
    2>&1 | tee "${log_path}"
}

variants=()
for seed in ${SEEDS}; do
  real_name="pair_direct_seed${seed}"
  shuffled_name="pair_direct_shuffled_seed${seed}"
  train_one "${real_name}" "${seed}" 0
  train_one "${shuffled_name}" "${seed}" 1
  variants+=("${real_name}" "${shuffled_name}")
done

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
echo "Adaptive robustness pilot completed successfully."
echo "Comparison: ${ANALYSIS_ROOT}/comparison/controller_variant_comparison.json"
echo "No diffusion generation or dataset validation/test evaluation was run."
echo "============================================================"
