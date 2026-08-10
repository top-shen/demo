#!/usr/bin/env bash
set -euo pipefail

# Offline relabel + lightweight controller ablations. No diffusion generation
# and no dataset validation/test split are used by this diagnostic pipeline.
SOURCE_TEACHER="${SOURCE_TEACHER:-./cache/adaptive/synth-m/run0_teacher_medium_q256_seed42.npz}"
TRAIN_TS="${TRAIN_TS:-./datasets/synth-m/train_ts.npy}"
RELABEL_TAG="${RELABEL_TAG:-q05_eps1pct}"
RELABEL_TEACHER="${RELABEL_TEACHER:-./cache/adaptive/synth-m/run0_teacher_medium_q256_seed42_${RELABEL_TAG}.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./save/adaptive_controller/synth-m/repair_${RELABEL_TAG}}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-./save/adaptive_pilot_analysis/synth-m/repair_${RELABEL_TAG}}"
LOG_DIR="${LOG_DIR:-./logs/repair_${RELABEL_TAG}}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
COPY_QUANTILE="${COPY_QUANTILE:-0.05}"
COPY_PAIRS="${COPY_PAIRS:-8192}"
EPSILON_RELATIVE_ORIGINAL="${EPSILON_RELATIVE_ORIGINAL:-0.01}"
LAMBDA_GATE="${LAMBDA_GATE:-1.0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

if [[ ! -f "${SOURCE_TEACHER}" ]]; then
  echo "Source Teacher not found: ${SOURCE_TEACHER}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_TS}" ]]; then
  echo "Training time-series file not found: ${TRAIN_TS}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}" "${ANALYSIS_ROOT}" "${LOG_DIR}"

echo "[1/4] Relabel existing Teacher without rerunning diffusion"
python -u tools/relabel_strength_teacher.py \
  --input-npz "${SOURCE_TEACHER}" \
  --output-npz "${RELABEL_TEACHER}" \
  --epsilon-relative-original "${EPSILON_RELATIVE_ORIGINAL}" \
  --epsilon-source "1pct-of-train-pilot-original-cttp" \
  --copy-quantile "${COPY_QUANTILE}" \
  --train-ts-path "${TRAIN_TS}" \
  --copy-num-pairs "${COPY_PAIRS}" \
  --copy-seed 2025 \
  --copy-source "train-only-random-different-series-pair-quantile" \
  2>&1 | tee "${LOG_DIR}/teacher_relabel.log"

GATE_POS_WEIGHT="${GATE_POS_WEIGHT:-$(RELABEL_TEACHER="${RELABEL_TEACHER}" python - <<'PY'
import os
from retrieval.strength_teacher import load_teacher_dataset

data, _ = load_teacher_dataset(os.environ["RELABEL_TEACHER"], for_training=True)
train = data["controller_split_ids"] == 0
positive = float(data["gate_targets"][train].sum())
negative = float(train.sum() - positive)
if positive <= 0 or negative <= 0:
    raise RuntimeError("Balanced gate training requires both classes")
print(negative / positive)
PY
)}"
echo "Explicit train-only gate positive weight: ${GATE_POS_WEIGHT}"

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
    --teacher-npz "${RELABEL_TEACHER}" \
    --output-dir "${output_dir}" \
    --feature-mode "${feature_mode}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --lambda-gate "${LAMBDA_GATE}" \
    --gate-pos-weight "${GATE_POS_WEIGHT}" \
    "$@" \
    2>&1 | tee "${log_path}"
}

echo "[2/4] Train prior/monotonic diagnostic variants"
train_variant score_only_prior_mono score_only --lambda-monotonic 0.2
train_variant score_plus_pair_prior_mono score_plus_pair --lambda-monotonic 0.2
train_variant score_only_prior_no_mono score_only --lambda-monotonic 0.0
train_variant score_plus_pair_prior_no_mono score_plus_pair --lambda-monotonic 0.0
train_variant score_only_direct_no_mono score_only \
  --lambda-monotonic 0.0 --lambda-residual 0.0 --direct-strength-head
train_variant score_plus_pair_direct_no_mono score_plus_pair \
  --lambda-monotonic 0.0 --lambda-residual 0.0 --direct-strength-head

COPY_THRESHOLD="$(RELABEL_TEACHER="${RELABEL_TEACHER}" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["RELABEL_TEACHER"]).with_suffix(".json")
print(json.loads(path.read_text())["copy_constraint"]["threshold"])
PY
)"
FIXED_SWEEP="${RELABEL_TEACHER%.npz}_fixed_sweep.json"

echo "[3/4] Analyze every controller variant"
variants=(
  score_only_prior_mono
  score_plus_pair_prior_mono
  score_only_prior_no_mono
  score_plus_pair_prior_no_mono
  score_only_direct_no_mono
  score_plus_pair_direct_no_mono
)
for name in "${variants[@]}"; do
  python tools/analyze_strength_controller.py \
    --teacher-npz "${RELABEL_TEACHER}" \
    --controller-predictions "${OUTPUT_ROOT}/${name}/validation_predictions.npz" \
    --fixed-sweep "${FIXED_SWEEP}" \
    --copy-threshold "${COPY_THRESHOLD}" \
    --output-dir "${ANALYSIS_ROOT}/${name}"
done

echo "[4/4] Compare learned, prior, and constant baselines"
compare_args=()
for name in "${variants[@]}"; do
  compare_args+=(--controller "${name}=${OUTPUT_ROOT}/${name}")
done
python tools/compare_strength_controllers.py \
  --teacher-npz "${RELABEL_TEACHER}" \
  "${compare_args[@]}" \
  --output-dir "${ANALYSIS_ROOT}/comparison" \
  2>&1 | tee "${LOG_DIR}/variant_comparison.log"

echo "============================================================"
echo "Adaptive repair pilot completed successfully."
echo "Relabeled Teacher: ${RELABEL_TEACHER}"
echo "Controllers:       ${OUTPUT_ROOT}"
echo "Comparison:        ${ANALYSIS_ROOT}/comparison/controller_variant_comparison.json"
echo "No diffusion generation or dataset test evaluation was run."
echo "============================================================"
