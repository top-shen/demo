#!/usr/bin/env bash
set -euo pipefail

# Final train-only feasibility check for adaptive strength. The expensive
# Teacher build is deliberately opt-in and resumable. No dataset valid/test
# evaluation is performed.
MAX_QUERIES="${MAX_QUERIES:-1024}"
SAMPLES_PER_ACTION="${SAMPLES_PER_ACTION:-3}"
GENERATION_SEED="${GENERATION_SEED:-42}"
GENERATION_DEVICE="${GENERATION_DEVICE:-cuda:0}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-${GENERATION_DEVICE}}"
CONTROLLER_DEVICE="${CONTROLLER_DEVICE:-cpu}"
CONTROLLER_SEEDS="${CONTROLLER_SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LAMBDA_GATE="${LAMBDA_GATE:-5.0}"
COPY_QUANTILE="${COPY_QUANTILE:-0.05}"
COPY_PAIRS="${COPY_PAIRS:-8192}"
EPSILON_RELATIVE_ORIGINAL="${EPSILON_RELATIVE_ORIGINAL:-0.01}"
FORCE_TEACHER_BUILD="${FORCE_TEACHER_BUILD:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

RUN_TAG="${RUN_TAG:-stable_q${MAX_QUERIES}_spa${SAMPLES_PER_ACTION}_seed${GENERATION_SEED}}"
LABEL_TAG="${LABEL_TAG:-q05_eps1pct}"
TEACHER_ROOT="${TEACHER_ROOT:-./cache/adaptive/synth-m}"
RAW_TEACHER="${RAW_TEACHER:-${TEACHER_ROOT}/run0_teacher_${RUN_TAG}_raw.npz}"
RAW_MANIFEST="${RAW_MANIFEST:-${RAW_TEACHER%.npz}.json}"
RELABEL_TEACHER="${RELABEL_TEACHER:-${TEACHER_ROOT}/run0_teacher_${RUN_TAG}_${LABEL_TAG}.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./save/adaptive_controller/synth-m/${RUN_TAG}_${LABEL_TAG}}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-./save/adaptive_pilot_analysis/synth-m/${RUN_TAG}_${LABEL_TAG}}"
LOG_DIR="${LOG_DIR:-./logs/${RUN_TAG}_${LABEL_TAG}}"
TRAIN_TS="${TRAIN_TS:-./datasets/synth-m/train_ts.npy}"

mkdir -p "${TEACHER_ROOT}" "${OUTPUT_ROOT}" "${ANALYSIS_ROOT}" "${LOG_DIR}"

if [[ "${FORCE_TEACHER_BUILD}" == "1" \
      || ! -s "${RAW_TEACHER}" \
      || ! -s "${RAW_MANIFEST}" ]]; then
  echo "[1/6] Build/resume stable Teacher candidates"
  echo "Expected diffusion trajectories: $((MAX_QUERIES * 7 * SAMPLES_PER_ACTION))"
  bash scripts/synth-m/build_adaptive_teacher_pilot.sh \
    --max-queries "${MAX_QUERIES}" \
    --samples-per-action "${SAMPLES_PER_ACTION}" \
    --seed "${GENERATION_SEED}" \
    --device "${GENERATION_DEVICE}" \
    --embedding-device "${EMBEDDING_DEVICE}" \
    --epsilon-sem 0.01 \
    --epsilon-sem-source "provisional-before-offline-relabel" \
    --copy-threshold 0.05 \
    --copy-threshold-source "provisional-before-offline-relabel" \
    --output-npz "${RAW_TEACHER}" \
    --resume \
    2>&1 | tee "${LOG_DIR}/teacher_build.log"
else
  echo "[1/6] Reuse complete raw Teacher: ${RAW_TEACHER}"
fi

echo "[2/6] Relabel with train-only q05 copy threshold and 1% CTTP margin"
python -u tools/relabel_strength_teacher.py \
  --input-npz "${RAW_TEACHER}" \
  --output-npz "${RELABEL_TEACHER}" \
  --epsilon-relative-original "${EPSILON_RELATIVE_ORIGINAL}" \
  --epsilon-source "1pct-of-stable-train-teacher-original-cttp" \
  --copy-quantile "${COPY_QUANTILE}" \
  --train-ts-path "${TRAIN_TS}" \
  --copy-num-pairs "${COPY_PAIRS}" \
  --copy-seed 2025 \
  --copy-source "train-only-random-different-series-pair-quantile-${COPY_QUANTILE}" \
  2>&1 | tee "${LOG_DIR}/teacher_relabel.log"

echo "[3/6] Validate train-only leakage protection and class balance"
TEACHER_NPZ="${RELABEL_TEACHER}" python - <<'PY'
import os
import numpy as np
from retrieval.strength_teacher import load_teacher_dataset

data, manifest = load_teacher_dataset(os.environ["TEACHER_NPZ"], for_training=True)
if manifest.get("split") != "train":
    raise RuntimeError("Stable Teacher must be train-only")
if np.any(data["query_sample_ids"] == data["reference_sample_ids"]):
    raise RuntimeError("Stable Teacher contains same-sample retrieval leakage")
gate = data["gate_targets"]
split = data["controller_split_ids"]
print("teacher rows:", len(gate))
for split_id, name in ((0, "controller-train"), (1, "controller-validation")):
    mask = split == split_id
    print(
        name,
        "rows/positive/negative:",
        int(mask.sum()),
        int(gate[mask].sum()),
        int((gate[mask] == 0).sum()),
    )
print("same-sample retrieval: 0")
print("stable Teacher validation: PASS")
PY

GATE_POS_WEIGHT="${GATE_POS_WEIGHT:-$(TEACHER_NPZ="${RELABEL_TEACHER}" python - <<'PY'
import os
from retrieval.strength_teacher import load_teacher_dataset

data, _ = load_teacher_dataset(os.environ["TEACHER_NPZ"], for_training=True)
train = data["controller_split_ids"] == 0
positive = float(data["gate_targets"][train].sum())
negative = float(train.sum() - positive)
if positive <= 0 or negative <= 0:
    raise RuntimeError("Go/no-go training requires both gate classes")
print(negative / positive)
PY
)}"
echo "Explicit train-only gate positive weight: ${GATE_POS_WEIGHT}"

train_one() {
  local name="$1"
  local feature_mode="$2"
  local seed="$3"
  local shuffled="$4"
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
    --teacher-npz "${RELABEL_TEACHER}" \
    --output-dir "${output_dir}" \
    --feature-mode "${feature_mode}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${CONTROLLER_DEVICE}" \
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

echo "[4/6] Train pair, score-only, and shuffled controls for three seeds"
variants=()
for seed in ${CONTROLLER_SEEDS}; do
  pair_name="pair_direct_seed${seed}"
  score_name="score_only_direct_seed${seed}"
  shuffled_name="pair_direct_shuffled_seed${seed}"
  train_one "${pair_name}" score_plus_pair "${seed}" 0
  train_one "${score_name}" score_only "${seed}" 0
  train_one "${shuffled_name}" score_plus_pair "${seed}" 1
  variants+=("${pair_name}" "${score_name}" "${shuffled_name}")
done

echo "[5/6] Compare all controller variants"
compare_args=()
for name in "${variants[@]}"; do
  compare_args+=(--controller "${name}=${OUTPUT_ROOT}/${name}")
done
python tools/compare_strength_controllers.py \
  --teacher-npz "${RELABEL_TEACHER}" \
  "${compare_args[@]}" \
  --output-dir "${ANALYSIS_ROOT}/comparison" \
  2>&1 | tee "${LOG_DIR}/variant_comparison.log"

echo "[6/6] Produce explicit train-only go/no-go summary"
python tools/summarize_adaptive_go_no_go.py \
  --comparison-json "${ANALYSIS_ROOT}/comparison/controller_variant_comparison.json" \
  --output-json "${ANALYSIS_ROOT}/go_no_go.json" \
  --teacher-npz "${RELABEL_TEACHER}" \
  --max-queries "${MAX_QUERIES}" \
  --samples-per-action "${SAMPLES_PER_ACTION}" \
  --generation-seed "${GENERATION_SEED}" \
  --controller-seeds "${CONTROLLER_SEEDS}" \
  2>&1 | tee "${LOG_DIR}/go_no_go.log"

echo "============================================================"
echo "Stable Teacher adaptive-strength experiment completed."
echo "Raw Teacher:       ${RAW_TEACHER}"
echo "Relabeled Teacher: ${RELABEL_TEACHER}"
echo "Comparison:        ${ANALYSIS_ROOT}/comparison/controller_variant_comparison.json"
echo "Go/no-go summary:  ${ANALYSIS_ROOT}/go_no_go.json"
echo "No dataset validation/test evaluation was run."
echo "============================================================"
