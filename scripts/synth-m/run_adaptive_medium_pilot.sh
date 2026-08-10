#!/usr/bin/env bash
set -euo pipefail

# Train-only medium development pilot. This script deliberately does not run
# dataset validation/test generation; model selection uses the grouped
# controller-validation rows stored in the teacher artifact.
MAX_QUERIES="${MAX_QUERIES:-256}"
SAMPLES_PER_ACTION="${SAMPLES_PER_ACTION:-1}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
EPSILON_SEM="${EPSILON_SEM:-0.01}"
COPY_THRESHOLD="${COPY_THRESHOLD:-0.05}"
RUN_TAG="${RUN_TAG:-medium_q${MAX_QUERIES}_seed${SEED}}"

TEACHER_ROOT="${TEACHER_ROOT:-./cache/adaptive/synth-m}"
CONTROLLER_ROOT="${CONTROLLER_ROOT:-./save/adaptive_controller/synth-m/${RUN_TAG}}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-./save/adaptive_pilot_analysis/synth-m/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-./logs/${RUN_TAG}}"
TEACHER_NPZ="${TEACHER_NPZ:-${TEACHER_ROOT}/run0_teacher_${RUN_TAG}.npz}"
TEACHER_MANIFEST="${TEACHER_MANIFEST:-${TEACHER_NPZ%.npz}.json}"
FIXED_SWEEP="${FIXED_SWEEP:-${TEACHER_NPZ%.npz}_fixed_sweep.json}"

mkdir -p "${TEACHER_ROOT}" "${CONTROLLER_ROOT}" "${ANALYSIS_ROOT}" "${LOG_DIR}"

echo "[1/4] Build/resume the train-only medium teacher (${MAX_QUERIES} queries)"
bash scripts/synth-m/build_adaptive_teacher_pilot.sh \
  --max-queries "${MAX_QUERIES}" \
  --samples-per-action "${SAMPLES_PER_ACTION}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --epsilon-sem "${EPSILON_SEM}" \
  --epsilon-sem-source "provisional-train-only-medium-pilot" \
  --copy-threshold "${COPY_THRESHOLD}" \
  --copy-threshold-source "provisional-train-only-medium-pilot" \
  --output-npz "${TEACHER_NPZ}" \
  --resume \
  2>&1 | tee "${LOG_DIR}/teacher_build.log"

echo "[2/4] Validate leakage protection and report class balance"
TEACHER_NPZ="${TEACHER_NPZ}" python - <<'PY'
import os
import numpy as np
from retrieval.strength_teacher import load_teacher_dataset

data, manifest = load_teacher_dataset(os.environ["TEACHER_NPZ"], for_training=True)
query_ids = data["query_sample_ids"]
reference_ids = data["reference_sample_ids"]
split_ids = data["controller_split_ids"]
gate = data["gate_targets"]
if np.any(query_ids == reference_ids):
    raise RuntimeError("Teacher contains same-sample retrieval leakage")
if manifest.get("split") != "train":
    raise RuntimeError("Medium teacher is not train-only")
print("teacher rows:", len(query_ids))
print("gate positive/negative:", int(gate.sum()), int((gate == 0).sum()))
for split_id, name in ((0, "controller-train"), (1, "controller-validation")):
    mask = split_ids == split_id
    print(
        name,
        "rows/positive/negative:",
        int(mask.sum()),
        int(gate[mask].sum()),
        int((gate[mask] == 0).sum()),
    )
print("same-sample retrieval: 0")
print("medium teacher validation: PASS")
PY

echo "[3/4] Train score_only and score_plus_pair"
score_only_checkpoint="${CONTROLLER_ROOT}/pilot_score_only/best.pt"
score_plus_pair_checkpoint="${CONTROLLER_ROOT}/pilot_score_plus_pair/best.pt"
if [[ "${FORCE_RETRAIN}" != "1" \
      && -s "${score_only_checkpoint}" \
      && -s "${score_plus_pair_checkpoint}" ]]; then
  echo "Reusing existing medium-pilot controller checkpoints."
  echo "Set FORCE_RETRAIN=1 to train them again."
else
  TEACHER_NPZ="${TEACHER_NPZ}" \
  OUTPUT_ROOT="${CONTROLLER_ROOT}" \
  LOG_DIR="${LOG_DIR}" \
  DEVICE="${DEVICE}" \
  EPOCHS="${EPOCHS}" \
  PATIENCE="${PATIENCE}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  SEED="${SEED}" \
  bash scripts/synth-m/train_adaptive_pilot_both.sh
fi

echo "[4/4] Analyze grouped controller-validation predictions"
for feature_mode in score_only score_plus_pair; do
  output_dir="${ANALYSIS_ROOT}/${feature_mode}"
  python tools/analyze_strength_controller.py \
    --teacher-npz "${TEACHER_NPZ}" \
    --teacher-manifest "${TEACHER_MANIFEST}" \
    --controller-predictions \
      "${CONTROLLER_ROOT}/pilot_${feature_mode}/validation_predictions.npz" \
    --fixed-sweep "${FIXED_SWEEP}" \
    --copy-threshold "${COPY_THRESHOLD}" \
    --output-dir "${output_dir}"
done

echo "============================================================"
echo "Synth-M train-only medium pilot completed successfully."
echo "Teacher:    ${TEACHER_NPZ}"
echo "Controllers:${CONTROLLER_ROOT}/pilot_{score_only,score_plus_pair}"
echo "Analyses:   ${ANALYSIS_ROOT}/{score_only,score_plus_pair}"
echo "No dataset test evaluation was run."
echo "============================================================"
