#!/usr/bin/env bash
set -euo pipefail

# Strict validation-only diagnostic. This script never invokes run.py,
# diffusion generation, training, Teacher construction, or the test split.
VALIDATION_ROOT="${VALIDATION_ROOT:-./save/adaptive_validation/synth-m/stable_q1024_spa3}"
TEACHER_MANIFEST="${TEACHER_MANIFEST:-./cache/adaptive/synth-m/run0_teacher_stable_q1024_spa3_seed42_q05_eps1pct.json}"
VALIDATION_DECISION="${VALIDATION_DECISION:-${VALIDATION_ROOT}/validation_decision.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${VALIDATION_ROOT}/oracle_ceiling}"
DATASET_FOLDER="${DATASET_FOLDER:-./datasets/synth-m}"
RUNS="${RUNS:-3}"
DEVICE="${DEVICE:-cuda:0}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-128}"
METRIC_AUDIT_ATOL="${METRIC_AUDIT_ATOL:-0.001}"
CTTP_RUNTIME_MODE="${CTTP_RUNTIME_MODE:-legacy_train}"
SCORER_REPEATS="${SCORER_REPEATS:-3}"
SCORER_SEED="${SCORER_SEED:-2026}"
STOCHASTIC_METRIC_RTOL="${STOCHASTIC_METRIC_RTOL:-0.05}"
THRESHOLD_CALIBRATION="${THRESHOLD_CALIBRATION:-}"

for path in \
  "${DATASET_FOLDER}/valid_ts.npy" \
  "${TEACHER_MANIFEST}" \
  "${VALIDATION_DECISION}" \
  "${VALIDATION_ROOT}/original/results.csv"; do
  if [[ ! -s "${path}" ]]; then
    echo "Required validation-only Oracle input is missing: ${path}" >&2
    exit 2
  fi
done

fixed_count=0
for directory in "${VALIDATION_ROOT}"/fixed_*; do
  if [[ -d "${directory}" && -s "${directory}/results.csv" ]]; then
    fixed_count=$((fixed_count + 1))
  fi
done
if [[ "${fixed_count}" -lt 2 ]]; then
  echo "At least two complete fixed-strength validation conditions are required." >&2
  exit 2
fi

echo "Running a validation-only policy-specific empirical ceiling."
echo "Saved pointwise-median predictions are read; candidate arrays are ignored."
echo "No generation, training, Teacher construction, or test evaluation will run."
echo "CTTP audit mode: ${CTTP_RUNTIME_MODE}; repeats=${SCORER_REPEATS}"
oracle_command=(
  python -u tools/analyze_oracle_ceiling.py
  --validation-root "${VALIDATION_ROOT}"
  --dataset-folder "${DATASET_FOLDER}"
  --dataset-split valid
  --teacher-manifest "${TEACHER_MANIFEST}"
  --validation-decision "${VALIDATION_DECISION}"
  --output-dir "${OUTPUT_DIR}"
  --expected-runs "${RUNS}"
  --device "${DEVICE}"
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
  --metric-audit-atol "${METRIC_AUDIT_ATOL}"
  --cttp-runtime-mode "${CTTP_RUNTIME_MODE}"
  --scorer-repeats "${SCORER_REPEATS}"
  --scorer-seed "${SCORER_SEED}"
  --stochastic-metric-rtol "${STOCHASTIC_METRIC_RTOL}"
)
if [[ -n "${THRESHOLD_CALIBRATION}" ]]; then
  oracle_command+=(--threshold-calibration "${THRESHOLD_CALIBRATION}")
fi
if [[ "${ALLOW_JOINT_REFERENCE_ORACLE:-0}" == "1" ]]; then
  oracle_command+=(--allow-joint-reference)
fi
"${oracle_command[@]}"

echo "Oracle decision: ${OUTPUT_DIR}/oracle_decision.json"
echo "Integrity audit: ${OUTPUT_DIR}/oracle_integrity_report.json"
echo "The test split was not read or evaluated."
