#!/usr/bin/env bash
set -euo pipefail

# Dataset-validation-only benchmark after the stable Teacher GO decision.
# The test split is never requested. Completed conditions are reused.
VALIDATION_ROOT="${VALIDATION_ROOT:-./save/adaptive_validation/synth-m/stable_q1024_spa3}"
LOG_DIR="${LOG_DIR:-./logs/adaptive_validation_synth_m_stable_q1024_spa3}"
TEACHER_NPZ="${TEACHER_NPZ:-./cache/adaptive/synth-m/run0_teacher_stable_q1024_spa3_seed42_q05_eps1pct.npz}"
TEACHER_MANIFEST="${TEACHER_MANIFEST:-${TEACHER_NPZ%.npz}.json}"
CONTROLLER_ROOT="${CONTROLLER_ROOT:-./save/adaptive_controller/synth-m/stable_q1024_spa3_seed42_q05_eps1pct}"
HANDCRAFTED_ROOT="${HANDCRAFTED_ROOT:-${CONTROLLER_ROOT}/handcrafted_score_prior}"
RUNS="${RUNS:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_BATCHES="${MAX_BATCHES:--1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

PAIR_CHECKPOINT="${PAIR_CHECKPOINT:-${CONTROLLER_ROOT}/pair_direct_seed42/best.pt}"
SCORE_CHECKPOINT="${SCORE_CHECKPOINT:-${CONTROLLER_ROOT}/score_only_direct_seed42/best.pt}"
SHUFFLED_CHECKPOINT="${SHUFFLED_CHECKPOINT:-${CONTROLLER_ROOT}/pair_direct_shuffled_seed42/best.pt}"
HANDCRAFTED_CHECKPOINT="${HANDCRAFTED_CHECKPOINT:-${HANDCRAFTED_ROOT}/best.pt}"

for path in \
  ./datasets/synth-m/valid_ts.npy \
  "${TEACHER_NPZ}" \
  "${TEACHER_MANIFEST}" \
  "${PAIR_CHECKPOINT}" \
  "${SCORE_CHECKPOINT}" \
  "${SHUFFLED_CHECKPOINT}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Required validation input is missing: ${path}" >&2
    exit 2
  fi
done
mkdir -p "${VALIDATION_ROOT}" "${LOG_DIR}" "${HANDCRAFTED_ROOT}"

if [[ ! -s "${HANDCRAFTED_CHECKPOINT}" ]]; then
  echo "[setup] Export locked handcrafted similarity-prior checkpoint"
  python -u tools/train_strength_controller.py \
    --teacher-npz "${TEACHER_NPZ}" \
    --output-dir "${HANDCRAFTED_ROOT}" \
    --feature-mode score_only \
    --max-residual 0.0 \
    --disable-gate-loss \
    --lambda-monotonic 0.0 \
    --lambda-residual 0.0 \
    --epochs 1 \
    --patience 1 \
    --batch-size 128 \
    --device cpu \
    --seed 42 \
    2>&1 | tee "${LOG_DIR}/handcrafted_checkpoint.log"
fi

run_rag_condition() {
  local name="$1"
  shift
  local output_dir="${VALIDATION_ROOT}/${name}"
  if [[ "${FORCE_RERUN}" != "1" && -s "${output_dir}/results.csv" ]]; then
    echo "[reuse] ${name}: ${output_dir}/results.csv"
    return
  fi
  echo "[validation] ${name}"
  bash scripts/synth-m/eval_rag.sh \
    --eval_split valid \
    --save_folder "${output_dir}" \
    --start_runid 0 \
    --n_runs "${RUNS}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_max_batches "${MAX_BATCHES}" \
    "$@" \
    2>&1 | tee "${LOG_DIR}/${name}.log"
}

run_adaptive_condition() {
  local name="$1"
  local checkpoint="$2"
  local feature_mode="$3"
  local max_residual="$4"
  local output_dir="${VALIDATION_ROOT}/${name}"
  if [[ "${FORCE_RERUN}" != "1" && -s "${output_dir}/results.csv" ]]; then
    echo "[reuse] ${name}: ${output_dir}/results.csv"
    return
  fi
  echo "[validation] ${name} (gate disabled)"
  bash scripts/synth-m/eval_adaptive.sh \
    --eval_split valid \
    --save_folder "${output_dir}" \
    --start_runid 0 \
    --n_runs "${RUNS}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_max_batches "${MAX_BATCHES}" \
    --rag_enabled true \
    --rag_mode diffusion \
    --rag_top_k 4 \
    --rag_selection top1 \
    --rag_adaptive_enabled true \
    --rag_controller_checkpoint_path "${checkpoint}" \
    --rag_controller_feature_mode "${feature_mode}" \
    --rag_controller_max_residual "${max_residual}" \
    --rag_controller_gate_threshold 0.0 \
    2>&1 | tee "${LOG_DIR}/${name}.log"
}

echo "[1/5] Original and Retrieval-only validation baselines"
run_rag_condition original \
  --rag_enabled false \
  --rag_save_predictions false
run_rag_condition retrieval_only \
  --rag_enabled true \
  --rag_mode retrieval_only \
  --rag_top_k 4 \
  --rag_selection top1 \
  --rag_adaptive_enabled false

echo "[2/5] Fixed-strength validation sweep"
for strength in 0.20 0.35 0.40 0.50 0.65 0.80 0.95; do
  label="${strength/./}"
  run_rag_condition "fixed_${label}" \
    --rag_enabled true \
    --rag_mode diffusion \
    --rag_top_k 4 \
    --rag_selection top1 \
    --rag_strength "${strength}" \
    --rag_adaptive_enabled false
done

echo "[3/5] Handcrafted and learned controllers (unsupported gate disabled)"
run_adaptive_condition handcrafted "${HANDCRAFTED_CHECKPOINT}" score_only 0.0
run_adaptive_condition learned_pair "${PAIR_CHECKPOINT}" score_plus_pair 0.15
run_adaptive_condition learned_score_only "${SCORE_CHECKPOINT}" score_only 0.15
run_adaptive_condition shuffled_pair "${SHUFFLED_CHECKPOINT}" score_plus_pair 0.15

echo "[4/5] Verify every condition used validation, never test"
VALIDATION_ROOT="${VALIDATION_ROOT}" RUNS="${RUNS}" python - <<'PY'
import os
from pathlib import Path

import yaml

root = Path(os.environ["VALIDATION_ROOT"])
expected_runs = int(os.environ["RUNS"])
for condition in sorted(path for path in root.iterdir() if path.is_dir()):
    if not (condition / "results.csv").is_file():
        continue
    for run in range(expected_runs):
        path = condition / str(run) / "eval_configs.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        split = config.get("eval", {}).get("split")
        if split != "valid":
            raise RuntimeError(f"Non-validation artifact detected: {path} split={split}")
print("validation split audit: PASS")
PY

echo "[5/5] Select validation-best fixed strength and compare controls"
python tools/summarize_adaptive_validation.py \
  --validation-root "${VALIDATION_ROOT}" \
  --teacher-manifest "${TEACHER_MANIFEST}" \
  --expected-runs "${RUNS}" \
  --cttp-margin 0.01 \
  --output-json "${VALIDATION_ROOT}/validation_decision.json" \
  --output-csv "${VALIDATION_ROOT}/validation_summary.csv" \
  2>&1 | tee "${LOG_DIR}/validation_summary.log"

echo "============================================================"
echo "Synth-M adaptive validation completed."
echo "Decision: ${VALIDATION_ROOT}/validation_decision.json"
echo "Summary:  ${VALIDATION_ROOT}/validation_summary.csv"
echo "The test split was not evaluated."
echo "============================================================"
