#!/usr/bin/env bash
set -euo pipefail

RAG_PREDICTIONS="./save/Weather_rag_eval/text2ts_msmdiffmv/0/rag_predictions.npz"
RAG_TRACE="./save/Weather_rag_eval/text2ts_msmdiffmv/0/retrieval_trace.jsonl"
RAG_INDEX="./cache/rag/Weather/train_longclip.npz"
BASELINE_PREDICTIONS="./save/Weather_verbalts_run0_eval/text2ts_msmdiffmv/0/rag_predictions.npz"
OUTPUT_DIR="./save/Weather_rag_eval/text2ts_msmdiffmv/0/reference_dependence"

BASELINE_ARGS=()
if [[ -f "${BASELINE_PREDICTIONS}" ]]; then
    BASELINE_ARGS=(--baseline-predictions "${BASELINE_PREDICTIONS}")
else
    echo "Advisory: baseline predictions not found; baseline-relative metrics will be omitted."
fi

python tools/analyze_reference_dependence.py \
    --rag-predictions "${RAG_PREDICTIONS}" \
    --retrieval-trace "${RAG_TRACE}" \
    --retrieval-index "${RAG_INDEX}" \
    "${BASELINE_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --copy-threshold 0.05 \
    --cases-per-group 2 \
    --selection-seed 42 \
    "$@"
