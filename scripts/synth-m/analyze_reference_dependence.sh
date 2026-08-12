#!/usr/bin/env bash
set -euo pipefail

RAG_PREDICTIONS="./save/synth-m_rag_eval/text2ts_msmdiffmv/0/rag_predictions.npz"
RAG_TRACE="./save/synth-m_rag_eval/text2ts_msmdiffmv/0/retrieval_trace.jsonl"
RAG_INDEX="./cache/rag/synth-m/train_longclip.npz"
BASELINE_PREDICTIONS="./save/synth-m_verbalts_run0_eval/text2ts_msmdiffmv/0/rag_predictions.npz"
OUTPUT_DIR="./save/synth-m_rag_eval/text2ts_msmdiffmv/0/reference_dependence"

COMMAND=(
    python tools/analyze_reference_dependence.py
    --rag-predictions "${RAG_PREDICTIONS}"
    --retrieval-trace "${RAG_TRACE}"
    --retrieval-index "${RAG_INDEX}"
)

if [[ -f "${BASELINE_PREDICTIONS}" ]]; then
    COMMAND+=(--baseline-predictions "${BASELINE_PREDICTIONS}")
else
    echo "Advisory: baseline predictions not found; baseline-relative metrics will be omitted."
fi

COMMAND+=(
    --output-dir "${OUTPUT_DIR}"
    --copy-threshold 0.05
    --cases-per-group 2
    --selection-seed 42
)

"${COMMAND[@]}" "$@"
