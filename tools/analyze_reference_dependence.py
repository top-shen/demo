"""Quantify and visualize how strongly RI-VerbalTS follows its reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.reference_dependence import analyze_reference_dependence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-predictions", required=True)
    parser.add_argument("--retrieval-trace", required=True)
    parser.add_argument("--retrieval-index", required=True)
    parser.add_argument("--baseline-predictions", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--copy-threshold",
        type=float,
        default=0.05,
        help="Operational near-copy threshold in train-standardized RMSE units.",
    )
    parser.add_argument("--cases-per-group", type=int, default=2)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--max-line-variables", type=int, default=6)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    result = analyze_reference_dependence(
        rag_predictions_path=args.rag_predictions,
        retrieval_trace_path=args.retrieval_trace,
        retrieval_index_path=args.retrieval_index,
        baseline_predictions_path=args.baseline_predictions or None,
        output_dir=args.output_dir,
        copy_threshold=args.copy_threshold,
        cases_per_group=args.cases_per_group,
        selection_seed=args.selection_seed,
        max_line_variables=args.max_line_variables,
        make_plots=not args.no_plots,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Reference-dependence analysis saved to {result['output_dir']}")


if __name__ == "__main__":
    main()
