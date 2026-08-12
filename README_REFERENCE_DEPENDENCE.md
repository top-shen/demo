# RI-VerbalTS reference-dependence analysis

This offline analysis asks whether RI-VerbalTS uses a retrieved series as a
helpful initialization or copies it too strongly. It reads existing artifacts;
it does not load a generator, run diffusion, train a model, or modify results.

## Research question and evidence boundary

The primary question is:

```text
Does RAG+VerbalTS move the retrieved reference toward the query's ground-truth
series, while retaining nontrivial candidate diversity?
```

This first diagnostic measures association with the selected reference. It is
not a causal reference-sensitivity experiment because each query has only one
observed reference. A later reference-swap experiment is needed to estimate how
much output variation is caused by changing the reference while holding query
and diffusion noise fixed.

Ground truth is used only after generation for analysis. Retrieval remains
strictly train-only.

## Inputs

For each dataset/run, the analyzer consumes:

- RAG `rag_predictions.npz`;
- RAG `retrieval_trace.jsonl`;
- the train-only retrieval index, including its `train_ts` and stable IDs;
- optionally, the matching `rag_enabled=false` prediction artifact.

Old phase-1 artifacts and newer adaptive-controller artifacts are both
supported. Samples are aligned by sample ID, not by file row order.

## Distance normalization

Raw Weather variables can have very different scales. All reported `sRMSE` and
`sMAE` distances first divide each variable by its standard deviation computed
from the retrieval index's **training time series only**:

```text
sRMSE(a,b) = sqrt(mean(((a-b) / train_variable_std)^2))
```

This gives every variable comparable influence and does not use test statistics
for normalization.

## Main diagnostics

| Output column | Meaning |
|---|---|
| `rag_reference_srmse` | How far the aggregate RAG output moves from its reference; lower means stronger retention. |
| `rag_reference_correlation` | Mean per-variable temporal correlation with the reference; higher means more shape retention. |
| `reference_target_srmse` | Reference error relative to test ground truth. |
| `rag_target_srmse` | RAG output error relative to test ground truth. |
| `target_correction_gain` | `d(reference,target)-d(RAG,target)`; positive means correction toward target. |
| `target_correction_fraction` | Correction gain divided by reference error; 1 is perfect correction, 0 is none, negative is degradation. |
| `reference_vs_target_distance_ratio` | `d(RAG,reference)/d(RAG,target)`; below 1 means output is closer to reference than target. |
| `candidate_diversity_srmse` | Mean pairwise distance among all generated candidates. |
| `candidate_reference_srmse_mean` | Mean candidate-to-reference distance before median aggregation. |
| `near_reference_copy` | Operational flag when `d(RAG,reference) <= copy_threshold`. |
| `reference_attraction` | With baseline available: `1-d(RAG,reference)/d(VerbalTS,reference)`. Positive means RAG is closer to the reference than original VerbalTS. |
| `target_gain_vs_baseline` | `d(VerbalTS,target)-d(RAG,target)`; positive means RAG improves raw sequence error over baseline. |

The default near-copy threshold is `0.05` train-standardized RMSE units. It is
an explicit operational threshold, not a universal scientific constant. Report
the threshold alongside copy rate and preferably repeat with nearby thresholds.

## Case selection without visual cherry-picking

The tool deterministically selects unique cases from predefined strata:

1. highest retrieval similarity;
2. lowest retrieval similarity;
3. largest positive target correction;
4. largest negative target correction;
5. strongest reference attraction (or closest to reference when baseline is absent);
6. fixed-seed random cases.

By default, two cases are selected from each stratum. The selection table is
saved before plotting.

For Synth-M, every variable is plotted. For Weather, the figure includes
train-standardized heatmaps for all 21 variables plus line plots for the six
variables with the largest combined reference error and RAG movement. The ten
RAG candidates are shown as translucent red curves, with the aggregate RAG
output, reference, ground truth, and optional original VerbalTS overlaid.

## Run

After completing RAG checkpoint-0 evaluation and the matching RAG-disabled
baseline, run:

```bash
bash scripts/synth-m/analyze_reference_dependence.sh
bash scripts/Weather/analyze_reference_dependence.sh
```

Or both sequentially:

```bash
bash scripts/synth-m/analyze_reference_dependence.sh && \
bash scripts/Weather/analyze_reference_dependence.sh
```

If the baseline artifact is absent, the scripts still produce reference/target
metrics and case figures, but omit baseline-relative columns and curves.

Useful overrides:

```bash
# Select three cases per stratum and use a stricter operational copy threshold.
bash scripts/synth-m/analyze_reference_dependence.sh \
  --cases-per-group 3 --copy-threshold 0.02

# Produce only tables/JSON, without Matplotlib figures.
bash scripts/Weather/analyze_reference_dependence.sh --no-plots
```

## Outputs

Each run writes a `reference_dependence/` directory containing:

```text
reference_dependence_metrics.csv       # one row per test sample
reference_dependence_summary.json      # aggregate rates and quantiles
reference_dependence_report.md         # readable analysis report + provenance
similarity_strata_summary.csv           # low/medium/high similarity groups
selected_cases.csv                      # auditable visualization selection
figures/reference_dependence_summary.png
figures/case_*.png
```

The summary figure shows:

- retrieval similarity versus movement from reference;
- reference-to-target versus RAG-to-target error, with an identity line;
- distribution of target-correction gain;
- movement from reference versus candidate diversity.

## Interpretation

Evidence of useful reference initialization looks like:

```text
target_correction_gain > 0 for a large fraction of cases
candidate_diversity_srmse > 0
near_reference_copy_rate is low
RAG improves target error or CTTP over retrieval-only
```

Evidence of excessive reference dependence looks like:

```text
very small RAG-reference distance
very high RAG-reference correlation
low candidate diversity
high near-copy rate
little or negative target correction
large positive reference attraction relative to original VerbalTS
```

Do not infer dependence from low FID: retrieval-only outputs real training
series and therefore receives an artificially favorable distributional score.
The next causal diagnostic should swap Top-K/random references for the same
query while fixing diffusion noise.

## Lightweight verification

The metric/alignment/selection tests require NumPy and pandas but no GPU:

```bash
python tests/run_reference_dependence_tests.py
```

Figure generation additionally uses Matplotlib, already pinned in
`requirements.txt`.
