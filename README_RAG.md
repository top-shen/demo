# Retrieval-Initialized VerbalTS (RI-VerbalTS), phase 1

> Phase 2 的可学习逐样本 strength/gate、train-only teacher 构建、配置与实验协议见 [README_ADAPTIVE_CONTROLLER.md](README_ADAPTIVE_CONTROLLER.md)。本页继续保留 phase-1 fixed-strength RAG 定义。

This module adds inference-time, text-to-text retrieval initialization to the
released VerbalTS generator. It does not change the training loss, add a
reference encoder, perform retrieval-aware fine-tuning, or implement CTTP
cross-modal retrieval.

## 1. Method

For a test caption, RI-VerbalTS embeds the query with frozen LongCLIP pooled
text features, L2-normalizes it, and retrieves the most similar training
captions by cosine similarity. Each caption row maps back to its paired
training time series. Synth-M contributes one caption row per series; Weather
contributes three caption rows per series, all mapped to the same series.

Given reference series `x_ref` and diffusion step `t`, initialization reuses
the repository's `DDPMSampler.forward`:

```text
x_t = sqrt(alpha_bar_t) * x_ref + sqrt(1 - alpha_bar_t) * epsilon
```

VerbalTS then reverses steps `t, t-1, ..., 0`. The conditioning caption in
every reverse step is the original query caption, never the retrieved caption.
If the best similarity is below `rag_min_similarity`, that test row falls back
to the original Gaussian initialization and full reverse trajectory.

`rag_strength` maps linearly to the discrete step index:

```text
start_step = round(rag_strength * (num_steps - 1))
```

Thus strength 0 maps to step 0 (maximum reference retention under the existing
nonzero first beta), and strength 1 maps to `num_steps - 1` (closest to the
original pure-noise path). An explicit nonnegative `rag_start_step` overrides
strength. Values outside the valid range fail early.

## 2. Build train-only indexes

LongCLIP must be present at `save/Longclip`. Build each index from the dataset's
`train_text_caps.npy` and `train_ts.npy` only:

```bash
python tools/build_retrieval_index.py \
  --dataset-folder ./datasets/synth-m \
  --dataset-name synth-m \
  --model-path ./save/Longclip \
  --output ./cache/rag/synth-m/train_longclip.npz \
  --batch-size 64 --device auto

python tools/build_retrieval_index.py \
  --dataset-folder ./datasets/Weather \
  --dataset-name Weather \
  --model-path ./save/Longclip \
  --output ./cache/rag/Weather/train_longclip.npz \
  --batch-size 64 --device auto
```

Equivalent scripts are `scripts/synth-m/build_rag_index.sh` and
`scripts/Weather/build_rag_index.sh`. The versioned NPZ contains normalized
embeddings, sample IDs, caption IDs, captions, training series stored once per
sample, dataset/model identity, split=`train`, and build parameters. Loading
rejects indexes that do not declare the training split or normalized vectors.

## 3. Evaluate released checkpoints

The RAG scripts keep released checkpoints separate from new outputs:

```bash
bash scripts/synth-m/eval_rag.sh
bash scripts/Weather/eval_rag.sh
```

The key direct commands are the same as the existing evaluation commands, with
`--evaluate_config_path configs/<dataset>/evaluate_rag.yaml`, a new
`--save_folder`, and `--checkpoint_folder` pointing at the released run roots.
For example, Synth-M writes under
`save/synth-m_rag_eval/text2ts_msmdiffmv/` while loading checkpoints from
`save/synth-m_eval/text2ts_msmdiffmv/`.

Any RAG YAML value can be overridden without changing old commands:

```bash
# Top-K sampled retrieval at strength 0.6
bash scripts/synth-m/eval_rag.sh \
  --rag_top_k 4 --rag_selection sample --rag_temperature 0.2 \
  --rag_strength 0.6 --rag_seed 42
```

When launching `run.py` directly, use the same additional flags after the
arguments shown in the supplied script.

For a one-run, one-batch checkpoint smoke test before full evaluation:

```bash
bash scripts/synth-m/eval_rag.sh \
  --start_runid 0 --n_runs 1 --batch_size 4 --eval_max_batches 1
```

`eval_max_batches` defaults to `-1`, so existing and full RAG evaluations still
consume the complete test split.

## 4. Configuration

All RAG fields live below `eval.rag`; CLI overrides use the `--rag_` prefix.

| YAML key | CLI flag | Meaning |
|---|---|---|
| `enabled` | `--rag_enabled` | Default false; no index/model is loaded when false. |
| `index_path` | `--rag_index_path` | Versioned train-only NPZ index. |
| `top_k` | `--rag_top_k` | Number of caption rows considered; supports 1, 4, 8 and other positive values. |
| `selection` | `--rag_selection` | `top1` or softmax `sample`. |
| `temperature` | `--rag_temperature` | Positive sampling temperature. |
| `seed` | `--rag_seed` | Deterministic reference-selection seed. |
| `min_similarity` | `--rag_min_similarity` | Best-score threshold for baseline fallback. |
| `start_step` | `--rag_start_step` | Explicit step; `-1` delegates to strength. |
| `strength` | `--rag_strength` | Float in `[0,1]`; supports the 0.2/0.4/0.6/0.8 grid. |
| `mode` | `--rag_mode` | `diffusion`, `retrieval_only`, or `random_reference`. |
| `diverse_reference` | `--rag_diverse_reference` | Re-select a reference per generated candidate. |
| `embedding_model_path` | `--rag_embedding_model_path` | Frozen LongCLIP directory. |
| `embedding_device` | `--rag_embedding_device` | `auto`, `cpu`, or a CUDA device. |
| `query_batch_size` | `--rag_query_batch_size` | LongCLIP query-encoding batch size. |
| `trace_path` | `--rag_trace_path` | Optional JSONL path; defaults inside each run output. |

`--rag_save_predictions` controls NPZ output and `--rag_prediction_path` can
override its location. RAG evaluation configs enable prediction saving by
default. Old YAML files and old commands leave RAG disabled.

## 5. Outputs

Each run writes:

- `retrieval_trace.jsonl`: test/query IDs and caption, selected reference IDs
  and caption, similarity, Top-K IDs/scores, step/strength, selection, seed,
  candidate ID, mode, and fallback status. It does not duplicate time series.
- `rag_predictions.npz`: aggregate prediction, every generated candidate,
  target series, test/caption IDs, query captions, and candidate-level selected
  reference IDs.
- Existing `results.csv` metrics, including FID, J-FTSD, and CTTP.

Together with the index's `train_ts` and trace mappings, these arrays support:

- generated-to-reference distance;
- nearest-training-series distance;
- a threshold-defined reference copy rate;
- pairwise candidate diversity;
- retrieval score distributions and reference usage frequencies.

The implemented post-hoc protocol and one-command visualization scripts are
documented in [README_REFERENCE_DEPENDENCE.md](README_REFERENCE_DEPENDENCE.md).

Distance/copy thresholds and normalization conventions are intentionally left
to the analysis protocol in phase 1; raw candidates and stable IDs are retained
so those definitions can be changed without rerunning diffusion.

## 6. Experiment groups

| Group | Settings |
|---|---|
| A. Original VerbalTS | `rag_enabled=false` |
| B. Retrieval-only | `rag_enabled=true`, `rag_mode=retrieval_only` |
| C. Random reference + diffusion | `rag_mode=random_reference` |
| D. Top-1 fixed step | `rag_selection=top1`, `rag_top_k=1`, set step/strength |
| E. Top-K sampled fixed step | `rag_selection=sample`, `rag_top_k=4` or `8` |

For grid experiments, override `rag_top_k` with 1/4/8 and `rag_strength` with
0.2/0.4/0.6/0.8. All random selection is derived from `rag_seed`, test sample
ID, and candidate ID, so it is stable across evaluator batch sizes.

## 7. Data leakage guard

Never build an index by concatenating split files. The builder has no split
argument and opens only files named `train_text_caps.npy` and `train_ts.npy`.
The serialized metadata records `split: train`, and the loader refuses another
value. The tests also place unique valid/test captions next to training files
and verify that neither enters the index.

## 8. Multiple candidates and median aggregation

The default retrieves one fixed reference per test condition, then draws
different diffusion noise for all `n_samples`. The original pointwise median is
therefore retained for paper-compatible evaluation.

With `diverse_reference=true`, every candidate may sample a different Top-K
reference. Pointwise median across unrelated local structures is not used:
candidate 0 is used for the aggregate FID/J-FTSD/CTTP row, and all candidates
plus their reference IDs are saved for diversity analysis. Report this mode
separately from the paper-compatible fixed-reference median.

## 9. Lightweight validation

The dependency-light retrieval checks use fake sentence embeddings and tiny
arrays; they do not load PyTorch, LongCLIP, a checkpoint, or a GPU:

```bash
python tests/run_numpy_retrieval_tests.py
python tests/run_adaptive_numpy_tests.py
```

The complete CPU-oriented smoke suite requires the repository's normal
PyTorch stack, but still does not load LongCLIP, a checkpoint, or a GPU:

```bash
python tests/run_rag_smoke_tests.py
# or, when pytest is installed
python -m pytest tests/test_retrieval.py -q
```

They cover train-only indexing, Weather caption mapping, leave-one-series-out,
normalization,
deterministic Top-1, seeded Top-K sampling, similarity fallback, reference
tensor conversion, reuse of the DDPM forward formula, the disabled baseline
path, reverse-loop bounds, and retrieval-only bypass. Real LongCLIP indexing
and released-checkpoint smoke evaluation still require an environment with the
pinned PyTorch/Transformers stack and, for practical speed, a supported GPU.

### Verification record for this implementation

On 2026-08-10, the phase-1 dependency-light suite passed its original 6 checks. The
current expanded runners additionally cover exclusion and adaptive-controller
math/schema checks. Temporary
schema indexes using deterministic test embeddings built successfully from the
complete released training arrays:
Synth-M produced 24,000 caption records for 24,000 series with shape
`(24000,128,2)`; Weather produced 30,576 caption records for 10,192 series
with shape `(10192,36,21)` and the expected three-to-one mapping. Python AST/
bytecode compilation and shell-script syntax checks also passed.

The available Python interpreters did not contain PyTorch/PyYAML/Transformers,
and both the configured mirror and official package sources were blocked by a
local TLS policy. Consequently, the 11-test PyTorch smoke suite, production
LongCLIP index build, disabled-RAG runtime evaluation, and released-checkpoint
inference were not executed in this environment. No production index or model
result is claimed; rerun the commands above in the pinned server environment.
