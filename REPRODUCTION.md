# VerbalTS Reproduction Guide

This repository is prepared for evaluating the released VerbalTS checkpoints on
Synth-M and Weather. Run all commands from the repository root.

## 1. Supported reproduction scope

The authors released six datasets, but the public repository only contains model
configs, scripts, CTTP checkpoints, and VerbalTS checkpoints for Synth-M and
Weather. Strict checkpoint-based evaluation is therefore limited to those two
datasets.

Synth-U, BlindWays, ETTm1, and Istanbul Traffic are preserved under `datasets/`,
but reproducing the paper results on them still requires official configs and
checkpoints, or a reimplementation of the unpublished CTTP training pipeline.

## 2. Server environment

Recommended baseline:

- Linux
- Python 3.10
- NVIDIA driver compatible with CUDA 12.1
- One A800 GPU

Create an isolated environment and install the pinned dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` selects the PyTorch 2.2.1 CUDA 12.1 wheel. A newer host CUDA
toolkit is not required as long as the NVIDIA driver supports CUDA 12.1.

## 3. Resource layout

`_downloads/` is the immutable source archive. Prepare the paths expected by the
released scripts with:

```bash
python tools/prepare_resources.py
```

The default mode tries hard links first and falls back to copies. Use
`--mode copy` when the source and target are on different filesystems. Existing
files are retained; pass `--force` only when intentionally rebuilding the
operational layout.

The prepared layout is:

```text
datasets/
  BlindWays/
  ETTm1/
  istanbul_traffic/
  synth-m/
  synth-u/
  Weather/
save/
  Longclip/
  synth-m_cttp/
  synth-m_eval/text2ts_msmdiffmv/{0,1,2}/
  Weather_cttp/
  Weather_eval/text2ts_msmdiffmv/{0,1,2}/
```

The preparation tool also changes the CTTP config copies to use
`./save/Longclip`. It does not modify files below `_downloads/`.

When transferring with Git, note that `_downloads/`, `datasets/`, and `save/`
are intentionally ignored because they contain multi-gigabyte assets. Transfer
the resource directories separately, or upload the complete working directory
with a file-transfer tool that does not rely on `git ls-files`.

## 4. Resource verification

Run the fast structural check:

```bash
python tools/verify_resources.py
```

After transfer to the server, run the full value/hash/CRC check:

```bash
python tools/verify_resources.py --deep
```

The expected result is zero failures and two advisories:

- ETTm1 has released attribute value `-1` at training indices 504-507. This does
  not block the text-conditioned VerbalTS path.
- Synth-M CTTP declares `n_var: 4`, while the released time series have two
  variables. A PyTorch load and forward check is still required on the server.

## 5. Evaluate released checkpoints

```bash
bash scripts/synth-m/eval.sh
bash scripts/Weather/eval.sh
```

Each script evaluates checkpoint runs `0`, `1`, and `2`. Results are written to:

```text
save/synth-m_eval/text2ts_msmdiffmv/results.csv
save/synth-m_eval/text2ts_msmdiffmv/results_stat_condgen.csv
save/Weather_eval/text2ts_msmdiffmv/results.csv
save/Weather_eval/text2ts_msmdiffmv/results_stat_condgen.csv
```

Use the configs in `configs/synth-m/` and `configs/Weather/`. The
`eval_configs.yaml` files distributed with the checkpoints describe an older
repository layout and are retained only as provenance.

## 6. Known paper/code differences

- The paper reports a learning rate of `1e-4`; the public VerbalTS scripts pass
  the code default of `1e-3`.
- The public scripts use `--cond_modal simple_text`. This uses the LongCLIP
  tokenizer with a learned embedding/Transformer rather than frozen LongCLIP
  text features inside the VerbalTS generator.
- Evaluation generates ten candidates and reports the elementwise median.
- FID and J-FTSD use CTTP embeddings whose real-data statistics are computed
  from the training split.
- The paper's Synth-M validation/test count of 2400 conflicts with the released
  arrays, which contain 4000 samples in each split.

Do not silently change these choices. Record any deviation in the experiment
log before comparing results with the paper.
