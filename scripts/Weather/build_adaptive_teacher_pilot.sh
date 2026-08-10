#!/usr/bin/env bash
set -euo pipefail

# Keep per-query progress visible through tee during long teacher builds.
python -u tools/build_strength_teacher.py \
  --dataset-folder ./datasets/Weather --dataset-name Weather \
  --index-path ./cache/rag/Weather/train_longclip.npz \
  --longclip-path ./save/Longclip \
  --diff-config configs/Weather/diff/model_text2ts_dep.yaml \
  --cond-config configs/Weather/cond/text_msmdiffmv.yaml \
  --verbalts-checkpoint ./save/Weather_eval/text2ts_msmdiffmv/0/ckpts/model_best_loss.pth \
  --cttp-config ./save/Weather_cttp/model_configs.yaml \
  --cttp-checkpoint ./save/Weather_cttp/clip_model_best.pth \
  --base-patch 1 --multipatch-num 3 --patch-length 3 \
  --max-queries 32 --samples-per-action 1 --resume \
  --output-npz ./cache/adaptive/Weather/run0_teacher_pilot.npz \
  "$@"
