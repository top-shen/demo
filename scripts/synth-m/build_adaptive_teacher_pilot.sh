python tools/build_strength_teacher.py \
  --dataset-folder ./datasets/synth-m --dataset-name synth-m \
  --index-path ./cache/rag/synth-m/train_longclip.npz \
  --longclip-path ./save/Longclip \
  --diff-config configs/synth-m/diff/model_text2ts_dep.yaml \
  --cond-config configs/synth-m/cond/text_msmdiffmv.yaml \
  --verbalts-checkpoint ./save/synth-m_eval/text2ts_msmdiffmv/0/ckpts/model_best_loss.pth \
  --cttp-config ./save/synth-m_cttp/model_configs.yaml \
  --cttp-checkpoint ./save/synth-m_cttp/clip_model_best.pth \
  --base-patch 4 --multipatch-num 3 --patch-length 3 \
  --max-queries 32 --samples-per-action 1 --resume \
  --output-npz ./cache/adaptive/synth-m/run0_teacher_pilot.npz \
  "$@"
