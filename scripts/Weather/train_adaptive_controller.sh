python tools/train_strength_controller.py \
  --teacher-npz ./cache/adaptive/Weather/run0_teacher.npz \
  --output-dir ./save/adaptive_controller/Weather/score_plus_pair \
  --feature-mode score_plus_pair --device cuda:0 \
  "$@"
