python tools/build_retrieval_index.py \
    --dataset-folder ./datasets/synth-m \
    --dataset-name synth-m \
    --model-path ./save/Longclip \
    --output ./cache/rag/synth-m/train_longclip.npz \
    --batch-size 64 \
    --device auto
