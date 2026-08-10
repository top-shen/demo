python tools/build_retrieval_index.py \
    --dataset-folder ./datasets/Weather \
    --dataset-name Weather \
    --model-path ./save/Longclip \
    --output ./cache/rag/Weather/train_longclip.npz \
    --batch-size 64 \
    --device auto
