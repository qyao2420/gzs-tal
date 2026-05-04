python ./eval.py \
    ./configs/gtea_clip.yaml \
    ./ckpt/A800_gtea_clip_clip_bittt \
    -epoch 105 \
    --use_clip \
    --use_ttt \
    --ttt_type bi_ttt \
    --bi_ttt_type double \
    --mini_batch_size 64 \
    --window_size 72 \
    --write_json \
    