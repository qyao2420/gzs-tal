python ./train.py \
    ./configs/gtea_clip.yaml \
    --output clip \
    --ckpt-freq 5 \
    --use_clip \
    --ttt_type bi_ttt \
    --bi_ttt_type double \
    --mini_batch_size 16 \
    --window_size 16 \
