python ./train.py \
    ./configs/zs_tal/gtea_iv_75_train.yaml \
    --output iv \
    --ckpt-freq 5 \
    --internvideo \
    --use_ttt \
    --ttt_type bi_ttt \
    --bi_ttt_type double \
    --mini_batch_size 64 \
    --window_size 72 \
    --num_ttt_encoders 6 \
