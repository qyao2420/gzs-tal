python ./train.py \
    ./configs/zs_tal/thumos_iv_8_50_train.yaml \
    --output iv \
    --ckpt-freq 5 \
    --internvideo \
    --ttt_type bi_ttt \
    --bi_ttt_type double \
    --mini_batch_size 64 \
    --window_size 72 \
    --num_ttt_encoders 1 \
    --encoder_version v0 \
