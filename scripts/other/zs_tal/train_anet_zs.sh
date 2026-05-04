python ./train.py \
    ./configs/zs_tal/anet_iv_50_train.yaml \
    --output iv_true_1 \
    --ckpt-freq 1 \
    --internvideo \
    --use_ttt \
    --ttt_type bi_ttt \
    --bi_ttt_type double \
    --mini_batch_size 6 \
    --window_size 6 \
    --num_ttt_encoders 1 \
