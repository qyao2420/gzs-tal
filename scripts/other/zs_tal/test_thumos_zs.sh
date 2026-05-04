python ./eval_copy.py \
    ./configs/zs_tal/thumos_iv_8_100_test.yaml \
    ./ckpt/zs_tal/thumos_iv_8_75_train_iv \
    -epoch 35 \
    --internvideo \
    --ttt_type bi_ttt \
    --bi_ttt_type double \
    --mini_batch_size 64 \
    --window_size 72 \
    --num_ttt_encoders 1 \
    --encoder_version v0 \
    --write_json \
