python ./eval_copy.py \
    ./configs/zs_tal/gtea_iv_100_test.yaml \
    ./ckpt/zs_tal/gtea_iv_50_train_iv \
    -epoch 10 \
    --internvideo \
    --ttt_type bi_ttt \
    --bi_ttt_type double \
    --mini_batch_size 64 \
    --window_size 72 \
    --num_ttt_encoders 6 \
    --write_json \
