python ./eval_copy.py \
    ./configs/zs_tal/anet_iv_100_test.yaml \
    ./ckpt/zs_tal/anet_iv_75_train_iv_true_1 \
    -epoch 10 \
    --internvideo \
    --use_ttt \
    --ttt_type bi_ttt \
    --bi_ttt_type double \
    --mini_batch_size 6 \
    --window_size 6 \
    --num_ttt_encoders 1 \
    --write_json \
