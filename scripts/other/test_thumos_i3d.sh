python ./eval_copy.py \
    ./configs/thumos_i3d.yaml \
    ./ckpt/thumos_i3d_clip \
    -epoch 45 \
    --use_clip \
    --ttt_type bi_ttt \
    --bi_ttt_type double \
    --mini_batch_size 32 \
    --window_size 32 \
    --num_ttt_encoders 1 \
