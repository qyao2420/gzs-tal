export CUDA_VISIBLE_DEVICES=3
python ./eval_kd.py \
    ./config_gzs/thumos_clip_50_gzs.yaml \
    ./ckpt/zs_tal/thumos_clip_50_gzs_no \
    -epoch 70 \
    --use_clip \
    --tent \
    --tsa_decoder 0 \
    --ttt_type bi_ttt \
    --bi_ttt_type double \
    --mini_batch_size 64 \
    --window_size 64 \
    --num_ttt_encoders 1 \
    --encoder_version v0 \
    --write_json \
    