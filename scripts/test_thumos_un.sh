export CUDA_VISIBLE_DEVICES=7
python ./eval.py \
    ./config_gzs/thumos_clip_50_gzs.yaml \
    ./ckpt/zs_tal/thumos_clip_50_gzs_no \
    -epoch 80 \
    --use_clip \
    --sar \
    --tsa_decoder 0 \
    --ttt_type un_ttt \
    --bi_ttt_type double \
    --mini_batch_size 64 \
    --window_size 64 \
    --num_ttt_encoders 1 \
    --encoder_version v0 \
    --write_json \
    