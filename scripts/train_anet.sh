export CUDA_VISIBLE_DEVICES=1
python ./train.py \
    ./config_gzs/anet_clip_50_gzs.yaml \
    --output Bi-kl_en-64_8-2 \
    --ckpt-freq 5 \
    --use_clip \
    --use_ttt \
    --tsa_decoder 0 \
    --ttt_type bi_ttt \
    --bi_ttt_type double \
    --mini_batch_size 64 \
    --window_size 64 \
    --num_ttt_encoders 1 \
    --encoder_version v0 \
