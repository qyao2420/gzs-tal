export CUDA_VISIBLE_DEVICES=4
python ./train.py \
    ./config_gzs/thumos_iv_50_gzs.yaml \
    --output Bi-kl_en-64_de-64-4h_2-8 \
    --ckpt-freq 5 \
    --internvideo \
    --use_ttt \
    --tsa_decoder 5 \
    --ttt_type bi_ttt \
    --bi_ttt_type double \
    --mini_batch_size 64 \
    --window_size 64 \
    --num_ttt_encoders 1 \
    --encoder_version v0 \
