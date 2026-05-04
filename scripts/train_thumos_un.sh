export CUDA_VISIBLE_DEVICES=3
python ./train.py \
    ./config_gzs/thumos_iv_50_gzs.yaml \
    --output Un_en-64_de-64-4h_8-2 \
    --ckpt-freq 5 \
    --internvideo \
    --use_ttt \
    --tsa_decoder 5 \
    --ttt_type un_ttt \
    --bi_ttt_type double \
    --mini_batch_size 64 \
    --window_size 64 \
    --num_ttt_encoders 1 \
    --encoder_version v0 \
