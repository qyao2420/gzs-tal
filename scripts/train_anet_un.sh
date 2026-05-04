export CUDA_VISIBLE_DEVICES=7
python ./train.py \
    ./config_gzs/anet_clip_50_gzs.yaml \
    --output no \
    --ckpt-freq 5 \
    --use_clip \
    --tsa_decoder 0 \
    --ttt_type un_ttt \
    --bi_ttt_type double \
    --mini_batch_size 64 \
    --window_size 64 \
    --num_ttt_encoders 1 \
    --encoder_version v0 \
