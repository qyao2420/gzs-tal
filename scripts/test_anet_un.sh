export CUDA_VISIBLE_DEVICES=6
python ./eval.py \
    ./config_gzs/anet_iv_50_gzs.yaml \
    ./ckpt/zs_tal/anet_iv_50_gzs_no \
    -epoch 40 \
    --internvideo \
    --sar \
    --tsa_decoder 0 \
    --ttt_type un_ttt \
    --bi_ttt_type double \
    --mini_batch_size 64 \
    --window_size 64 \
    --num_ttt_encoders 1 \
    --encoder_version v0 \
    --write_json \
