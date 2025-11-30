export CUDA_VISIBLE_DEVICES=3,4

LOGFILE="p1_same.log"

echo "Starting training on GPU 3 and 4..."
echo "Log file: $LOGFILE"

git pull && nohup accelerate launch --num_processes=2 train.py \
    --seed=123 \
    --data_root="/datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FontDiffuser" \
    --report_to="wandb" \
    --resolution=64 \
    --style_image_size=64 \
    --content_image_size=64 \
    --content_encoder_downsample_size=3 \
    --channel_attn=True \
    --content_start_channel=64 \
    --style_start_channel=64 \
    --train_batch_size=16 \
    --perceptual_coefficient=0.01 \
    --offset_coefficient=0.5 \
    --max_train_steps=400000 \
    --ckpt_interval=50000 \
    --gradient_accumulation_steps=1 \
    --log_interval=50 \
    --learning_rate=1e-4 \
    --lr_scheduler="linear" \
    --lr_warmup_steps=10000 \
    --drop_prob=0.1 \
    --mixed_precision="no" \
    --lang_mode="same" \
    --experience_name="P1_SAME" \
    --output_dir="ckpt/p1_same" \
    --rsi_mode "rsi" \
    > "$LOGFILE" 2>&1 &