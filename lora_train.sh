config_dir="configs/movi-c_lora_dino"

# Your command
/home/lr-2002/anaconda3/envs/lsd/bin/python /home/lr-2002/code/latent-slot-diffusion/train_lsd.py \
    --enable_xformers_memory_efficient_attention \
    --dataloader_num_workers 4 \
    --learning_rate 3e-4 \
    --lora_lr 3e-5 \
    --mixed_precision fp16 \
    --num_validation_images 16 \
    --val_batch_size 16 \
    --max_train_steps 200000 \
    --checkpointing_steps 2000 \
    --checkpoints_total_limit 2 \
    --gradient_accumulation_steps 1 \
    --seed 42 \
    --encoder_lr_scale 1.0 \
    --train_split_portion 0.9 \
    --output_dir ./logs/movi-c-lora/train_lora_and_dino \
    --backbone_config pretrain_dino \
    --slot_attn_config ${config_dir}/slot_attn/config.json \
    --unet_config pretrain_sd \
    --scheduler_config ${config_dir}/scheduler/scheduler_config.json \
    --dataset_root "./movi_c/image/**.npy" \
    --dataset_glob "**/*.png" \
    --train_batch_size 16 \
    --resolution 256 \
    --validation_steps 1 \
    --tracker_project_name latent_slot_diffusion \
    --validation_steps 1000 \
    --encoder_cnn_config ${config_dir}/encoder_cnn/encoder_cnn_config.json \
    --train_slot true \
    --tune_unet true \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --train_dino true