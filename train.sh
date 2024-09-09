CUDA_VISIBLE_DEVICES=0  python train_lsd.py  \
--enable_xformers_memory_efficient_attention --dataloader_num_workers 4 --learning_rate 1e-4 \
--mixed_precision fp16 --num_validation_images 16 --val_batch_size 16 --max_train_steps 200000 \
--checkpointing_steps 1 --checkpoints_total_limit 2 --gradient_accumulation_steps 1 \
--seed 42 --encoder_lr_scale 1.0 --train_split_portion 0.9 \
--output_dir ./logs/movi-e/use_mask_cnn_with_postion_embedding_fixed_validation/ \
--backbone_config configs/movi-e/backbone/confIG.json \
--slot_attn_config configs/movi-e/slot_attn/config.json \
--encoder_cnn_config configs/movi-e/encoder_cnn/encoder_cnn_config.json \
--unet_config configs/movi-e/unet/config.json \
--scheduler_config configs/movi-e/scheduler/scheduler_config.json \
--dataset_root "/home/lr-2002/code/ABC-s/dataset/billiards_balls_n2m10r30size128/training" \
--dataset_glob '**/*.png' --train_batch_size 32 --resolution 256 --validation_steps 1 \
--tracker_project_name latent_slot_diffusion --use_mask  True --resume_from_checkpoint latest