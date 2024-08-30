# Parameters Description
- `train_dino` will only work if backbone is `pretrain_dino` and use `train_dino` then, the code will train the `dino` model
  - the default `dino_lr` is set to 3e-5, if you change it in the script, it will change in the training part 
  - the `use_roi` will work if you use `pretrain_dino` only 
- `train_slot` will only work if you use `slot attention` to train 
- `tune_unet` will only work if you use `pretrain_sd` as unet config, and it will make up `lora` layers in the unet part 
  - what's more, notice the `lora_lr` to change the lora learning rate 
  - it is separate from other `dino`/`backbone` model