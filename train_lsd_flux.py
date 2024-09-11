import argparse
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import peft
from peft import PeftModel
import copy
import gc
import hashlib
import importlib
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import peft
from torchvision.ops import roi_align
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from util.data import GlobVideoDataset_Mask_Movi
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
# from diffusers.training_utils import compute_snr # diffusers is still working on this, uncomment in future versions
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, reduce, repeat

from src.models.backbone import UNetEncoder
from src.models.slot_attn import MultiHeadSTEVESA
from src.models.unet_with_pos import UNet2DConditionModelWithPos
from src.models.Encoder_CNN import EncoderCNN
from src.data.dataset import GlobDataset_MASK
from torch import nn
from src.parser import parse_args

from src.models.utils import ColorMask, PositionNet
if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def dino_in_slots_out(batch, feat, object_encoder_cnn):
    if args.use_roi:
        masks = batch['mask']
        num_slots = int(masks.max().item()) + 1
        mask_resized = F.interpolate(masks, size=(32, 32), mode='nearest', align_corners=None)
        masks = mask_resized
        all_rois = []
        num_roi_per_batch = []
        # Loop over each image in the batch
        for batch_idx in range(masks.shape[0]):
            mask = masks[batch_idx, 0]  # Extract mask for the current image, shape [32, 32]

            # Initialize a list to hold RoIs for the current image
            rois = []

            # Loop over each unique class in the mask (excluding background class 0)
            for class_idx in mask.unique():
                if class_idx == 0:
                    continue  # Skip background class if present

                # Create a binary mask for the current class
                class_mask = (mask == class_idx).nonzero(as_tuple=False)

                if class_mask.numel() == 0:
                    continue  # Skip if no pixels for this class

                # Extract the bounding box for the current class
                y1, x1 = class_mask[:, 0].min().item(), class_mask[:, 1].min().item()
                y2, x2 = class_mask[:, 0].max().item(), class_mask[:, 1].max().item()

                # Append the RoI in the format [batch_index, x1, y1, x2, y2]
                rois.append([batch_idx, x1, y1, x2, y2])

            # Append the RoIs for the current image to the all_rois list
            all_rois.extend(rois)
            num_roi_per_batch.append(len(rois))

        # Convert all RoIs to a tensor
        all_rois = torch.tensor(all_rois, dtype=torch.float16, device=feat.device)
        max_rois = num_slots
        # Desired output size for each RoI
        output_size = (2, 2)  # 2x2 output

        # Apply RoIAlign
        aligned_features = roi_align(
            input=feat.to(all_rois.dtype),
            boxes=all_rois,
            output_size=output_size,
            spatial_scale=1.0,  # Assuming feature map and mask are at the same scale
            sampling_ratio=-1  # Use adaptive sampling
        )

        # `aligned_features` will have the shape [num_rois, channels, output_height, output_width]
        # print(aligned_features.shape)  # Expected output: [num_rois, 768, 2, 2]

        # Initialize the padded output tensor with zeros
        batch_size = feat.size(0)
        channels = feat.size(1)
        output_shape = (batch_size, max_rois, channels, output_size[0], output_size[1])

        padded_aligned_features = torch.zeros(output_shape)
        padded_rois = torch.zeros((batch_size, max_rois, 4))
        # Track the current index in `aligned_features`
        current_index = 0

        # Loop through each batch to place aligned features in the padded tensor
        for batch_idx in range(batch_size):
            num_rois = num_roi_per_batch[batch_idx]

            if num_rois > 0:
                # Extract features for current batch
                batch_features = aligned_features[current_index:current_index + num_rois]
                batch_rois = all_rois[current_index:current_index + num_rois, 1:]
                # Place them in the padded tensor
                padded_aligned_features[batch_idx, :num_rois] = batch_features
                padded_rois[batch_idx, :num_rois] = batch_rois
                # Update the current index
                current_index += num_rois

        # `padded_aligned_features` shape: [batch_size, max_rois, 768, 2, 2]
        # print(padded_aligned_features.shape)  # Expected output: [16, max_rois, 768, 2, 2]
        pos_emb = object_encoder_cnn.bbx_to_pos(padded_rois.to(object_encoder_cnn.device))
        pos_emb = pos_emb.unsqueeze(-1).repeat(1, 1, 1, 4)
        flatten_slots = padded_aligned_features.flatten(-2)
        cated_slots = torch.cat([flatten_slots.to(pos_emb.device), pos_emb], dim=-2).permute(0, 1, 3, 2)
        slots = object_encoder_cnn.cat_to_slots(cated_slots).flatten(1,2)
        return slots, num_slots

@torch.no_grad()
def log_validation(
    val_dataset,
    backbone,
    slot_attn,
    unet,
    vae,
    scheduler,
    args,
    accelerator,
    weight_dtype,
    global_step,
    object_encoder_cnn,
    num_slots,
):
    logger.info(
        f"Running validation... \n."
    )
    unet = accelerator.unwrap_model(unet)
    backbone = accelerator.unwrap_model(backbone)
    if args.train_slot:
        slot_attn = accelerator.unwrap_model(slot_attn)
    object_encoder_cnn = accelerator.unwrap_model(object_encoder_cnn)
    colorizer = ColorMask(
        num_slots=num_slots,
        log_img_size=256,
        norm_mean=0,
        norm_std=1,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in scheduler.config:
        variance_type = scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    # use a more efficient scheduler at test time
    module = importlib.import_module("diffusers")
    scheduler_class = getattr(module, args.validation_scheduler)
    scheduler = scheduler_class.from_config(
        scheduler.config, **scheduler_args)

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=None,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(
        device=accelerator.device).manual_seed(args.seed)

    num_digits = len(str(args.max_train_steps))
    folder_name = f"image_logging_{global_step:0{num_digits}}"
    image_log_dir = os.path.join(accelerator.logging_dir, folder_name, )
    os.makedirs(image_log_dir, exist_ok=True)

    images = []
    image_count = 0

    for batch_idx, batch in enumerate(val_dataloader):

        pixel_values = batch["pixel_values"].to(
            device=accelerator.device, dtype=weight_dtype)

        with torch.autocast("cuda"):
            model_input = vae.encode(pixel_values).latent_dist.sample()
            pixel_values_recon = vae.decode(model_input).sample

            if args.backbone_config == "pretrain_dino":
                pixel_values_vit = batch["pixel_values_vit"].to(device=accelerator.device,
                                                                dtype=weight_dtype)
                feat = backbone(pixel_values_vit)
            else:
                feat = backbone(pixel_values)
            if args.use_roi:
                slots, num_slots = dino_in_slots_out(batch, feat, object_encoder_cnn)
            elif args.use_slot_query:
                slots, num_slots = from_feat_to_cross_attention_slots(feat, batch['mask'], object_encoder_cnn)
            else:
                if args.use_mask:
                    logger.info('use mask to validation ')

                    masks = batch['mask'].to(feat.device)
                    num_slots = int(masks.max().item()) + 1
                    feat = object_encoder_cnn.spatial_pos(feat)
                    mask_resized = F.interpolate(masks, size=(64, 64), mode='nearest', align_corners=None)
                    # mask_resized = mask_resized.permute(0,1,3,4,2)
                    mask_resized = [mask_resized == i for i in range(num_slots)]
                    masked_emb = [feat * mask for mask in mask_resized]
                    masked_emb = torch.stack(masked_emb, dim=0).flatten(end_dim=1)

                    objects_emb = object_encoder_cnn(masked_emb).reshape(-1, num_slots, args.d_model)
                    slots = object_encoder_cnn.mlp(object_encoder_cnn.layer_norm(objects_emb))
                    # add empty here
                    # replace slots with gaussian noise
                    need_to_replace = torch.stack([mask.sum(dim=(2, 3)) == 0 for mask in mask_resized]).permute(1,0,2).to(torch.int)
                    slots = slots * (1 - need_to_replace) + slot_attn.empty_slot.expand(*slots.shape).to(slots.device) * need_to_replace


                    slots, attn = slot_attn(feat[:, None], slots)
                    slots = slots[:, 0]
                else:
                    slots, attn = slot_attn(feat[:, None])  # for the time dimension
                    slots = slots[:, 0]

            # slots, attn = slot_attn(feat[:, None])  # for the time dimension
            # slots = slots[:, 0]
            images_gen = pipeline(
                prompt_embeds=slots,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=25,
                generator=generator,
                guidance_scale=1, # todo 1.5
                output_type="pt",
            ).images
        if args.use_roi or args.use_slot_query:
            grid_image = colorizer.get_heatmap(img=(pixel_values * 0.5 + 0.5),
                                               attn=torch.zeros((pixel_values.shape[0],5,64,64)),
                                               recon=[pixel_values_recon * 0.5 + 0.5,
                                                      images_gen])  # pixel is vae decode; images_gen is slot recon

        else:
            grid_image = colorizer.get_heatmap(img=(pixel_values * 0.5 + 0.5),
                                               attn=reduce(
                                                   attn[:, 0], 'b num_h (h w) s -> b s h w', h=int(np.sqrt(attn.shape[-2])),
                                                   reduction='mean'
                                               ),
                                               recon=[pixel_values_recon * 0.5 + 0.5, images_gen]) # pixel is vae decode; images_gen is slot recon
        ndarr = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        images.append(im)
        img_path = os.path.join(image_log_dir, f"image_{batch_idx:02}.jpg")
        im.save(img_path, optimize=True, quality=95)
        image_count += pixel_values.shape[0]
        if image_count >= args.num_validation_images:
            break

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "validation", np_images, global_step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

    return images




def get_bounding_boxes(masks):
    B, num_slots, H, W = masks.shape
    bounding_boxes = torch.zeros(B, num_slots, 4)  # (x, y, w, h) for each slot

    for b in range(B):
        for slot in range(num_slots):
            mask = masks[b, slot]

            # Find non-zero (True) elements in the mask
            non_zero_indices = torch.nonzero(mask, as_tuple=True)

            if len(non_zero_indices[0]) > 0:
                # Get min and max coordinates for the mask
                y_min = torch.min(non_zero_indices[0])
                y_max = torch.max(non_zero_indices[0])
                x_min = torch.min(non_zero_indices[1])
                x_max = torch.max(non_zero_indices[1])

                # Calculate (x, y, w, h)
                x = x_min.item()
                y = y_min.item()
                w = (x_max - x_min + 1).item()  # width
                h = (y_max - y_min + 1).item()  # height

                bounding_boxes[b, slot] = torch.tensor([x, y, w, h])

    return bounding_boxes


def from_feat_to_cross_attention_slots(feat, masks, encoder_cnn:EncoderCNN):
    """
    feat and some layers feature in
    average init slots
    slots update with cross attention
    return slots

    9 *16
    ->
    144


    16 *9 -> 144
    todo test
    """

    masks = masks.to(feat.device)
    num_slots = int(masks.max().item()) + 1
    feat = encoder_cnn.spatial_pos(feat)
    mask_resized = F.interpolate(masks, size=(64, 64), mode='nearest', align_corners=None)
    # mask_resized = mask_resized.permute(0,1,3,4,2)
    mask_resized = [mask_resized == i for i in range(num_slots)]
    bbx = get_bounding_boxes(torch.stack(mask_resized, dim=0).squeeze(2).permute(1,0,2,3))
    masked_emb = [feat * mask for mask in mask_resized] # 64 *64
    # todo delete corresponding zero-emb
    masked_emb = torch.stack(masked_emb, dim=0).permute(1,0, 2,3,4)
    slots = torch.mean(masked_emb, (3,4))
    tf_input = masked_emb.flatten(3,4) #need pos
    plc_slots = []
    for i in range(slots.shape[1]):
        plc_slots.append(encoder_cnn.cross_attention(slots[:, i].unsqueeze(1), tf_input[:,i].permute(0,2,1)))
    slots = torch.stack(plc_slots, dim=1).squeeze(2)
    bbx = bbx.to(slots.device)
    ppn = PositionNet(slots.shape[-1], slots.shape[-1]).to(slots.device)
    slots = ppn(bbx, slots)
    return slots, num_slots
def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        # gradient_accumulation_steps=args.gradient_accumulation_steps, # for manually handled case, should not pass it here
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id


    # Load scheduler and models
    if args.unet_config == "pretrain_sd":
        print('-----------pretrain model is ', args.pretrained_model_name)
        noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name, subfolder="scheduler")
    else:
        noise_scheduler_config = DDPMScheduler.load_config(args.scheduler_config)
        noise_scheduler = DDPMScheduler.from_config(noise_scheduler_config)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name, subfolder="vae")

    if os.path.exists(args.backbone_config):
        train_backbone = True
        backbone_config = UNetEncoder.load_config(args.backbone_config)
        backbone = UNetEncoder.from_config(backbone_config)
    elif args.backbone_config == "pretrain_dino":
        train_backbone = False
        if args.train_dino :
            train_backbone=True
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        class DINOBackbone(torch.nn.Module):
            def __init__(self, dinov2):
                super().__init__()
                self.dinov2 = dinov2

            def forward(self, x):
                enc_out = self.dinov2.forward_features(x)
                return rearrange(
                    enc_out["x_norm_patchtokens"],
                    "b (h w ) c -> b c h w",
                    h=int(np.sqrt(enc_out["x_norm_patchtokens"].shape[-2]))
                )
        backbone = DINOBackbone(dinov2)
        backbone = backbone.to(torch.float32)
    else:
        raise ValueError(
            f"Unknown unet config {args.unet_config}")

    if args.train_slot:
        train_slot = args.train_slot
        slot_attn_config = MultiHeadSTEVESA.load_config(args.slot_attn_config)
        slot_attn = MultiHeadSTEVESA.from_config(slot_attn_config)

    object_encoder_cnn_config = EncoderCNN.load_config(args.encoder_cnn_config)
    object_encoder_cnn = EncoderCNN.from_config(object_encoder_cnn_config)

    if os.path.exists(args.unet_config):
        train_unet = True
        unet_config = UNet2DConditionModelWithPos.load_config(args.unet_config)
        unet = UNet2DConditionModelWithPos.from_config(unet_config)
    elif args.unet_config == "pretrain_sd":
        if 'FLUX' in  args.pretrained_model_name :
            from diffusers import FluxTransformer2DModel
            transformer = FluxTransformer2DModel.from_single_file('https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors')
        train_unet = False
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name, subfolder="unet", revision=args.revision
        )
        if args.tune_unet:
            train_unet =True
            from peft import get_peft_model, LoraConfig
            unet_lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha= args.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            unet = get_peft_model(unet, unet_lora_config)
            lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
            lora_parameters = list(lora_layers)
    else:
        raise ValueError(
            f"Unknown unet config {args.unet_config}")

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:

                # continue if not one of [UNetEncoder, MultiHeadSTEVESA, UNet2DConditionModelWithPos]
                if not isinstance(model, (UNetEncoder, MultiHeadSTEVESA, UNet2DConditionModelWithPos, EncoderCNN, PeftModel)):
                    continue

                sub_dir = model._get_name().lower()

                model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()
    peftmodel = None
    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again

            model = models.pop()
            if len(models) >1 and isinstance(model, peft.PeftModel):
                peftmodel = model
                continue
            sub_dir = model._get_name().lower()

            if isinstance(model, UNetEncoder) or isinstance(model, peft.PeftModel):
                # load diffusers style into model
                load_model = UNetEncoder.from_pretrained(
                    input_dir, subfolder=sub_dir)
                model.register_to_config(**load_model.config)
            elif isinstance(model, MultiHeadSTEVESA):
                load_model = MultiHeadSTEVESA.from_pretrained(
                    input_dir, subfolder=sub_dir)
                model.register_to_config(**load_model.config)
            elif isinstance(model, UNet2DConditionModelWithPos):
                load_model = UNet2DConditionModelWithPos.from_pretrained(
                    input_dir, subfolder=sub_dir)
                model.register_to_config(**load_model.config)
            elif isinstance(model, EncoderCNN):
                load_model = EncoderCNN.from_pretrained(
                    input_dir, subfolder=sub_dir)
                model.register_to_config(**load_model.config)
            # todo need to load object encoder here
            else:
                raise ValueError(
                    f"Unknown model type {type(model)}")

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    if not train_backbone:
        try:
            backbone.requires_grad_(False)
        except:
            pass
    if not train_unet:
        if not args.tune_unet:
            unet.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            try:
                backbone.enable_xformers_memory_efficient_attention()
            except AttributeError:
                pass
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        try:
            backbone.enable_gradient_checkpointing()
        except AttributeError:
                pass

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if train_unet and accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if train_backbone :
        model_unwrapped = accelerator.unwrap_model(backbone)
        all_params = all(p.dtype == torch.float32 for p in model_unwrapped.parameters())
        if not all_params:
            raise ValueError(
                f"Backbone loaded as datatype {accelerator.unwrap_model(backbone).dtype}. {low_precision_error_string}"
            )

    if accelerator.unwrap_model(object_encoder_cnn).dtype != torch.float32:
        raise ValueError(
            f"Object_encoder_cnn loaded as datatype {accelerator.unwrap_model(object_encoder_cnn).dtype}. {low_precision_error_string}"
        )
    if args.train_slot:
        if accelerator.unwrap_model(slot_attn).dtype != torch.float32:
            raise ValueError(
                f"Slot Attn loaded as datatype {accelerator.unwrap_model(slot_attn).dtype}. {low_precision_error_string}"
            )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if args.tune_unet:
        lora_parameters = lora_parameters
    else:
        lora_parameters = []
    params_to_optimize = (list(slot_attn.parameters() if args.train_slot else [])
                          + list(object_encoder_cnn.parameters()) + \
        (list(backbone.parameters()) if train_backbone else []) + \
        (list(unet.parameters()) if (train_unet and not args.tune_unet) else []) + \
                          (lora_parameters)
        )
    params_to_optimize = [x.to(torch.float32) for x in params_to_optimize]
    # params_group = [
    #     {'params': list(slot_attn.parameters() if args.train_slot else [] )  + list(object_encoder_cnn.parameters()) + \
    #      (list(backbone.parameters()) if train_backbone else [])+ (lora_parameters),
    #      'lr': args.learning_rate * args.encoder_lr_scale}
    # ]
    #
    params_group = [
        {'params': list(slot_attn.parameters() if args.train_slot else [] )
          + list(object_encoder_cnn.parameters()) ,
         'lr': args.learning_rate * args.encoder_lr_scale},
    ]
    if train_unet:
        if not args.tune_unet:
            params_group.append(
                {'params': unet.parameters(), "lr": args.learning_rate}
            )
        else:
            params_group.append({
               'params': (lora_parameters), "lr": args.lora_lr})

    if train_backbone:
        if args.train_dino:
            print('using train_dino ')
            params_group.append(
                {'params':list(backbone.parameters()), 'lr': args.dino_lr})
        else:
            params_group.append(
                {'params':list(backbone.parameters()), 'lr': args.learning_rate * args.encoder_lr_scale})

    optimizer = optimizer_class(
        params_group,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # implement your lr_sceduler here, here I use constant functions as
    # the template for your reference
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=[lambda _: 1, lambda _: 1] if train_unet else [lambda _: 1]
    #     )
    #
    # lr_lam = lambda  _:1
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lambda _:1 for x in range(len(params_group))]
    )
    # train_dataset = GlobDataset_MASK(
    #     root=args.dataset_root,
    #     img_size=args.resolution,
    #     img_glob=args.dataset_glob,
    #     data_portion=(0.0, args.train_split_portion),
    #     vit_norm=args.backbone_config == "pretrain_dino",
    #     random_flip=args.flip_images,
    #     vit_input_resolution=args.vit_input_resolution
    # )
    train_dataset = GlobVideoDataset_Mask_Movi(args.dataset_root, 'train', img_size=128, target_shape=256, ep_len=1, vit_norm=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    val_dataset = GlobVideoDataset_Mask_Movi(args.dataset_root, 'val', img_size=128, target_shape=256, ep_len=1, vit_norm=True)
    # validation set is only for visualization
    # val_dataset = GlobDataset_MASK(
    #     root=args.dataset_root,
    #     img_size=args.resolution,
    #     img_glob=args.dataset_glob,
    #     data_portion=(args.train_split_portion if args.train_split_portion < 1. else 0.9, 1.0),
    #     vit_norm=args.backbone_config == "pretrain_dino",
    #     vit_input_resolution=args.vit_input_resolution
    # )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader, lr_scheduler, object_encoder_cnn = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler, object_encoder_cnn
    )

    if args.train_slot:
        slot_attn = accelerator.prepare(slot_attn)
    if train_backbone:
        backbone = accelerator.prepare(backbone)
    if train_unet:
        unet = accelerator.prepare(unet)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not train_backbone:
        try:
            backbone.to(accelerator.device, dtype=weight_dtype)
        except:
            pass
    if not train_unet:
        if args.tune_unet:
            unet.to(accelerator.device, dtype=torch.float32)
        else:
            unet.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            args.tracker_project_name, config=tracker_config
        )

    # Train!
    total_batch_size = args.train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    accumulate_steps = 0 # necessary for args.gradient_accumulation_steps > 1

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint.rstrip('/')) # only the checkpoint folder name is needed, not the full path
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.print(f'the path is {os.path.join(args.output_dir, path)}')
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            accumulate_steps = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        position=0, leave=True
    )
    bce_loss_calculator = nn.BCELoss()
    loss_calculator = nn.Sigmoid()
    for epoch in range(first_epoch, args.num_train_epochs):
        """
        1. where is the noise added
        2.


        """
        if train_unet:
            unet.train()
        if train_backbone:
            backbone.train()
        if args.train_slot:
            slot_attn.train()
        bce_loss = None
        object_encoder_cnn.train()
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

            # Convert images to latent space
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor

            # Sample noise that we'll add to the model input
            if args.offset_noise:
                noise = torch.randn_like(model_input) + 0.1 * torch.randn(
                    model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device
                )
            else:
                noise = torch.randn_like(model_input)
            bsz, channels, height, width = model_input.shape
            # Samplefalseandom timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
            )
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(
                model_input, noise, timesteps)

            # timestep is not used, but should we?
            if args.backbone_config == "pretrain_dino":
                # todo here use pretrain_dino or other to make the backbone to get feature
                # todo it seems that the feat was just feed into the slot_attn
                pixel_values_vit = batch["pixel_values_vit"].to(dtype=weight_dtype)
                feat = backbone(pixel_values_vit)
            else:
                feat = backbone(pixel_values)
            if args.use_roi:
                slots, num_slots = dino_in_slots_out(batch, feat, object_encoder_cnn)
            elif args.use_slot_query:
                slots, num_slots = from_feat_to_cross_attention_slots(feat, batch['mask'], object_encoder_cnn)
            else:
                if args.use_mask:
                    masks = batch['mask']
                    num_slots = int(masks.max().item()) + 1
                    mask_resized = F.interpolate(masks, size=(64, 64), mode='nearest', align_corners=None)
                    feat = object_encoder_cnn.spatial_pos(feat)
                    mask_resized = [mask_resized == i for i in range(num_slots)]
                    masked_emb = [feat * mask for mask in mask_resized]
                    masked_emb = torch.stack(masked_emb, dim=0).flatten(end_dim=1)

                    objects_emb = object_encoder_cnn(masked_emb).reshape(-1, num_slots, args.d_model) # TODO Why no update ?
                    slots = object_encoder_cnn.mlp(object_encoder_cnn.layer_norm(objects_emb))
                    # slots = object_encoder_cnn.mlp(object_encoder_cnn.layer_norm(objects_emb) + objects_emb ) # TODO use this if maskloss is not working

                    # replace slots with gaussian noise
                    need_to_replace = torch.stack([mask.sum(dim=(2, 3)) == 0 for mask in mask_resized]).permute(1,0,2).to(torch.int)
                    slots = slots * (1 - need_to_replace) + slot_attn.empty_slot.expand(*slots.shape).to(slots.device) *  need_to_replace

                    # calculate slots
                    slots, attn, attn_logits = slot_attn(feat[:, None], slots, need_logits=True)
                    slots = slots[:, 0]


                    # calculate mask loss
                    reshaped_masks = torch.stack(mask_resized).squeeze(dim=2).permute(1,2,3,0).flatten(1,2).to(torch.float32)
                    attn_logits_flatten= attn.squeeze(1).squeeze(1)
                    # attn_logits_flatten= attn_logits.squeeze(1)
                    # bce_loss = bce_loss_calculator(attn_logits_flatten, reshaped_masks)
                else:
                    num_slots = slot_attn.num_slots
                    slots, attn = slot_attn(feat[:, None])  # for the time dimension
                    slots = slots[:, 0]

            if not train_unet:
                slots = slots.to(dtype=weight_dtype)

            # Predict the noise residual
            model_pred = unet(
                noisy_model_input, timesteps, slots,
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(
                    model_input, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Compute instance loss
            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(),
                                  target.float(), reduction="mean")
                original_loss = loss.clone()
                if bce_loss is not None:
                    loss += bce_loss
                if torch.isnan(loss):
                    print('nan loss', loss, 'saving all the tensor to calculate loss')

                    save_path =  f"wrong_loss.pt"
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    model_pred_path = 'model_pred.pt'
                    torch.save(model_pred.float(), model_pred_path)
                    target_path = 'target.pt'
                    torch.save(target.float(), target_path)

            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                base_weight = (
                    torch.stack(
                        [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                )

                if noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective needs to be floored to an SNR weight of one.
                    mse_loss_weights = base_weight + 1
                else:
                    # Epsilon and sample both use the same loss weights.
                    mse_loss_weights = base_weight
                loss = F.mse_loss(model_pred.float(),
                                  target.float(), reduction="none")
                loss = loss.mean(
                    dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)
            accumulate_steps += 1
            # if accelerator.sync_gradients:
            if (accumulate_steps+1) % args.gradient_accumulation_steps == 0:
                params_to_clip = params_to_optimize
                accelerator.clip_grad_norm_(
                    params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if (accumulate_steps+1) % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    images = []

                    if global_step % args.validation_steps == 0:
                        images = log_validation(
                            val_dataset=val_dataset,
                            backbone=backbone,
                            slot_attn=slot_attn if args.train_slot else None,
                            unet=unet,
                            vae=vae,
                            scheduler=noise_scheduler,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            global_step=global_step,
                            object_encoder_cnn=object_encoder_cnn,
                            num_slots=num_slots
                        )
            if bce_loss is not None:
                logs = {"loss": loss.detach().item(
                ), "lr": lr_scheduler.get_last_lr()[0], "mask_loss": bce_loss.detach().item() ,
                "recon_loss": original_loss.detach().item()}
            else:
                logs = {"loss": loss.detach().item(
                ), "lr": lr_scheduler.get_last_lr()[0],
                "recon_loss": original_loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(
            args.output_dir, f"checkpoint-{global_step}-last")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
