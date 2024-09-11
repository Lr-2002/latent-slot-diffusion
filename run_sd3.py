import torch
from diffusers  import StableDiffusion3Pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    'stabilityai/stable-diffusion-3-medium-diffusers',
    text_encoder_3=None,
    tokenizer_3 = None,
    torch_dtype=torch.float16)
pipe = pipe.to('cuda')
image = pipe(
    'A cat hold a pen',
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0
).images[0]
image.save('./test_sd3_image.png')