import numpy as np
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import torch
from ip_adapter.utils import BLOCKS as BLOCKS
from ip_adapter.utils import controlnet_BLOCKS as controlnet_BLOCKS
import cv2
from PIL import Image
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
)

from ip_adapter import StyleStudio_Adapter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16

def main(args):
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    image_encoder_path = "h94/IP-Adapter/sdxl_models/image_encoder"
    csgo_ckpt ='InstantX/CSGO/csgo_4_32.bin'
    pretrained_vae_name_or_path ='madebyollin/sdxl-vae-fp16-fix'

    vae = AutoencoderKL.from_pretrained(pretrained_vae_name_or_path,torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
        vae=vae,
    )
    pipe.enable_vae_tiling()

    target_style_blocks = BLOCKS['style']

    stylestudio = StyleStudio_Adapter(
        pipe, image_encoder_path, csgo_ckpt, device, num_style_tokens=32,
        target_style_blocks=target_style_blocks,
        controlnet_adapter=False,
        style_model_resampler=True,

        fuAttn=args.fuAttn,
        fuSAttn=args.fuSAttn,
        fuIPAttn=args.fuIPAttn,
        fuScale=args.fuScale,
        end_fusion=args.end_fusion,
        adainIP=args.adainIP,
        adaptive_fusion=args.adaptive_fusion,
        rho=args.rho,
        end_fusion_max=args.end_fusion_max,
    )

    seed = 42
    generator = torch.Generator("cuda").manual_seed(seed)
    if args.fuAttn or args.fuSAttn or args.fuIPAttn:
        # for teacher model
        num_sample=2
    else:
        num_sample=1 
    
    if args.adainIP:
        print("enable cross modal adain")
    
    if num_sample == 2:
        init_latents = torch.randn((1, 4, 128, 128), generator=generator, device="cuda", dtype=torch.float16)
        init_latents = init_latents.repeat(num_sample, 1, 1, 1)
        assert torch.equal(init_latents[0], init_latents[1]) is True
        print("enable teacher model")
    else:
        init_latents = torch.randn((num_sample, 4, 128, 128), generator=generator, device="cuda", dtype=torch.float16)
    
    style_path = args.style_path
    prompt = args.prompt

    style_image = cv2.resize(cv2.imread(style_path), (512, 512))
    style_image = Image.fromarray(cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB))
    
    if args.neg_style_path is None:
        neg_pil_style_image = None
    else:
        neg_style_img = cv2.resize(cv2.imread(args.neg_style_path), (512, 512))
        neg_pil_style_image = Image.fromarray(cv2.cvtColor(neg_style_img, cv2.COLOR_BGR2RGB))

    import json, time
    t0 = time.time()
    images = stylestudio.generate(
        pil_style_image=style_image,
        neg_pil_style_image=neg_pil_style_image,
        prompt=prompt,
        negative_prompt="",
        height=1024,
        width=1024,
        style_scale=1.0,
        guidance_scale=5,
        num_images_per_prompt=1,
        num_samples=num_sample,
        num_inference_steps=args.num_inference_steps,
        end_fusion=args.end_fusion,

        generator=generator,
        latents=init_latents,
    )
    elapsed = time.time() - t0
    if args.fuAttn or args.fuSAttn or args.fuIPAttn:
        assert len(images) == 2
        images[1].save("./test.jpg")
    else:
        images[0].save("./test.jpg")

    print("final")

    if args.log_json:
        log = {"args": vars(args), "elapsed_sec": round(elapsed, 1)}
        if stylestudio.fusion_controller is not None:
            log["fusion"] = stylestudio.fusion_controller.to_dict()
        with open(args.log_json, "w") as f:
            json.dump(log, f, indent=2)
        print(f"log saved to {args.log_json}")

    
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fuAttn", action="store_true")
    parser.add_argument("--fuSAttn", action="store_true")
    parser.add_argument("--fuIPAttn", action="store_true")
    parser.add_argument("--fuScale", type=int, default=0)
    parser.add_argument("--end_fusion", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--adainIP", action="store_true")
    parser.add_argument("--prompt", type=str, default="A red apple")
    parser.add_argument("--style_path", type=str, default="assets/style1.jpg")
    parser.add_argument("--neg_style_path", type=str, default=None)
    parser.add_argument("--adaptive_fusion", action="store_true")
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--end_fusion_max", type=int, default=30)
    parser.add_argument("--log_json", type=str, default=None)

    args = parser.parse_args()

    main(args)