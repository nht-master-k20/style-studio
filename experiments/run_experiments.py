"""Batch runner cho thi nghiem adaptive end_fusion.

Load model MOT lan cho moi condition, loop qua prompt x style, skip case da xong
(co du .jpg + .json) -> resume duoc qua nhieu phien Kaggle.

Vi du:
  python experiments/run_experiments.py --condition fixed20 \
      --prompts experiments/prompts_test.txt --styles_dir experiments/styles \
      --image_encoder_path <path> --csgo_ckpt <path>
  python experiments/run_experiments.py --condition adaptive --rho 0.2 ...
"""
import argparse
import glob
import json
import os
import sys
import time

CONDITIONS = {
    "fixed5": {"end_fusion": 5},
    "fixed10": {"end_fusion": 10},
    "fixed20": {"end_fusion": 20},
    "fixed30": {"end_fusion": 30},
    "adaptive": {"adaptive": True},
}


def plan_runs(prompts, style_paths, out_dir):
    pending = []
    for pi, prompt in enumerate(prompts):
        for sp in style_paths:
            stem = os.path.join(out_dir, f"p{pi:02d}__{os.path.splitext(os.path.basename(sp))[0]}")
            if not (os.path.exists(stem + ".jpg") and os.path.exists(stem + ".json")):
                pending.append((pi, prompt, sp, stem))
    return pending


def load_adapter(args, cond):
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    from ip_adapter.utils import BLOCKS
    from ip_adapter import StyleStudio_Adapter

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, add_watermarker=False, vae=vae,
    )
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    return StyleStudio_Adapter(
        pipe, args.image_encoder_path, args.csgo_ckpt, torch.device("cuda"),
        num_style_tokens=32,
        target_style_blocks=BLOCKS["style"],
        controlnet_adapter=False,
        style_model_resampler=True,
        fuSAttn=True,
        fuScale=0,
        adainIP=True,
        end_fusion=cond.get("end_fusion", 0),
        adaptive_fusion=cond.get("adaptive", False),
        rho=args.rho,
        end_fusion_max=args.end_fusion_max,
    )


def run_case(adapter, args, cond, pi, prompt, style_path, stem):
    import cv2
    import torch
    from PIL import Image

    img = cv2.resize(cv2.imread(style_path), (512, 512))
    style_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    generator = torch.Generator("cuda").manual_seed(42)
    init_latents = torch.randn((1, 4, 128, 128), generator=generator,
                               device="cuda", dtype=torch.float16).repeat(2, 1, 1, 1)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    images = adapter.generate(
        pil_style_image=style_image,
        prompt=prompt,
        negative_prompt="",
        height=1024, width=1024,
        style_scale=1.0, guidance_scale=5,
        num_images_per_prompt=1, num_samples=2,
        num_inference_steps=args.num_steps,
        end_fusion=cond.get("end_fusion", 0),
        generator=generator, latents=init_latents,
    )
    elapsed = time.time() - t0
    images[1].save(stem + ".jpg")   # index 1 = student (index 0 = teacher)
    log = {
        "condition": args.condition, "tag": args.tag,
        "prompt_idx": pi, "prompt": prompt, "style": os.path.basename(style_path),
        "num_steps": args.num_steps, "seed": 42,
        "elapsed_sec": round(elapsed, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    if adapter.fusion_controller is not None:
        log["fusion"] = adapter.fusion_controller.to_dict()
    with open(stem + ".json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"[done] {stem} ({elapsed:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--styles_dir", required=True)
    ap.add_argument("--out_root", default="experiments/outputs")
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--end_fusion_max", type=int, default=30)
    ap.add_argument("--image_encoder_path", default="h94/IP-Adapter/sdxl_models/image_encoder")
    ap.add_argument("--csgo_ckpt", default="InstantX/CSGO/csgo_4_32.bin")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cond = CONDITIONS[args.condition]
    with open(args.prompts) as f:
        prompts = [line.strip() for line in f if line.strip()]
    style_paths = sorted(glob.glob(os.path.join(args.styles_dir, "*.jpg")))
    assert style_paths, f"khong tim thay style .jpg trong {args.styles_dir}"
    out_dir = os.path.join(args.out_root, args.condition + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(out_dir, exist_ok=True)

    pending = plan_runs(prompts, style_paths, out_dir)
    total = len(prompts) * len(style_paths)
    print(f"[runner] condition={args.condition} pending={len(pending)}/{total} -> {out_dir}")
    if args.dry_run:
        for pi, prompt, sp, stem in pending:
            print(f"  p{pi:02d} x {os.path.basename(sp)}")
        return

    adapter = load_adapter(args, cond)
    for case in pending:
        run_case(adapter, args, cond, *case)


if __name__ == "__main__":
    main()
