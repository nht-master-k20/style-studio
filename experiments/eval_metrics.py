"""Tinh CLIP-T, CLIP-I(style), LPIPS layout-stability tu outputs cua run_experiments.py.

  python experiments/eval_metrics.py --out_root experiments/outputs \
      --styles_dir experiments/styles --results_dir experiments/results
"""
import argparse
import glob
import itertools
import json
import os


def collect_rows(out_root, styles_dir, img_feat_fn, txt_feat_fn):
    """img_feat_fn/txt_feat_fn tra ve tensor da normalize (1, D)."""
    from PIL import Image

    style_feats = {}
    rows = []
    for cond_dir in sorted(p for p in glob.glob(os.path.join(out_root, "*")) if os.path.isdir(p)):
        for jpath in sorted(glob.glob(os.path.join(cond_dir, "*.json"))):
            with open(jpath) as f:
                meta = json.load(f)
            # Derive the condition label from the JSON content itself (written by
            # run_experiments.py's run_case()) rather than the output directory
            # basename, so this stays correct even if directory naming ever
            # diverges from condition/tag. Mirrors run_experiments.py's own
            # out_dir naming: args.condition + (f"_{args.tag}" if args.tag else "").
            cond = meta["condition"] + (f"_{meta['tag']}" if meta.get("tag") else "")
            gen = Image.open(jpath[:-5] + ".jpg").convert("RGB")
            gen_f = img_feat_fn(gen)
            sname = meta["style"]
            if sname not in style_feats:
                style_img = Image.open(os.path.join(styles_dir, sname)).convert("RGB")
                style_feats[sname] = img_feat_fn(style_img)
            txt_f = txt_feat_fn(meta["prompt"])
            rows.append({
                "condition": cond,
                "prompt_idx": meta["prompt_idx"],
                "style": sname,
                "clip_t": float(gen_f @ txt_f.T),
                "clip_i_style": float(gen_f @ style_feats[sname].T),
                "stop_step": (meta.get("fusion") or {}).get("stop_step"),
                "elapsed_sec": meta.get("elapsed_sec"),
                "peak_vram_gb": meta.get("peak_vram_gb"),
            })
    return rows


def layout_lpips(out_root, dist_fn):
    """dist_fn(PIL, PIL) -> float. Mean pairwise LPIPS giua cac style cung prompt."""
    from PIL import Image

    rows = []
    for cond_dir in sorted(p for p in glob.glob(os.path.join(out_root, "*")) if os.path.isdir(p)):
        # This function only globs .jpg files by design (image-only grouping),
        # so it doesn't otherwise open any JSON. To still derive `cond` from JSON
        # content (per Fix 1) rather than the directory basename, peek at one
        # representative case's JSON per directory -- all cases sharing a
        # directory necessarily share the same condition/tag by construction of
        # run_experiments.py's out_dir naming, so any case in the directory is
        # representative. Falls back to the directory basename only if no JSON
        # is present (e.g. a stray .jpg with no matching .json).
        json_paths = sorted(glob.glob(os.path.join(cond_dir, "*.json")))
        if json_paths:
            with open(json_paths[0]) as f:
                rep_meta = json.load(f)
            cond = rep_meta["condition"] + (f"_{rep_meta['tag']}" if rep_meta.get("tag") else "")
        else:
            cond = os.path.basename(cond_dir)
        groups = {}
        for jpg in sorted(glob.glob(os.path.join(cond_dir, "*.jpg"))):
            pi = int(os.path.basename(jpg)[1:3])
            groups.setdefault(pi, []).append(jpg)
        for pi, paths in sorted(groups.items()):
            if len(paths) < 2:
                continue
            dists = [dist_fn(Image.open(a).convert("RGB"), Image.open(b).convert("RGB"))
                     for a, b in itertools.combinations(paths, 2)]
            rows.append({"condition": cond, "prompt_idx": pi,
                         "lpips_mean": sum(dists) / len(dists)})
    return rows


def make_clip_fns(model_id, device):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def img_feat(img):
        inputs = processor(images=img, return_tensors="pt").to(device)
        f = model.get_image_features(**inputs)
        return (f / f.norm(dim=-1, keepdim=True)).cpu()

    @torch.no_grad()
    def txt_feat(text):
        inputs = processor(text=[text], return_tensors="pt",
                           padding=True, truncation=True).to(device)
        f = model.get_text_features(**inputs)
        return (f / f.norm(dim=-1, keepdim=True)).cpu()

    return img_feat, txt_feat


def make_lpips_fn(device):
    import lpips
    import torch
    import torchvision.transforms.functional as TF

    loss = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def dist(img_a, img_b):
        ts = [TF.to_tensor(im.resize((256, 256))).mul(2).sub(1).unsqueeze(0).to(device)
              for im in (img_a, img_b)]
        return float(loss(ts[0], ts[1]))

    return dist


def main():
    import pandas as pd
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="experiments/outputs")
    ap.add_argument("--styles_dir", default="experiments/styles")
    ap.add_argument("--results_dir", default="experiments/results")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)

    img_feat, txt_feat = make_clip_fns(args.clip_model, device)
    rows = collect_rows(args.out_root, args.styles_dir, img_feat, txt_feat)
    pd.DataFrame(rows).to_csv(os.path.join(args.results_dir, "results.csv"), index=False)
    print(f"results.csv: {len(rows)} rows")

    lrows = layout_lpips(args.out_root, make_lpips_fn(device))
    pd.DataFrame(lrows).to_csv(os.path.join(args.results_dir, "lpips_layout.csv"), index=False)
    print(f"lpips_layout.csv: {len(lrows)} rows")


if __name__ == "__main__":
    main()
