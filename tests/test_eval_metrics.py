import json
import os

import torch
from PIL import Image

from experiments.eval_metrics import collect_rows, layout_lpips


def make_case(root, cond, pi, style, prompt="a cat", fusion=None):
    d = os.path.join(root, cond)
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"p{pi:02d}__{style}")
    Image.new("RGB", (32, 32), (pi * 40 % 255, 80, 120)).save(stem + ".jpg")
    meta = {"condition": cond, "prompt_idx": pi, "prompt": prompt,
            "style": style + ".jpg", "elapsed_sec": 100.0, "peak_vram_gb": 10.0}
    if fusion:
        meta["fusion"] = fusion
    with open(stem + ".json", "w") as f:
        json.dump(meta, f)


def fake_img_feat(img):
    t = torch.tensor([float(img.size[0]), 1.0, 0.0])
    return (t / t.norm()).unsqueeze(0)


def fake_txt_feat(text):
    t = torch.tensor([1.0, 1.0, 0.0])
    return (t / t.norm()).unsqueeze(0)


def test_collect_rows(tmp_path):
    root = str(tmp_path / "out")
    styles = str(tmp_path / "styles")
    os.makedirs(styles)
    Image.new("RGB", (32, 32), (200, 10, 10)).save(os.path.join(styles, "styleA.jpg"))
    make_case(root, "fixed20", 0, "styleA")
    make_case(root, "adaptive", 0, "styleA",
              fusion={"stop_step": 7, "r_history": [[1, 1.0], [7, 0.1]]})

    rows = collect_rows(root, styles, fake_img_feat, fake_txt_feat)
    assert len(rows) == 2
    by_cond = {r["condition"]: r for r in rows}
    assert by_cond["adaptive"]["stop_step"] == 7
    assert by_cond["fixed20"]["stop_step"] is None
    for r in rows:
        assert -1.0 <= r["clip_t"] <= 1.0
        assert -1.0 <= r["clip_i_style"] <= 1.0


def test_layout_lpips_groups_by_prompt(tmp_path):
    root = str(tmp_path / "out")
    for style in ["styleA", "styleB", "styleC"]:
        make_case(root, "fixed20", 0, style)
    make_case(root, "fixed20", 1, "styleA")  # prompt 1 chi co 1 style -> khong co cap

    def fake_dist(img_a, img_b):
        return 0.5

    rows = layout_lpips(root, fake_dist)
    assert rows == [{"condition": "fixed20", "prompt_idx": 0, "lpips_mean": 0.5}]
