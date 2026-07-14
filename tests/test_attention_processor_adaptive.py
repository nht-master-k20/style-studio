import torch
import pytest
from diffusers.models.attention_processor import Attention

from ip_adapter.attention_processor import AttnProcessor2_0_hijack
from ip_adapter.fusion_controller import FusionController


class SpyController:
    """Duck-typed controller ghi lai moi call."""
    def __init__(self, active=True):
        self.active = active
        self.reports = []          # [(layer, step, d)]
        self.is_active_calls = []

    def register(self, name):
        pass

    def is_active(self, step):
        self.is_active_calls.append(step)
        return self.active

    def report(self, name, step, d):
        self.reports.append((name, step, d))


def make_attn(processor):
    torch.manual_seed(0)
    attn = Attention(query_dim=16, heads=2, dim_head=8)
    attn.set_processor(processor)
    return attn


def batch4_input():
    torch.manual_seed(1)
    x = torch.randn(4, 3, 16)
    x[1] = x[1] + 0.5   # student khac teacher de attention map khac nhau
    x[3] = x[3] + 0.5
    return x


def test_legacy_mode_unchanged_without_controller():
    """Khong co controller: fusion theo denoise_step <= end_fusion nhu code goc."""
    x = batch4_input()
    proc_fused = AttnProcessor2_0_hijack(fuSAttn=True, end_fusion=1,
                                         attn_name="t", num_inference_step=4)
    proc_plain = AttnProcessor2_0_hijack(fuSAttn=False, end_fusion=0,
                                         attn_name="t", num_inference_step=4)
    out_fused = make_attn(proc_fused)(x)
    out_plain = make_attn(proc_plain)(x)
    # teacher (index 0, 2) khong bi anh huong boi fusion
    assert torch.allclose(out_fused[0], out_plain[0], atol=1e-5)
    # student (index 1, 3) bi hijack -> khac voi khong hijack
    assert not torch.allclose(out_fused[1], out_plain[1], atol=1e-4)


def test_controller_gates_fusion_on():
    ctrl = SpyController(active=True)
    proc = AttnProcessor2_0_hijack(fuSAttn=True, end_fusion=0,  # end_fusion=0 bi ignore khi co controller
                                   attn_name="layer_x", num_inference_step=4,
                                   fusion_controller=ctrl)
    make_attn(proc)(batch4_input())
    assert ctrl.is_active_calls == [1]
    assert len(ctrl.reports) == 1
    name, step, d = ctrl.reports[0]
    assert name == "layer_x" and step == 1 and d > 0


def test_controller_gates_fusion_off():
    ctrl = SpyController(active=False)
    proc_ctrl = AttnProcessor2_0_hijack(fuSAttn=True, end_fusion=30,
                                        attn_name="t", num_inference_step=4,
                                        fusion_controller=ctrl)
    proc_plain = AttnProcessor2_0_hijack(fuSAttn=False, end_fusion=0,
                                         attn_name="t", num_inference_step=4)
    x = batch4_input()
    out_ctrl = make_attn(proc_ctrl)(x)
    out_plain = make_attn(proc_plain)(x)
    # controller bao inactive -> khong fusion du end_fusion=30, khong report
    assert ctrl.reports == []
    assert torch.allclose(out_ctrl, out_plain, atol=1e-5)


def test_divergence_matches_manual_computation():
    ctrl = SpyController(active=True)
    proc = AttnProcessor2_0_hijack(fuSAttn=True, end_fusion=0, attn_name="t",
                                   num_inference_step=4, fusion_controller=ctrl)
    attn = make_attn(proc)
    x = batch4_input()
    # tinh tay attention probs truoc khi goi
    q = attn.to_q(x).view(4, 3, 2, 8).transpose(1, 2)
    k = attn.to_k(x).view(4, 3, 2, 8).transpose(1, 2)
    probs = (q @ k.transpose(-2, -1) / (8 ** 0.5)).softmax(dim=-1)
    expected_d = (probs[3] - probs[2]).abs().mean().item()
    attn(x)
    _, _, d = ctrl.reports[0]
    assert d == pytest.approx(expected_d, rel=1e-3)


def test_registers_with_real_controller():
    ctrl = FusionController()
    AttnProcessor2_0_hijack(fuSAttn=True, end_fusion=0, attn_name="layer_z",
                            num_inference_step=4, fusion_controller=ctrl)
    assert "layer_z" in ctrl.layers


def test_end_to_end_with_real_controller_stops():
    """Controller thuc: 1 layer, min_steps=1, rho lon -> stop ngay buoc 1, buoc 2 khong report."""
    ctrl = FusionController(rho=2.0, end_fusion_max=30, min_steps=1)
    proc = AttnProcessor2_0_hijack(fuSAttn=True, end_fusion=0, attn_name="only",
                                   num_inference_step=4, fusion_controller=ctrl)
    attn = make_attn(proc)
    x = batch4_input()
    attn(x)                       # buoc 1: fuse + report -> r=1.0 < 2.0 -> stop
    assert ctrl.stopped and ctrl.stop_step == 1
    attn(x)                       # buoc 2: is_active False -> SDPA path
    assert len(ctrl.history) == 1  # khong co report moi
