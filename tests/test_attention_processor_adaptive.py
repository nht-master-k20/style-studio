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


def batch4_input(seed=1, shift=0.5):
    """Latent khac nhau moi buoc denoise duoc mo phong bang seed khac nhau."""
    torch.manual_seed(seed)
    x = torch.randn(4, 3, 16)
    x[1] = x[1] + shift   # student khac teacher
    x[3] = x[3] + shift
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


def test_no_report_on_first_fused_step():
    """Buoc fusion dau tien chua co map teacher truoc -> khong report, nhung van fuse."""
    ctrl = SpyController(active=True)
    proc = AttnProcessor2_0_hijack(fuSAttn=True, end_fusion=0, attn_name="layer_x",
                                   num_inference_step=4, fusion_controller=ctrl)
    make_attn(proc)(batch4_input(seed=1))
    assert ctrl.is_active_calls == [1]
    assert ctrl.reports == []      # chua co gi de so sanh o buoc 1


def test_reports_teacher_temporal_change_from_second_step():
    """Tu buoc 2: report d(t) = thay doi cua map teacher giua hai buoc (>0 khi latent doi)."""
    ctrl = SpyController(active=True)
    proc = AttnProcessor2_0_hijack(fuSAttn=True, end_fusion=0, attn_name="layer_x",
                                   num_inference_step=4, fusion_controller=ctrl)
    attn = make_attn(proc)
    attn(batch4_input(seed=1))     # buoc 1: luu map, khong report
    attn(batch4_input(seed=2))     # buoc 2: co map truoc -> report
    assert ctrl.is_active_calls == [1, 2]
    assert len(ctrl.reports) == 1
    name, step, d = ctrl.reports[0]
    assert name == "layer_x" and step == 2 and d > 0


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


def test_temporal_divergence_matches_manual_computation():
    ctrl = SpyController(active=True)
    proc = AttnProcessor2_0_hijack(fuSAttn=True, end_fusion=0, attn_name="t",
                                   num_inference_step=4, fusion_controller=ctrl)
    attn = make_attn(proc)
    x1, x2 = batch4_input(seed=1), batch4_input(seed=2)

    def teacher_map(x):
        # map teacher (index 2), head-averaged, giong het processor
        q = attn.to_q(x).view(4, 3, 2, 8).transpose(1, 2)
        k = attn.to_k(x).view(4, 3, 2, 8).transpose(1, 2)
        probs = (q @ k.transpose(-2, -1) / (8 ** 0.5)).softmax(dim=-1)
        return probs[2].mean(dim=0)

    expected_d = (teacher_map(x2) - teacher_map(x1)).abs().mean().item()
    attn(x1)
    attn(x2)
    assert len(ctrl.reports) == 1
    _, step, d = ctrl.reports[0]
    assert step == 2 and d == pytest.approx(expected_d, rel=1e-3)


def test_prev_map_reset_between_generates():
    """denoise_step ve 1 (generate moi) phai quen map cua run truoc -> buoc 1 khong report."""
    ctrl = SpyController(active=True)
    proc = AttnProcessor2_0_hijack(fuSAttn=True, end_fusion=0, attn_name="t",
                                   num_inference_step=2, fusion_controller=ctrl)
    attn = make_attn(proc)
    attn(batch4_input(seed=1))     # step 1
    attn(batch4_input(seed=2))     # step 2 -> report; denoise_step == num_inference_step -> reset ve 0
    assert [s for _, s, _ in ctrl.reports] == [2]
    attn(batch4_input(seed=3))     # generate moi, step 1 -> KHONG report (da quen map)
    assert [s for _, s, _ in ctrl.reports] == [2]   # van chi 1 report


def test_registers_with_real_controller():
    ctrl = FusionController()
    AttnProcessor2_0_hijack(fuSAttn=True, end_fusion=0, attn_name="layer_z",
                            num_inference_step=4, fusion_controller=ctrl)
    assert "layer_z" in ctrl.layers


def test_end_to_end_with_real_controller_stops():
    """Controller thuc, 1 layer, min_steps=1, rho lon -> stop o buoc 2 (report dau tien)."""
    ctrl = FusionController(rho=2.0, end_fusion_max=30, min_steps=1)
    proc = AttnProcessor2_0_hijack(fuSAttn=True, end_fusion=0, attn_name="only",
                                   num_inference_step=4, fusion_controller=ctrl)
    attn = make_attn(proc)
    attn(batch4_input(seed=1))       # buoc 1: fuse, chua report
    assert not ctrl.stopped and ctrl.history == []
    attn(batch4_input(seed=2))       # buoc 2: report -> r=1.0 <= 2.0 va step>=min_steps -> stop
    assert ctrl.stopped and ctrl.stop_step == 2
    attn(batch4_input(seed=3))       # buoc 3: is_active False -> khong report moi
    assert len(ctrl.history) == 1
