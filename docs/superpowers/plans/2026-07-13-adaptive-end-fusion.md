# Adaptive Attention Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thay hyperparameter `end_fusion` cố định của StyleStudio bằng cơ chế dừng adaptive dựa trên độ hội tụ attention teacher–student, kèm bộ thí nghiệm định lượng chứng minh adaptive ≥ best fixed mà không cần tune per-case.

**Architecture:** Một `FusionController` (pure Python) được chia sẻ cho mọi self-attention hijack processor; processor báo độ lệch attention teacher–student mỗi bước fusion, controller quyết định dừng toàn cục. Experiment runner load model một lần, loop qua prompt×style có checkpoint/resume. Eval script tính CLIP-T / CLIP-I / LPIPS.

**Tech Stack:** PyTorch, diffusers 0.25.1, transformers 4.45.2, lpips, pandas, matplotlib, pytest. Code chạy trên Kaggle T4 (CUDA); unit test chạy local trên Mac (CPU).

**Spec:** `docs/superpowers/specs/2026-07-13-adaptive-end-fusion-design.md` (đã duyệt).

## Global Constraints

- Repo làm việc: `/Users/namtt/Documents/study/CS2309/StyleStudio` (git clone của Westlake-AGI-Lab/StyleStudio). Mọi commit trên branch `adaptive-fusion`.
- **Tương thích ngược tuyệt đối:** không truyền `fusion_controller`/`adaptive_fusion` → hành vi hệt code gốc (baseline sạch).
- Pin: `diffusers==0.25.1`, `transformers==4.45.2`. Trên Kaggle: KHÔNG cài lại torch (dùng bản có sẵn kèm CUDA).
- Seed cố định **42**, cấu hình generate: 1024×1024, guidance 5, `--fuSAttn --adainIP`, 50 bước (trừ smoke test).
- Mặc định controller: `rho=0.2`, `end_fusion_max=30`, `min_steps=5`.
- Chạy test: `python -m pytest tests/ -v` từ thư mục `StyleStudio/` (cwd nằm trong `sys.path` nên `ip_adapter` import được).
- Python: venv có sẵn tại `/Users/namtt/Documents/study/CS2309/.venv` (python 3.10).

---

## Trạng thái triển khai (cập nhật 2026-07-14)

**Task 0-5: xong**, mỗi task đã qua implementer + task reviewer (spec compliance + code quality), cộng một whole-branch review cuối cho Task 0-5. 23/23 test pass. Ledger đầy đủ tại `.superpowers/sdd/progress.md` trong repo làm việc.

Commit range trên branch `adaptive-fusion`: `46f6902` (Task 1) → `f709699` (Task 2) → `0d795cb` (Task 3) → `182c251` (Task 4) → `ae259e5` (Task 5) → `ab2990d` (fix sau review Task 5). Task 0 không tạo commit (chỉ setup môi trường).

3 sai khác so với code mẫu trong plan (đã duyệt lúc review, đã cập nhật vào code mẫu bên dưới cho khớp thực tế):
1. Task 1: điều kiện dừng `r <= self.rho` (không phải `<` như plan gốc) — xem ghi chú tại Task 1 Step 4.
2. Task 5: `collect_rows`/`layout_lpips` lấy `condition` từ nội dung JSON thay vì tên thư mục.
3. Task 5: `analyze_results.py`'s `main()` guard khi `adaptive_gap()` trả `None`.

**Việc cần xác nhận khi chạy Task 6 trên Kaggle (không chặn tiến độ, chỉ cần lưu ý):**
- `pipe.enable_attention_slicing()` trong `load_adapter()` được gọi **trước** `StyleStudio_Adapter.set_ip_adapter()` — hàm này thay toàn bộ attention processor bằng bản hijack, nhiều khả năng làm mất tác dụng slicing. Nếu OOM trên T4, đây là nghi phạm đầu tiên (early-stop + `enable_vae_tiling()` vẫn giảm VRAM độc lập).
- `plot_figures`'s đường tham chiếu `ρ` trong `r_curves.png` hiện hardcode `0.2` — nên đọc từ JSON sau khi Task 7 chốt ρ, nếu khác 0.2.

---

### Task 0: Branch + môi trường test local

**Files:**
- Không sửa code; chỉ setup.

**Interfaces:**
- Produces: branch `adaptive-fusion`; venv có torch/diffusers/pytest để các task sau chạy test.

- [x] **Step 1: Tạo branch**

```bash
cd /Users/namtt/Documents/study/CS2309/StyleStudio
git checkout -b adaptive-fusion
```

Expected: `Switched to a new branch 'adaptive-fusion'`

- [x] **Step 2: Cài deps test vào venv (CPU, Mac)**

```bash
cd /Users/namtt/Documents/study/CS2309
.venv/bin/pip install torch torchvision diffusers==0.25.1 transformers==4.45.2 \
  huggingface-hub==0.24.6 tokenizers==0.20.1 einops opencv-python pillow pytest "numpy<2"
```

Expected: cài thành công (torch bản macOS arm64, CPU/MPS — nhỏ, không phải bản CUDA).

- [x] **Step 3: Verify import chain của repo hoạt động**

```bash
cd /Users/namtt/Documents/study/CS2309/StyleStudio
../.venv/bin/python -c "from ip_adapter.attention_processor import AttnProcessor2_0_hijack; print('ok')"
```

Expected: `ok`. (Nếu lỗi thiếu module nào, pip install bổ sung đúng module đó rồi chạy lại.)

---

### Task 1: FusionController (pure Python) — TDD

**Files:**
- Create: `ip_adapter/fusion_controller.py`
- Test: `tests/test_fusion_controller.py`

**Interfaces:**
- Produces (Task 2, 3 dùng):
  - `FusionController(rho=0.2, end_fusion_max=30, min_steps=5)`
  - `.register(layer_name: str) -> None`
  - `.is_active(step: int) -> bool`
  - `.report(layer_name: str, step: int, d: float) -> None`
  - `.reset() -> None`
  - `.stopped: bool`, `.stop_step: int | None`, `.history: list[tuple[int, float]]`
  - `.to_dict() -> dict` (JSON-serializable)

- [x] **Step 1: Viết failing tests**

Tạo `tests/test_fusion_controller.py`:

```python
import json
import pytest

from ip_adapter.fusion_controller import FusionController


def make_ctrl(**kw):
    ctrl = FusionController(**kw)
    ctrl.register("layer_a")
    ctrl.register("layer_b")
    return ctrl


def report_step(ctrl, step, d_a, d_b):
    ctrl.report("layer_a", step, d_a)
    ctrl.report("layer_b", step, d_b)


def test_active_before_any_report():
    ctrl = make_ctrl()
    assert ctrl.is_active(1)
    assert not ctrl.stopped


def test_stops_when_converged_after_min_steps():
    ctrl = make_ctrl(rho=0.2, min_steps=3)
    report_step(ctrl, 1, 1.0, 1.0)   # baseline
    report_step(ctrl, 2, 0.5, 0.5)   # r=0.5
    report_step(ctrl, 3, 0.1, 0.1)   # r=0.1 < 0.2 va step >= min_steps -> stop
    assert ctrl.stopped
    assert ctrl.stop_step == 3
    assert not ctrl.is_active(4)


def test_no_stop_before_min_steps():
    ctrl = make_ctrl(rho=0.2, min_steps=5)
    report_step(ctrl, 1, 1.0, 1.0)
    report_step(ctrl, 2, 0.01, 0.01)  # r rat thap nhung step < min_steps
    assert not ctrl.stopped
    assert ctrl.is_active(3)


def test_end_fusion_max_cap():
    ctrl = make_ctrl(rho=0.0, end_fusion_max=10)  # rho=0 -> khong bao gio stop theo r
    assert ctrl.is_active(10)
    assert not ctrl.is_active(11)


def test_baseline_is_running_max():
    # do lech tang o dau (layer truoc cross-attn dau tien co d(1)=0) roi giam
    ctrl = make_ctrl(rho=0.2, min_steps=2)
    report_step(ctrl, 1, 0.0, 1.0)
    report_step(ctrl, 2, 2.0, 0.9)   # layer_a ramp len 2.0 -> baseline_a=2.0
    # r(2) = mean(2.0/2.0, 0.9/1.0) = 0.95 -> khong stop
    assert not ctrl.stopped
    report_step(ctrl, 3, 0.2, 0.1)   # r = mean(0.1, 0.1) = 0.1 < 0.2 -> stop
    assert ctrl.stopped
    assert ctrl.stop_step == 3


def test_all_zero_layer_treated_converged():
    ctrl = make_ctrl(rho=0.2, min_steps=1)
    report_step(ctrl, 1, 0.0, 1.0)
    report_step(ctrl, 2, 0.0, 0.1)   # layer_a luon 0 -> r_a=0; r=mean(0, 0.1)=0.05 -> stop
    assert ctrl.stopped


def test_decision_waits_for_all_layers():
    ctrl = make_ctrl(rho=1.0, min_steps=1)  # rho=1.0: stop ngay khi du bao cao
    ctrl.report("layer_a", 1, 0.5)
    assert not ctrl.stopped               # layer_b chua bao cao
    ctrl.report("layer_b", 1, 0.5)
    assert ctrl.stopped                   # du bao cao -> quyet dinh


def test_report_after_stop_is_noop():
    ctrl = make_ctrl(rho=1.0, min_steps=1)
    report_step(ctrl, 1, 0.5, 0.5)
    assert ctrl.stopped
    ctrl.report("layer_a", 2, 9.9)        # khong crash, khong doi trang thai
    assert ctrl.stop_step == 1


def test_reset():
    ctrl = make_ctrl(rho=1.0, min_steps=1)
    report_step(ctrl, 1, 0.5, 0.5)
    assert ctrl.stopped
    ctrl.reset()
    assert not ctrl.stopped
    assert ctrl.stop_step is None
    assert ctrl.history == []
    assert ctrl.is_active(1)
    # layers van giu sau reset
    report_step(ctrl, 1, 0.5, 0.5)
    assert ctrl.stopped


def test_history_records_r_per_step():
    ctrl = make_ctrl(rho=0.0, min_steps=1)
    report_step(ctrl, 1, 1.0, 1.0)
    report_step(ctrl, 2, 0.5, 0.5)
    assert ctrl.history == [(1, 1.0), (2, 0.5)]


def test_to_dict_json_serializable():
    ctrl = make_ctrl(rho=0.2)
    report_step(ctrl, 1, 1.0, 1.0)
    s = json.dumps(ctrl.to_dict())
    assert "stop_step" in s and "r_history" in s
```

- [x] **Step 2: Chạy test, xác nhận FAIL**

```bash
cd /Users/namtt/Documents/study/CS2309/StyleStudio
../.venv/bin/python -m pytest tests/test_fusion_controller.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'ip_adapter.fusion_controller'`

- [x] **Step 3: Implement `ip_adapter/fusion_controller.py`**

```python
class FusionController:
    """Decide when to stop teacher attention fusion, based on student->teacher convergence.

    Each self-attn hijack processor register()s at setup and report()s the divergence
    d = mean|attn_student - attn_teacher| (cond branch) at every fusion step, BEFORE the
    teacher map is copied over. Once all registered layers have reported for step t, the
    controller computes r(t) and decides whether to stop fusing from step t+1 onward.

    Per-layer normalization uses a RUNNING MAX baseline (not d at step 1): layers placed
    before the first cross-attention see identical teacher/student hidden states at step 1
    (same init latents), so their d(1) is ~0 and would break ratio normalization.
    """

    _EPS = 1e-12

    def __init__(self, rho=0.2, end_fusion_max=30, min_steps=5):
        self.rho = rho
        self.end_fusion_max = end_fusion_max
        self.min_steps = min_steps
        self.layers = []
        self.reset()

    def register(self, layer_name):
        if layer_name not in self.layers:
            self.layers.append(layer_name)

    def reset(self):
        self.stopped = False
        self.stop_step = None
        self._baseline = {}   # layer_name -> running max of d
        self._pending = {}    # step -> {layer_name: d}
        self.history = []     # [(step, r)]

    def is_active(self, step):
        return (not self.stopped) and step <= self.end_fusion_max

    def report(self, layer_name, step, d):
        if self.stopped:
            return
        self._pending.setdefault(step, {})[layer_name] = d
        if len(self._pending[step]) == len(self.layers):
            self._decide(step)

    def _decide(self, step):
        ratios = []
        for name, d in self._pending[step].items():
            base = max(self._baseline.get(name, 0.0), d)
            self._baseline[name] = base
            ratios.append(d / base if base > self._EPS else 0.0)
        r = sum(ratios) / len(ratios)
        self.history.append((step, r))
        del self._pending[step]
        if step >= self.min_steps and r <= self.rho:
            self.stopped = True
            self.stop_step = step

    def to_dict(self):
        return {
            "rho": self.rho,
            "end_fusion_max": self.end_fusion_max,
            "min_steps": self.min_steps,
            "stop_step": self.stop_step,
            "r_history": [[s, round(r, 6)] for s, r in self.history],
        }
```

- [x] **Step 4: Chạy test, xác nhận PASS**

```bash
../.venv/bin/python -m pytest tests/test_fusion_controller.py -v
```

Expected: 11 passed.

Lưu ý case `test_stops_when_converged_after_min_steps`: bước 3 có `d=0.1`, baseline=1.0 → r=0.1 < 0.2 và `3 >= min_steps=3` → stop. Nếu test fail vì so sánh float, kiểm tra lại logic chứ không sửa test.

**[Đã xác nhận khi triển khai]** Điều kiện dùng `r <= self.rho` (không phải `r < self.rho`): hai test `test_decision_waits_for_all_layers` và `test_report_after_stop_is_noop` dùng `rho=1.0` và kỳ vọng dừng khi `r` tính ra đúng bằng `1.0` — với `<` nghiêm ngặt, `1.0 < 1.0` là `False` nên 2 test đó sẽ fail với đúng code phía trên. Đây là điểm spec (§3.2.4 dùng `<`) và bộ test tự mâu thuẫn nhau; đã chốt dùng `<=` (không ảnh hưởng ở `rho=0.2` dùng trong thực nghiệm chính).

- [x] **Step 5: Commit**

```bash
git add ip_adapter/fusion_controller.py tests/test_fusion_controller.py
git commit -m "feat: add FusionController for adaptive teacher-fusion stopping"
```

---

### Task 2: Adaptive gate trong `AttnProcessor2_0_hijack` — TDD

**Files:**
- Modify: `ip_adapter/attention_processor.py:1186-1294` (class `AttnProcessor2_0_hijack`)
- Test: `tests/test_attention_processor_adaptive.py`

**Interfaces:**
- Consumes: `FusionController` (Task 1) — duck-typed: chỉ cần `.register/.is_active/.report`.
- Produces: `AttnProcessor2_0_hijack(..., fusion_controller=None)` — Task 3 truyền controller vào đây qua `set_ip_adapter`.

- [x] **Step 1: Viết failing tests**

Tạo `tests/test_attention_processor_adaptive.py`:

```python
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
```

- [x] **Step 2: Chạy test, xác nhận FAIL**

```bash
../.venv/bin/python -m pytest tests/test_attention_processor_adaptive.py -v
```

Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'fusion_controller'` (test đầu `test_legacy_mode_unchanged_without_controller` có thể PASS vì không đụng controller — chấp nhận).

- [x] **Step 3: Sửa `AttnProcessor2_0_hijack`**

Trong `ip_adapter/attention_processor.py`, sửa `__init__` (dòng ~1186-1209) — thêm param và register:

```python
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        save_in_unet='down',
        atten_control=None,
        fuSAttn=False,
        fuScale=0,
        end_fusion=0,
        attn_name=None,
        num_inference_step=50,
        fusion_controller=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.atten_control = atten_control
        self.save_in_unet = save_in_unet

        self.fuSAttn = fuSAttn
        self.fuScale = fuScale
        self.denoise_step = 0
        self.end_fusion = end_fusion
        self.name = attn_name
        self.num_inference_step = num_inference_step
        self.fusion_controller = fusion_controller
        if self.fusion_controller is not None and self.fuSAttn:
            self.fusion_controller.register(attn_name)
```

Sửa nhánh fusion trong `__call__` (dòng ~1262-1274). Thay:

```python
        if self.fuSAttn and self.denoise_step <= self.end_fusion:
            assert query.shape[0] == 4
            scale_factor = 1 / math.sqrt(torch.tensor(head_dim, dtype=query.dtype))
            attn_probs = (torch.matmul(query, key.transpose(-2, -1)) * scale_factor).softmax(dim=-1)
            attn_probs[1] = attn_probs[0]
            attn_probs[3] = attn_probs[2]
            hidden_states = torch.matmul(attn_probs, value)
```

bằng:

```python
        if self.fusion_controller is not None:
            fusion_now = self.fuSAttn and self.fusion_controller.is_active(self.denoise_step)
        else:
            fusion_now = self.fuSAttn and self.denoise_step <= self.end_fusion
        if fusion_now:
            assert query.shape[0] == 4
            scale_factor = 1 / math.sqrt(torch.tensor(head_dim, dtype=query.dtype))
            attn_probs = (torch.matmul(query, key.transpose(-2, -1)) * scale_factor).softmax(dim=-1)
            if self.fusion_controller is not None:
                # cond branch divergence, measured BEFORE the teacher copy
                d = (attn_probs[3] - attn_probs[2]).abs().mean().item()
                self.fusion_controller.report(self.name, self.denoise_step, d)
            attn_probs[1] = attn_probs[0]
            attn_probs[3] = attn_probs[2]
            hidden_states = torch.matmul(attn_probs, value)
```

(phần `else:` với SDPA giữ nguyên).

- [x] **Step 4: Chạy toàn bộ test, xác nhận PASS**

```bash
../.venv/bin/python -m pytest tests/ -v
```

Expected: 17 passed (11 controller + 6 processor).

- [x] **Step 5: Commit**

```bash
git add ip_adapter/attention_processor.py tests/test_attention_processor_adaptive.py
git commit -m "feat: adaptive fusion gate in AttnProcessor2_0_hijack via FusionController"
```

---

### Task 3: Wire vào `StyleStudio_Adapter` + flags CLI + fix bug end_fusion

**Files:**
- Modify: `ip_adapter/ip_adapter.py:868-926` (`StyleStudio_Adapter.__init__`), `:940-944` (`set_ip_adapter`), `:1109-1135` (`generate`)
- Modify: `infer_StyleStudio.py`

**Interfaces:**
- Consumes: `FusionController` (Task 1), processor param `fusion_controller` (Task 2).
- Produces (Task 4 dùng): `StyleStudio_Adapter(..., adaptive_fusion=False, rho=0.2, end_fusion_max=30)`; thuộc tính `adapter.fusion_controller` (None nếu không adaptive); CLI flags `--adaptive_fusion --rho --end_fusion_max --log_json` trong `infer_StyleStudio.py`.

- [x] **Step 1: Sửa `StyleStudio_Adapter.__init__`**

Thêm 3 param vào chữ ký (sau `save_attn_map=False,`):

```python
                save_attn_map=False,
                adaptive_fusion=False,
                rho=0.2,
                end_fusion_max=30,
                ):
```

Ngay TRƯỚC dòng `self.pipe = sd_pipe.to(self.device)` (hiện ở dòng ~910), thêm:

```python
        self.fusion_controller = None
        if adaptive_fusion:
            from .fusion_controller import FusionController
            self.fusion_controller = FusionController(rho=rho, end_fusion_max=end_fusion_max)
            print(f"adaptive fusion: rho={rho}, end_fusion_max={end_fusion_max}")
```

(Bắt buộc trước `set_ip_adapter()` vì hàm đó cần `self.fusion_controller`.)

- [x] **Step 2: Truyền controller trong `set_ip_adapter`**

Trong `StyleStudio_Adapter.set_ip_adapter` (dòng ~941-944), sửa:

```python
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor_hijack(
                                        fuSAttn=self.fuSAttn,
                                        end_fusion=self.end_fusion,
                                        attn_name=name,
                                        fusion_controller=self.fusion_controller)
```

- [x] **Step 3: Reset controller mỗi lần generate**

Trong `StyleStudio_Adapter.generate` (dòng ~1135), ngay SAU `self.set_num_inference_step(num_T=num_inference_steps)`, thêm:

```python
        if self.fusion_controller is not None:
            self.fusion_controller.reset()
```

- [x] **Step 4: Sửa `infer_StyleStudio.py` — flags, fix bug end_fusion, JSON log**

4a. Thêm args (trong block `argparse` cuối file):

```python
    parser.add_argument("--adaptive_fusion", action="store_true")
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--end_fusion_max", type=int, default=30)
    parser.add_argument("--log_json", type=str, default=None)
```

4b. Truyền vào constructor — trong `main()`, thêm vào cuối call `StyleStudio_Adapter(...)`:

```python
        adainIP=args.adainIP,
        adaptive_fusion=args.adaptive_fusion,
        rho=args.rho,
        end_fusion_max=args.end_fusion_max,
    )
```

4c. **Fix bug:** call `stylestudio.generate(...)` trong `main()` không truyền `end_fusion` → default `end_fusion=20` của `generate()` GHI ĐÈ giá trị `--end_fusion` từ CLI (qua `set_endFusion`). Thêm vào call generate:

```python
        num_inference_steps=args.num_inference_steps,
        end_fusion=args.end_fusion,
```

4d. JSON log + timing — bọc quanh call generate trong `main()`:

```python
    import json, time
    t0 = time.time()
    images = stylestudio.generate(
        ...  # nhu tren, giu nguyen cac kwargs khac
    )
    elapsed = time.time() - t0
```

và sau khi save ảnh (`images[...].save("./test.jpg")`), thêm:

```python
    if args.log_json:
        log = {"args": vars(args), "elapsed_sec": round(elapsed, 1)}
        if stylestudio.fusion_controller is not None:
            log["fusion"] = stylestudio.fusion_controller.to_dict()
        with open(args.log_json, "w") as f:
            json.dump(log, f, indent=2)
        print(f"log saved to {args.log_json}")
```

- [x] **Step 5: Verify compile + test suite vẫn xanh**

```bash
../.venv/bin/python -m py_compile infer_StyleStudio.py ip_adapter/ip_adapter.py
../.venv/bin/python -m pytest tests/ -v
```

Expected: compile OK, 17 passed. (Smoke test model thật để dành cho Task 6 trên Kaggle — Mac không đủ RAM.)

- [x] **Step 6: Commit**

```bash
git add ip_adapter/ip_adapter.py infer_StyleStudio.py
git commit -m "feat: wire adaptive fusion into StyleStudio_Adapter and CLI; fix end_fusion passthrough bug"
```

---

### Task 4: Experiment assets + batch runner có resume

**Files:**
- Create: `experiments/prompts_test.txt`, `experiments/prompts_dev.txt`, `experiments/styles/README.md`, `experiments/run_experiments.py`
- Test: `tests/test_run_experiments.py`

**Interfaces:**
- Consumes: `StyleStudio_Adapter(..., adaptive_fusion=, rho=, end_fusion_max=)` (Task 3).
- Produces: outputs dạng `experiments/outputs/<condition>[_<tag>]/p<idx:02d>__<style_stem>.jpg|.json`; hàm `plan_runs(prompts, style_paths, out_dir) -> list[(pi, prompt, style_path, stem)]` (Task 5 đọc layout này).

- [x] **Step 1: Tạo prompt files và styles README**

`experiments/prompts_test.txt` (10 dòng):

```
A goat is playing on the beach
A red car in the city street
An astronaut riding a horse
A cup of coffee on a wooden table
A lighthouse on a rocky coast
A woman reading a book in a park
A plate of sushi
A castle on a hill at sunset
A bicycle leaning against a brick wall
A panda eating bamboo in a forest
```

`experiments/prompts_dev.txt` (2 dòng, không trùng test):

```
A dog wearing sunglasses
A sailboat on a calm lake
```

`experiments/styles/README.md`:

```markdown
# Style images cho thi nghiem

Copy 6 anh style vao thu muc nay (jpg):
- style1.jpg, style2.jpg, style3.jpg  <- tu ../../assets/
- style4.jpg                          <- tu Kaggle dataset tnamt9/assets
- style5.jpg, style6.jpg              <- 2 anh moi, chon chat lieu KHAC biet
  (vd: 1 anh son dau day texture, 1 anh flat vector/minimal) de test set da dang.

Dev split dung style1.jpg + style4.jpg (cung file, khac prompt — prompts_dev.txt
khong trung prompts_test.txt nen khong leak prompt; ghi ro trong report).

Runner glob *.jpg trong thu muc nay — so luong style linh hoat.
```

```bash
cp assets/style1.jpg assets/style2.jpg assets/style3.jpg experiments/styles/
```

- [x] **Step 2: Viết failing test cho `plan_runs`**

Tạo `tests/test_run_experiments.py`:

```python
import os

from experiments.run_experiments import plan_runs


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("x")


def test_plan_runs_full_pending(tmp_path):
    out = str(tmp_path / "cond")
    runs = plan_runs(["p one", "p two"], ["/s/styleA.jpg", "/s/styleB.jpg"], out)
    assert len(runs) == 4
    pi, prompt, sp, stem = runs[0]
    assert pi == 0 and prompt == "p one" and sp == "/s/styleA.jpg"
    assert stem == os.path.join(out, "p00__styleA")


def test_plan_runs_skips_completed(tmp_path):
    out = str(tmp_path / "cond")
    touch(os.path.join(out, "p00__styleA.jpg"))
    touch(os.path.join(out, "p00__styleA.json"))
    touch(os.path.join(out, "p01__styleA.jpg"))  # thieu .json -> van pending
    runs = plan_runs(["p one", "p two"], ["/s/styleA.jpg"], out)
    stems = [r[3] for r in runs]
    assert os.path.join(out, "p00__styleA") not in stems
    assert os.path.join(out, "p01__styleA") in stems
    assert len(runs) == 1
```

Tạo file rỗng `experiments/__init__.py` và `tests/__init__.py` KHÔNG cần — pytest rootdir tự xử lý; nhưng để import `experiments.run_experiments` cần `experiments/__init__.py`:

```bash
touch experiments/__init__.py
```

- [x] **Step 3: Chạy test, xác nhận FAIL**

```bash
../.venv/bin/python -m pytest tests/test_run_experiments.py -v
```

Expected: FAIL — `ModuleNotFoundError` hoặc `ImportError: cannot import name 'plan_runs'`

- [x] **Step 4: Viết `experiments/run_experiments.py`**

```python
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
```

- [x] **Step 5: Chạy test + dry-run local, xác nhận PASS**

```bash
../.venv/bin/python -m pytest tests/test_run_experiments.py -v
../.venv/bin/python experiments/run_experiments.py --condition fixed20 \
  --prompts experiments/prompts_test.txt --styles_dir experiments/styles --dry_run
```

Expected: 2 passed; dry-run in `pending=30/30` (10 prompts × 3 styles hiện có local) và list các case, KHÔNG load model.

- [x] **Step 6: Commit**

```bash
git add experiments/ tests/test_run_experiments.py
git commit -m "feat: experiment runner with checkpoint/resume, prompts and styles scaffolding"
```

---

### Task 5: Eval metrics + analysis scripts

**Files:**
- Create: `experiments/eval_metrics.py`, `experiments/analyze_results.py`
- Test: `tests/test_eval_metrics.py`, `tests/test_analyze_results.py`

**Interfaces:**
- Consumes: layout output của Task 4 (`<out_root>/<condition>/pNN__style.jpg|.json`).
- Produces: `experiments/results/results.csv` (cột: condition, prompt_idx, style, clip_t, clip_i_style, stop_step, elapsed_sec, peak_vram_gb), `experiments/results/lpips_layout.csv` (condition, prompt_idx, lpips_mean), và các figure PNG + `summary_table.csv`, `winners.csv`.

- [x] **Step 1: Cài deps eval**

```bash
../.venv/bin/pip install lpips pandas matplotlib
```

- [x] **Step 2: Viết failing tests**

Tạo `tests/test_eval_metrics.py` — test hàm thu thập rows với feature extractor giả (không tải CLIP thật):

```python
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
```

Tạo `tests/test_analyze_results.py`:

```python
import pandas as pd

from experiments.analyze_results import summarize, winners


def sample_df():
    rows = []
    for cond, ct, ci in [("fixed5", 0.20, 0.60), ("fixed20", 0.30, 0.70),
                         ("adaptive", 0.29, 0.71)]:
        for pi in range(2):
            for style in ["a", "b"]:
                rows.append({"condition": cond, "prompt_idx": pi, "style": style,
                             "clip_t": ct + pi * 0.01, "clip_i_style": ci,
                             "stop_step": 7 if cond == "adaptive" else None,
                             "elapsed_sec": 100, "peak_vram_gb": 10})
    return pd.DataFrame(rows)


def test_summarize_means_per_condition():
    s = summarize(sample_df())
    assert set(s.index) == {"fixed5", "fixed20", "adaptive"}
    assert s.loc["fixed20", "clip_t"] > s.loc["fixed5", "clip_t"]


def test_winners_counts_fixed_only():
    w = winners(sample_df(), metric="clip_t", fixed_conditions=["fixed5", "fixed20"])
    # fixed20 thang ca 4 case tren clip_t
    assert w["fixed20"] == 4 and w.get("fixed5", 0) == 0
```

- [x] **Step 3: Chạy test, xác nhận FAIL**

```bash
../.venv/bin/python -m pytest tests/test_eval_metrics.py tests/test_analyze_results.py -v
```

Expected: FAIL — module không tồn tại.

- [x] **Step 4: Viết `experiments/eval_metrics.py`**

```python
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
            # condition lay tu noi dung JSON (khong phai ten thu muc), robust voi run co --tag
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
        # peek mot JSON dai dien de lay condition/tag (moi case trong cung thu muc
        # deu chung condition/tag do quy uoc dat ten out_dir cua run_experiments.py);
        # fallback ve ten thu muc neu khong co JSON nao
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
```

- [x] **Step 5: Viết `experiments/analyze_results.py`**

```python
"""Tong hop bang + figure tu results.csv / lpips_layout.csv / cac JSON adaptive.

  python experiments/analyze_results.py --results_dir experiments/results \
      --out_root experiments/outputs
"""
import argparse
import glob
import json
import os

FIXED_CONDITIONS = ["fixed5", "fixed10", "fixed20", "fixed30"]


def summarize(df):
    return df.groupby("condition")[["clip_t", "clip_i_style", "elapsed_sec",
                                    "peak_vram_gb"]].mean().round(4)


def winners(df, metric, fixed_conditions=FIXED_CONDITIONS):
    """Voi moi (prompt_idx, style): fixed condition nao co metric cao nhat. Tra ve dict dem."""
    sub = df[df["condition"].isin(fixed_conditions)]
    counts = {}
    for _, grp in sub.groupby(["prompt_idx", "style"]):
        best = grp.loc[grp[metric].idxmax(), "condition"]
        counts[best] = counts.get(best, 0) + 1
    return counts


def adaptive_gap(df, metric, fixed_conditions=FIXED_CONDITIONS):
    """Gap trung binh giua adaptive va per-case best fixed (>=0 la match/vuot)."""
    gaps = []
    for (pi, style), grp in df.groupby(["prompt_idx", "style"]):
        fixed_best = grp[grp["condition"].isin(fixed_conditions)][metric].max()
        ada = grp[grp["condition"] == "adaptive"][metric]
        if len(ada):
            gaps.append(float(ada.iloc[0]) - float(fixed_best))
    return sum(gaps) / len(gaps) if gaps else None


def plot_figures(df, out_root, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # histogram buoc dung adaptive
    stops = df[df["condition"] == "adaptive"]["stop_step"].dropna()
    if len(stops):
        plt.figure(figsize=(6, 4))
        plt.hist(stops, bins=range(1, 32))
        plt.xlabel("stop step")
        plt.ylabel("#cases")
        plt.title("Adaptive fusion stop step distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "hist_stop_steps.png"), dpi=150)
        plt.close()

    # duong hoi tu r(t) tu JSON adaptive
    plt.figure(figsize=(7, 4.5))
    for jpath in sorted(glob.glob(os.path.join(out_root, "adaptive", "*.json")))[:12]:
        with open(jpath) as f:
            fusion = json.load(f).get("fusion") or {}
        hist = fusion.get("r_history") or []
        if hist:
            steps, rs = zip(*hist)
            plt.plot(steps, rs, alpha=0.6)
    plt.axhline(0.2, color="red", linestyle="--", label="rho")
    plt.xlabel("denoise step")
    plt.ylabel("r(t)")
    plt.title("Teacher-student attention convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "r_curves.png"), dpi=150)
    plt.close()


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="experiments/results")
    ap.add_argument("--out_root", default="experiments/outputs")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.results_dir, "results.csv"))
    summary = summarize(df)
    lp = os.path.join(args.results_dir, "lpips_layout.csv")
    if os.path.exists(lp):
        ldf = pd.read_csv(lp)
        summary = summary.join(ldf.groupby("condition")["lpips_mean"].mean().round(4))
    summary.to_csv(os.path.join(args.results_dir, "summary_table.csv"))
    print(summary)

    win_rows = []
    for metric in ["clip_t", "clip_i_style"]:
        counts = winners(df, metric)
        gap = adaptive_gap(df, metric)
        print(f"\n[{metric}] per-case winners among fixed: {counts}")
        if gap is None:
            print(f"[{metric}] adaptive mean gap to per-case best fixed: N/A (no adaptive rows yet)")
        else:
            print(f"[{metric}] adaptive mean gap to per-case best fixed: {gap:+.4f}")
        for cond, n in counts.items():
            win_rows.append({"metric": metric, "condition": cond, "wins": n})
        # bo qua dong adaptive_gap khi chua co run "adaptive" nao (gap=None), thay vi ghi NaN
        if gap is not None:
            win_rows.append({"metric": metric, "condition": "adaptive_gap", "wins": gap})
    pd.DataFrame(win_rows).to_csv(os.path.join(args.results_dir, "winners.csv"), index=False)

    plot_figures(df, args.out_root, args.results_dir)
    print(f"\nfigures + tables saved to {args.results_dir}")


if __name__ == "__main__":
    main()
```

- [x] **Step 6: Chạy toàn bộ test, xác nhận PASS**

```bash
../.venv/bin/python -m pytest tests/ -v
```

Expected: 23 passed (17 cũ + 4 eval + 2 analyze).

- [x] **Step 7: Commit**

```bash
git add experiments/eval_metrics.py experiments/analyze_results.py \
        tests/test_eval_metrics.py tests/test_analyze_results.py
git commit -m "feat: evaluation metrics (CLIP-T/CLIP-I/LPIPS) and analysis scripts"
```

---

### Task 6: Push lên GitHub fork + smoke test trên Kaggle

**Files:**
- Không sửa code (trừ khi smoke test lộ bug — fix + commit).

**Interfaces:**
- Consumes: toàn bộ Task 1-5.
- Produces: branch `adaptive-fusion` trên fork GitHub của user; xác nhận cơ chế chạy đúng trên GPU thật.

- [ ] **Step 1: Fork + push (cần user có GitHub, dùng `gh` nếu có)**

```bash
gh repo fork Westlake-AGI-Lab/StyleStudio --clone=false
git remote add fork https://github.com/<GITHUB_USERNAME>/StyleStudio.git
git push -u fork adaptive-fusion
```

Expected: branch hiện trên fork. (`<GITHUB_USERNAME>` = tài khoản của user, vd. namtt36. Nếu không có `gh`, user tự fork trên web rồi chạy 2 lệnh sau.)

- [ ] **Step 2: Notebook Kaggle smoke test** — tạo notebook mới (GPU T4 x2, Internet On) với các cell:

```python
# Cell 1: clone branch
!git clone -b adaptive-fusion https://github.com/<GITHUB_USERNAME>/StyleStudio.git /kaggle/working/StyleStudio
%cd /kaggle/working/StyleStudio

# Cell 2: deps (khong dong den torch cua Kaggle)
!pip install -q diffusers==0.25.1 transformers==4.45.2 tokenizers==0.20.1 \
  huggingface-hub==0.24.6 accelerate peft safetensors einops omegaconf opencv-python lpips pandas

# Cell 3: checkpoints
from huggingface_hub import hf_hub_download, snapshot_download
import os
ie_root = snapshot_download('h94/IP-Adapter', allow_patterns=['sdxl_models/image_encoder/*'])
IMAGE_ENCODER = os.path.join(ie_root, 'sdxl_models', 'image_encoder')
CSGO = hf_hub_download('InstantX/CSGO', 'csgo_4_32.bin')

# Cell 4: chay unit tests tren GPU env
!python -m pytest tests/ -q

# Cell 5: smoke run adaptive (25 buoc cho nhanh)
!mkdir -p experiments/styles && cp assets/style1.jpg experiments/styles/
!printf 'A goat is playing on the beach\n' > /tmp/p1.txt
!python experiments/run_experiments.py --condition adaptive \
  --prompts /tmp/p1.txt --styles_dir experiments/styles \
  --num_steps 25 --image_encoder_path "$IMAGE_ENCODER" --csgo_ckpt "$CSGO" \
  --out_root experiments/smoke

# Cell 6: smoke run fixed20 (so sanh)
!python experiments/run_experiments.py --condition fixed20 \
  --prompts /tmp/p1.txt --styles_dir experiments/styles \
  --num_steps 25 --image_encoder_path "$IMAGE_ENCODER" --csgo_ckpt "$CSGO" \
  --out_root experiments/smoke

# Cell 7: kiem tra JSON
import json, glob
for j in glob.glob('experiments/smoke/*/*.json'):
    d = json.load(open(j))
    print(j, '| elapsed', d['elapsed_sec'], '| stop_step', (d.get('fusion') or {}).get('stop_step'))
    if d.get('fusion'):
        print('  r(t):', [round(r,3) for _, r in d['fusion']['r_history']])
```

Lưu ý: `$IMAGE_ENCODER`/`$CSGO` là biến Python — trong notebook thực tế dùng `{IMAGE_ENCODER}`/`{CSGO}` (cú pháp interpolation của IPython `!`).

- [ ] **Step 3: Tiêu chí smoke test PASS**

1. `pytest` xanh trên Kaggle.
2. Run adaptive hoàn thành, không OOM; JSON có `stop_step` trong khoảng `[5, 30]` và `r_history` giảm dần sau vài bước đầu (được phép tăng ở 2-3 bước đầu do running-max baseline).
3. Ảnh adaptive nhìn hợp lý (có style, layout không vỡ), so sánh bằng mắt với fixed20.
4. `elapsed_sec` của adaptive ≤ fixed20 + 10% (kỳ vọng nhanh hơn nếu dừng < 20).

Nếu fail → debug (dùng systematic-debugging skill), fix, commit, push lại, chạy lại smoke.

---

### Task 7: Dev split — chọn ρ và đóng băng

**Files:**
- Create: `experiments/CHOSEN_RHO.md` (ghi quyết định)

**Interfaces:**
- Consumes: runner (Task 4), eval (Task 5), fork đã smoke-test (Task 6).
- Produces: giá trị ρ chốt cho Task 8.

- [ ] **Step 1: Chuẩn bị style images đầy đủ** — trên Kaggle, copy đủ 6 style vào `experiments/styles/` (style1-3 từ assets, style4 từ dataset `tnamt9/assets`, style5-6 user bổ sung vào dataset đó). Dev dùng style1 + style4.

```bash
mkdir -p experiments/styles_dev
cp experiments/styles/style1.jpg experiments/styles/style4.jpg experiments/styles_dev/
```

- [ ] **Step 2: Chạy sweep ρ trên dev split (12 runs ≈ 1h)**

```bash
for RHO in 0.1 0.2 0.3; do
python experiments/run_experiments.py --condition adaptive --rho $RHO --tag rho$RHO \
  --prompts experiments/prompts_dev.txt --styles_dir experiments/styles_dev \
  --image_encoder_path "$IMAGE_ENCODER" --csgo_ckpt "$CSGO" \
  --out_root experiments/outputs_dev
done
```

- [ ] **Step 3: Eval dev + chọn ρ**

```bash
python experiments/eval_metrics.py --out_root experiments/outputs_dev \
  --styles_dir experiments/styles_dev --results_dir experiments/results_dev
```

Tiêu chí chọn: ρ có `clip_t` + `clip_i_style` tổng cao nhất; nếu xấp xỉ (chênh < 0.005), chọn ρ cho stop_step nhỏ hơn (nhanh hơn). Xem cả ảnh bằng mắt.

- [ ] **Step 4: Ghi quyết định + commit**

`experiments/CHOSEN_RHO.md`: ghi ρ được chọn, bảng số liệu dev, ngày chạy. Commit + push:

```bash
git add experiments/CHOSEN_RHO.md && git commit -m "docs: freeze rho from dev split" && git push fork adaptive-fusion
```

---

### Task 8: Main experiments (5 điều kiện × 60 case, ~18h GPU)

**Interfaces:**
- Consumes: ρ đã chốt (Task 7).
- Produces: `experiments/outputs/{fixed5,fixed10,fixed20,fixed30,adaptive}/` đầy đủ 60 case mỗi thư mục.

- [ ] **Step 1: Chạy theo phiên** — mỗi phiên Kaggle (~11h hiệu dụng) chạy tuần tự các condition; runner tự resume nên đứt phiên không mất dữ liệu:

```bash
for COND in adaptive fixed20 fixed10 fixed5 fixed30; do
python experiments/run_experiments.py --condition $COND --rho <RHO_CHOT> \
  --prompts experiments/prompts_test.txt --styles_dir experiments/styles \
  --image_encoder_path "$IMAGE_ENCODER" --csgo_ckpt "$CSGO"
done
```

(Thứ tự ưu tiên adaptive + fixed20 trước — nếu cháy quota vẫn có cặp so sánh chính.)

- [ ] **Step 2: Bảo toàn outputs giữa các phiên** — cuối MỖI phiên, lưu `experiments/outputs/` thành Kaggle Dataset (Save Version → output; hoặc `!zip -r outputs.zip experiments/outputs`), phiên sau tải lại vào đúng chỗ trước khi chạy tiếp (resume dựa trên file có sẵn).

- [ ] **Step 3: Kiểm tra đủ 300 case**

```bash
find experiments/outputs -name "*.jpg" | wc -l   # ky vong: 300
find experiments/outputs -name "*.json" | wc -l  # ky vong: 300
```

Nếu thiếu → chạy lại đúng condition đó (runner tự điền chỗ trống). Nếu quota gần cạn → kích hoạt phương án rút gọn trong spec (bỏ fixed30, giảm style).

---

### Task 9: Eval + figures + tổng hợp cho report

**Interfaces:**
- Consumes: outputs đầy đủ (Task 8).
- Produces: `experiments/results/` — results.csv, lpips_layout.csv, summary_table.csv, winners.csv, hist_stop_steps.png, r_curves.png.

- [ ] **Step 1: Chạy eval trên Kaggle (GPU giúp CLIP-L nhanh)**

```bash
python experiments/eval_metrics.py
python experiments/analyze_results.py
```

Expected: results.csv 300 rows; lpips_layout.csv 50 rows (5 cond × 10 prompts); các PNG/CSV trong `experiments/results/`.

- [ ] **Step 2: Kiểm tra kết quả khớp claim**

- `winners.csv`: phân bố winner giữa các fixed phải phân tán (không một fixed nào thắng > 70% case) — nếu tập trung 1 giá trị, ghi nhận trung thực trong report và phân tích.
- `adaptive_gap` ≥ ~-0.005 trên cả clip_t lẫn clip_i_style → claim "≥ best fixed" đứng vững.
- `hist_stop_steps.png`: phân bố bước dừng có variance (không dồn 1 giá trị) — bằng chứng "per-case".

- [ ] **Step 3: Tải results + outputs về máy, commit results nhỏ (CSV + PNG, không commit ảnh 300 case)**

```bash
git add experiments/results/ && git commit -m "results: main experiment tables and figures" && git push fork adaptive-fusion
```

- [ ] **Step 4: Grid ảnh định tính** — chọn 3 prompt × 3 style tiêu biểu, ghép grid so sánh 5 điều kiện (dùng matplotlib, code tùy biến lúc đó) cho report.

---

## Self-Review (đã chạy)

1. **Spec coverage:** cơ chế (§3) → Task 1-3; backward-compat → Task 2/3 + test legacy; thí nghiệm (§4.1-4.2) → Task 4, 7, 8; metrics (§4.3) → Task 5; figures (§4.4) → Task 5, 9; ngân sách/rủi ro (§4.5, §5) → Task 6 smoke + resume + thứ tự ưu tiên trong Task 8. Bug end_fusion passthrough (phát hiện khi lập plan) → Task 3 Step 4c.
2. **Placeholders:** không còn TBD; mọi bước code đều có code đầy đủ. `<GITHUB_USERNAME>`/`<RHO_CHOT>` là giá trị runtime do user/Task 7 quyết định — có hướng dẫn cụ thể tại chỗ.
3. **Type consistency:** `FusionController.register/is_active/report/reset/to_dict` nhất quán giữa Task 1 (định nghĩa), Task 2 (processor gọi), Task 3 (adapter tạo + reset + to_dict trong log), Task 4 (runner đọc `adapter.fusion_controller.to_dict()`); layout `pNN__style.jpg|.json` nhất quán giữa Task 4 (ghi) và Task 5 (đọc: `plan_runs`, `collect_rows`, `layout_lpips` parse `p[1:3]`).
