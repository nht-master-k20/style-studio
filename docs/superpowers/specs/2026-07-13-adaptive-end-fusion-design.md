# Adaptive Attention Fusion cho StyleStudio — Design

**Ngày:** 2026-07-13
**Bối cảnh:** Đồ án CS2309. Deliverable: report + thí nghiệm định lượng.
**Compute:** Kaggle T4 (quota ~40h trong 10 ngày, phiên tối đa 12h). Codebase: `StyleStudio/` (clone của [Westlake-AGI-Lab/StyleStudio](https://github.com/Westlake-AGI-Lab/StyleStudio), CVPR 2025).

## 1. Vấn đề

Teacher Model của StyleStudio ổn định layout bằng cách **copy nguyên** self-attention map của teacher (SDXL gốc) sang student (bản stylized) trong `end_fusion` bước denoise đầu (`ip_adapter/attention_processor.py:1264-1270`, class `AttnProcessor2_0_hijack`):

```python
if self.fuSAttn and self.denoise_step <= self.end_fusion:
    assert query.shape[0] == 4  # [teacher-uncond, student-uncond, teacher-cond, student-cond]
    attn_probs = softmax(QK^T / sqrt(d))
    attn_probs[1] = attn_probs[0]
    attn_probs[3] = attn_probs[2]
```

`end_fusion` là hyperparameter cố định (paper dùng 20/50 bước), nhưng giá trị tối ưu phụ thuộc từng cặp (prompt, style). Ngoài ra, trong các bước fusion code phải materialize full attention matrix (bỏ SDPA) — chậm và tốn VRAM (nguồn OOM trên T4).

## 2. Claim chính của report

> **Adaptive stopping đạt chất lượng ≥ end_fusion cố định tốt nhất mà không cần tune tay cho từng (prompt, style).**

Phụ (bonus): dừng sớm → ít bước attention materialized hơn → nhanh hơn, ít VRAM hơn.

Framing trung thực: phương pháp thay hyperparameter per-case (`end_fusion`) bằng **một hằng số toàn cục duy nhất** (ρ) dùng chung cho mọi sample, chọn một lần trên dev split.

## 3. Cơ chế: Adaptive Attention Fusion

### 3.1 Trực giác

Trong các bước fusion, attention map của student (trước khi bị copy) và teacher đều đã được tính. Khi layout đã "khóa", student tự hội tụ về teacher → độ lệch giữa hai map giảm → không cần ép nữa. Dừng fusion khi độ lệch đã giảm đủ sâu so với ban đầu.

### 3.2 Thành phần mới: `ip_adapter/fusion_controller.py`

```
FusionController(rho=0.2, end_fusion_max=30, min_steps=5)
├── register(layer_name)          # mỗi hijack processor đăng ký lúc setup
├── report(layer_name, step, d)   # processor báo độ lệch mỗi bước fusion
├── is_active(step) -> bool       # processor hỏi trước khi fuse
├── history                       # chuỗi r(t) + bước dừng, để log ra JSON
└── reset()                       # gọi đầu mỗi generate()
```

Logic quyết định (khi đủ báo cáo của mọi layer đã đăng ký ở bước `t`):

1. Mỗi layer đo trên **nhánh cond**, trước khi copy (tensor có sẵn, chi phí ~0):
   `d_layer(t) = mean |attn_probs[3] − attn_probs[2]|`
2. Chuẩn hóa theo layer (các layer có scale/resolution attention khác nhau — 4096 vs 1024 tokens). Baseline là **running max** thay vì `d_layer(1)`, vì các layer đứng trước cross-attn đầu tiên có `d_layer(1) = 0` (teacher/student cùng init latents, chưa phân kỳ):
   `r_layer(t) = d_layer(t) / max_{t' ≤ t} d_layer(t')` (nếu baseline ~0 → coi layer đã hội tụ, `r_layer = 0`)
3. Tổng hợp: `r(t) = mean_layers r_layer(t)`
4. **Dừng khi `r(t) ≤ ρ` và `t ≥ min_steps` (=5, tránh dừng non khi độ lệch còn đang tăng), hoặc `t ≥ end_fusion_max`.** Sau khi dừng, không bật lại trong lần generate đó; mọi layer quay về nhánh SDPA. Nếu một layer nào đó không báo cáo (đếm lệch), controller không ra quyết định → fusion chạy tới trần `end_fusion_max` — fallback an toàn.

   *(Cập nhật khi triển khai: dùng `≤` thay vì `<` — bộ unit test của chính thiết kế này đòi hỏi dừng khi `r(t)` bằng đúng `ρ` ở biên `ρ=1.0`; với `ρ=0.2` dùng trong thực nghiệm, hai điều kiện cho kết quả giống hệt nhau nên không ảnh hưởng claim của report.)*

Ghi chú: chỉ đo nhánh cond vì với guidance_scale=5, nhánh cond quyết định layout.

### 3.3 Thay đổi file hiện có (tối thiểu, giữ tương thích ngược)

| File | Thay đổi |
|---|---|
| `ip_adapter/attention_processor.py` | `AttnProcessor2_0_hijack.__init__` nhận `fusion_controller=None`. Điều kiện fusion: nếu có controller → `controller.is_active(self.denoise_step)`; nếu không → giữ nguyên `denoise_step <= end_fusion` (chế độ gốc không đổi hành vi). Trong nhánh fusion: tính `d_layer`, gọi `controller.report(...)` trước khi copy. |
| `ip_adapter/ip_adapter.py` | `StyleStudio_Adapter.__init__` nhận `adaptive_fusion=False, rho=0.2, end_fusion_max=30`; tạo controller và phát cho các hijack processor khi setup; `controller.reset()` đầu `generate()`. |
| `infer_StyleStudio.py` | Flags mới: `--adaptive_fusion`, `--rho`, `--end_fusion_max`, `--log_json <path>`. Sau khi generate: ghi JSON gồm chuỗi `r(t)`, bước dừng, thời gian chạy, args. |

Nguyên tắc: controller là optional — chạy không flag thì code hệt bản gốc (baseline sạch).

## 4. Thiết kế thí nghiệm

### 4.1 Test set
- **10 prompts** đa dạng (động vật / đồ vật / cảnh / người) × **6 style images** (4 từ `assets/` + 2 bổ sung khác chất liệu) = **60 cặp**.
- **Seed cố định 42** (như repo). 1 seed do quota — ghi nhận là limitation.
- **Dev split chọn ρ** (không trùng test): 2 prompts × 2 styles × ρ ∈ {0.1, 0.2, 0.3} = 12 runs. Chọn ρ trên dev rồi đóng băng.
- Cấu hình chung: `--fuSAttn --adainIP`, 50 bước, 1024×1024, guidance 5 — như mặc định repo.

### 4.2 Điều kiện so sánh (5 × 60 = 300 runs chính)

| Điều kiện | Vai trò |
|---|---|
| Fixed `end_fusion` ∈ {5, 10, 20, 30} | Baseline sweep — chứng minh giá trị tốt nhất thay đổi theo case |
| Adaptive (ρ từ dev) | Phương pháp đề xuất |
| `fuSAttn` off (tùy chọn nếu dư quota) | Tham chiếu không-teacher |

### 4.3 Metrics
1. **CLIP-T** (output ↔ prompt) — text alignment.
2. **CLIP-I style** (output ↔ style image) — style fidelity. Tái dùng `metrics.py` của repo nếu phù hợp (cần kiểm tra khi lập plan). Ghi chú limitation của CLIP-I.
3. **Layout stability** — LPIPS trung bình giữa các output cùng prompt, khác style, cùng seed (thấp = ổn định).
4. **Hiệu quả** (phụ) — số bước fusion trung bình, giây/ảnh, peak VRAM.

### 4.4 Figure/bảng trung tâm
1. Bảng chính: adaptive vs từng fixed trên 3 metrics.
2. **% case mà mỗi fixed value thắng** (phân tán → không fixed nào đúng cho mọi case) + gap của adaptive so với per-case best. Linh hồn của claim.
3. Curve hội tụ `r(t)` của sample tiêu biểu + histogram bước dừng adaptive (từ log JSON, không tốn run thêm).
4. Grid ảnh định tính side-by-side.

### 4.5 Ngân sách giờ (ước ~3.5 phút/run, batch-2, 50 bước, T4)

| Hạng mục | Runs | Giờ |
|---|---|---|
| Dev chọn ρ | 12 | ~1h |
| Main 5 điều kiện | 300 | ~18h |
| No-teacher (tùy chọn) | 60 | ~1.5h |
| Tính metrics | — | ~1h |
| **Tổng + buffer** | | **~25h / 40h** ✓ |

## 5. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| OOM trên T4 (batch-2 + materialized attention) | Patch `enable_attention_slicing()` (đã có trong notebook Kaggle); adaptive dừng sớm giảm thêm áp lực |
| Cháy quota | Phương án rút gọn định sẵn: 8 prompts × 5 styles, bỏ fixed=30 → ~11h |
| Adaptive không match best-fixed | Report vẫn đứng: phân tích nguyên nhân qua curve `r(t)`; fallback claim = hiệu quả tính toán |
| Đứt phiên Kaggle 12h | Script chạy theo batch có checkpoint: mỗi run ghi ảnh + JSON ngay, resume theo danh sách còn thiếu |

## 6. Ngoài phạm vi (ghi nhận, không làm)

- Per-layer adaptive stopping (Phương án B) — chỉ làm ablation nếu dư thời gian.
- Soft-decay blending (Phương án C) — mâu thuẫn với claim bỏ-hyperparameter.
- Drop teacher batch sau khi dừng fusion (tiết kiệm ~2× phần còn lại) — stretch goal, không cam kết.
- Nhiều seed, CSD metric, port sang DiT.

## 7. Cập nhật sau triển khai (Task 0-5, 2026-07-13/14)

Đã triển khai trên branch `adaptive-fusion` (repo `/Users/namtt/Documents/study/CS2309/StyleStudio`), qua `superpowers:subagent-driven-development` — mỗi task có implementer + reviewer riêng, cộng một whole-branch review cuối. 23/23 test pass. Commit range: `46f6902..ab2990d` (xem plan để chi tiết từng task).

**3 sai khác so với bản thiết kế gốc (đã duyệt trong lúc review, không phải lỗi implementer):**

1. **Điều kiện dừng `≤` thay vì `<`** (đã sửa ở §3.2 điểm 4 phía trên) — bộ test của chính spec này yêu cầu.
2. **`collect_rows`/`layout_lpips` (eval_metrics.py) lấy `condition` từ nội dung JSON** (`meta["condition"] + "_" + meta["tag"]`) thay vì tên thư mục output — robust hơn với các run có `--tag` (vd. dev ρ-sweep ở Task 7), không đổi giá trị thực tế so với cách cũ.
3. **`analyze_results.py`'s `main()` guard khi `adaptive_gap()` trả `None`** (chưa có run "adaptive" nào) — in `"N/A"` và bỏ qua dòng đó trong `winners.csv` thay vì crash `TypeError`.

**Việc cần xác nhận khi lên Kaggle (Task 6), không chặn tiến độ:**

- `experiments/run_experiments.py`'s `load_adapter()` gọi `pipe.enable_attention_slicing()` **trước** khi `StyleStudio_Adapter.set_ip_adapter()` thay toàn bộ attention processor bằng bản hijack — nhiều khả năng lệnh slicing bị ghi đè và không còn tác dụng giảm VRAM như §5 kỳ vọng. Nếu OOM trên T4, đây là nghi phạm đầu tiên; early-stop của adaptive fusion + `enable_vae_tiling()` vẫn giảm VRAM độc lập với việc này.
- `analyze_results.py`'s `plot_figures` vẽ đường `ρ` tham chiếu hardcode `0.2` trong hình `r_curves.png` — cần đọc từ JSON (hoặc tham số) sau khi Task 7 chốt giá trị ρ thật, nếu ρ chốt khác 0.2.

## 8. Sửa tín hiệu hội tụ + bug `end_fusion` (2026-07-14, verify trên Kaggle T4)

**Triệu chứng khi smoke test:** adaptive luôn TRÙNG full-fusion; log cho `stop_step=null`, `r(t)≈1.0` mọi bước → controller không bao giờ dừng.

**Hai nguyên nhân độc lập:**

1. **Tín hiệu §3.1 sai giả định.** `d=mean|attn_student − attn_teacher|` (nhánh cond) KHÔNG giảm mà **tăng đơn điệu**: cross-attention bơm style tích lũy nên self-attention của student ngày càng lệch teacher, không hội tụ. Chuẩn hóa running-max khi đó ghim `r=d/max≈1.0` → `r≤ρ` không bao giờ đạt. (Unit test cũ pass vì cấp `d` giảm nhân tạo.)
2. **Bug forward `end_fusion` (độc lập).** `infer_StyleStudio.py` không truyền `--end_fusion` vào `generate()`; `generate()` có default cứng `end_fusion=20` + guard `if end_fusion != self.end_fusion: set_endFusion(...)` ⇒ mọi `--end_fusion` bị ghi đè về 20. Chỉ ảnh hưởng đường CLI `infer`; `experiments/run_experiments.py` không dính (set qua constructor). Fix = thêm `end_fusion=args.end_fusion` trong lời gọi `generate()`.

**Đổi tín hiệu (Phương án được chọn):** dùng **độ ổn định thời gian của attention teacher** thay cho độ lệch student–teacher:

> `d_layer(t) = mean |A_teacher_cond(t) − A_teacher_cond(t−1)|`, head-averaged để tiết kiệm VRAM.

Đo trên index 2 (teacher, không bị copy đè); bước fusion đầu không có map trước → không report (history bắt đầu từ bước 2). Trực giác khớp §3.1 nhưng đúng thực nghiệm: "layout khóa ⇔ attention teacher ngừng đổi giữa các bước" → `d` giảm tự nhiên khi denoise settle → `r` hạ xuống dưới `ρ`. **Logic `FusionController` (running-max, mean-layer, dừng khi `r≤ρ` sau `min_steps`) GIỮ NGUYÊN** — chỉ đổi đại lượng được report.

File đổi: `ip_adapter/attention_processor.py` (tín hiệu mới + lưu map bước trước `_prev_teacher_attn`, reset ở `denoise_step==1`), `ip_adapter/fusion_controller.py` (docstring), `infer_StyleStudio.py` (forward `end_fusion`), `tests/test_attention_processor_adaptive.py` (report từ bước 2), notebook smoke test.

**Cần làm tiếp:** chốt lại `ρ` trên dev split với tín hiệu MỚI — giá trị 0.2 (từ thiết kế cũ) không còn cơ sở, phải sweep lại. Ngưỡng `end_fusion_max`/`min_steps` cũng nên xem lại theo phân bố `stop_step` thực tế.
