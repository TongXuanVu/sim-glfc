# Phân tích kết quả thấp — `log_tar_6.txt`

> File log: `src/training_log/glfc/seed2021/log_tar_6.txt`  
> Cấu hình: `method=glfc`, `task_size=6`, `learning_rate=2.0`, `dataset=tabular`

---

## 📊 Số liệu quan sát

| Round | Task | Accuracy | Nhận xét |
|-------|------|----------|----------|
| 0     | 0    | 2.35%    | Gần random |
| 1     | 0    | 2.35%    | Gần random |
| 2     | 0    | 2.35%    | Gần random |
| 3     | 0    | 2.35%    | Gần random |
| 4     | 0    | **15.18%** | Bắt đầu hội tụ |
| 5     | 0    | **17.50%** | Đỉnh task 0 |
| 6     | 1    | 17.25%   | Forgetting nhỏ |
| 7     | 1    | 14.38%   | Sụt giảm |
| 8     | 1    | 14.82%   | Hồi phục |
| 9     | 1    | 15.89%   | Tiếp tục tăng |
| 10    | 1    | 15.41%   | Dao động |
| 11    | 1    | 15.77%   | Ổn định |

**Random baseline với 34 classes = 1/34 ≈ 2.94%** — round 0–3 gần bằng random!

---

## 🔴 Nguyên nhân 1 (Cao nhất): Eval trên toàn bộ 34 classes từ round đầu

**Vị trí bug:** `fl_main.py`, line 147

```python
# Code hiện tại (SAI logic với tabular):
acc_global = model_global_eval(model_g, test_dataset, task_id, args.numclass, args.device)
#                                                               ↑↑↑ args.numclass = 34 (cố định!)

# Bên trong model_global_eval:
test_dataset.getTestData([0, task_size * (task_id + 1)])
# → [0, 34 * (0+1)] = [0, 34] → test ngay trên TẤT CẢ 34 class!
```

**Hậu quả:** Task 0 model chỉ học được ~6 classes (của task 0), nhưng bị đánh giá trên 34 classes → 28 classes còn lại predict sai hoàn toàn → accuracy ≈ random = 2.35%.

**Sửa:**
```python
# Nên dùng task_size thay vì numclass:
acc_global = model_global_eval(model_g, test_dataset, task_id, args.task_size, args.device)
# → getTestData([0, 6*(0+1)]) = [0, 6] → chỉ test class đã học
```

---

## 🔴 Nguyên nhân 2 (Cao): LR schedule được thiết kế cho CIFAR100, không phù hợp với 12 rounds

**Vị trí:** `GLFC.py`, lines 146–163

```python
for epoch in range(self.epochs):   # epochs_local = 20
    if (epoch + ep_g * 20) % 200 == 100:   # Giảm LR tại bước 100
        lr = learning_rate / 5              # step 100 = round 4, epoch 20 → round 5!
    elif (epoch + ep_g * 20) % 200 == 150:
        lr = learning_rate / 25
    elif (epoch + ep_g * 20) % 200 == 180:
        lr = learning_rate / 125
```

**Vấn đề:** Schedule được thiết kế cho 200 steps (CIFAR gốc). Với 12 rounds × 20 epochs = 240 steps:
- LR không giảm cho đến **round 4–5** (step ~100)
- Vì vậy round 0–3 train với LR = 2.0 (quá lớn) → model không hội tụ

**Đây là lý do accuracy bỗng tăng từ 2.35% lên 15.18% tại round 4**: LR schedule bắt đầu giảm.

**Sửa — điều chỉnh cho 12 rounds:**
```python
# Giảm LR sớm hơn, phù hợp với tổng 240 steps
if (epoch + ep_g * 20) % 80 == 40:    # step 40, 120, 200...
    lr = learning_rate / 5
elif (epoch + ep_g * 20) % 80 == 60:
    lr = learning_rate / 25
```

---

## 🔴 Nguyên nhân 3 (Cao): MLP quá nhỏ — capacity bottleneck

**Vị trí:** `myNetwork.py`, lines 67–79

```python
class MLP_FeatureExtractor(nn.Module):
    def __init__(self, in_dim=32, hidden=32):
        self.body = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(),   # ← không mở rộng chiều
            nn.Linear(32, 32), nn.ReLU()    # ← không mở rộng chiều
        )
        self.fc = nn.Linear(32, 1)   # dummy: chỉ để lấy in_features=32
```

**Vấn đề:** Input 32 → hidden 32 → output 32. Không có sự mở rộng feature space. Với 34 classes cần phân biệt từ 32 features, model thiếu capacity. So sánh: ResNet-18 gốc xuất 512 chiều cho 100 classes.

**Sửa — tăng capacity:**
```python
class MLP_FeatureExtractor(nn.Module):
    def __init__(self, in_dim=32, hidden=128):
        self.body = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.fc = nn.Linear(64, 1)  # dummy → in_features = 64
```

---

## 🟡 Nguyên nhân 4 (Trung bình): `signal=True` tại task 0 nhưng `last_class=None` → exemplar bị skip

**Vị trí:** `GLFC.py` + `beforeTrain()`

```python
# beforeTrain() với tabular:
self.last_class = self.current_class if group != 0 else None
# group=1 (non-old_client_0) → last_class = current_class của task 0

# update_new_set():
if self.signal and (self.last_class != None):
    # ← Nếu signal=True nhưng last_class = None → KHÔNG tạo exemplar
    self._construct_exemplar_set(...)
```

**Vấn đề:** Ở task 0 round 0, entropy tăng từ 0 → ~3.5 > 1.2 → `signal=True`. Nhưng `last_class=None` (chưa có task trước) → block exemplar bị skip. Lần tiếp theo cùng task: entropy không tăng nữa → `signal=False` → exemplar vẫn không được tạo trong cả task 0.

**Hậu quả:** Chuyển sang task 1 không có exemplar để replay → catastrophic forgetting.

---

## 🟡 Nguyên nhân 5 (Trung bình): `entropy_signal` threshold 1.2 chưa được tune cho tabular

**Vị trí:** `GLFC.py`, line 195

```python
if overall_avg - self.last_entropy > 1.2:   # threshold từ bài báo gốc (cho ảnh)
    res = True
```

**Vấn đề:** Threshold 1.2 được calibrate cho không gian entropy của ảnh với ResNet-18. Với tabular data và MLP nhỏ, phân phối entropy có thể khác → ngưỡng này có thể quá cao hoặc quá thấp.

---

## 🟡 Nguyên nhân 6 (Trung bình): Learning rate 2.0 quá lớn cho tabular data

**Vị trí:** `option.py`, default `learning_rate=2.0`

```python
opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
# learning_rate = 2.0 → quá lớn cho dữ liệu tabular với MLP
```

**Vấn đề:** LR = 2.0 là hyperparameter được tune cho ResNet + CIFAR. MLP trên tabular data thường cần LR nhỏ hơn nhiều (0.01–0.1).

---

## 📋 Tổng hợp và mức độ ưu tiên

| # | Nguyên nhân | File | Mức ảnh hưởng | Khó sửa |
|---|---|---|---|---|
| 1 | Eval trên 34 classes thay vì chỉ classes đã học | `fl_main.py:147` | 🔴 Rất cao | Dễ |
| 2 | LR schedule thiết kế cho 200 steps, không phù hợp 12 rounds | `GLFC.py:146` | 🔴 Cao | Dễ |
| 3 | MLP quá nhỏ (32→32→32) cho 34 classes | `myNetwork.py:67` | 🔴 Cao | Dễ |
| 4 | Task 0 không xây dựng được exemplar | `GLFC.py:96` | 🟡 Trung bình | Trung bình |
| 5 | Entropy threshold 1.2 chưa tune cho tabular | `GLFC.py:195` | 🟡 Trung bình | Trung bình |
| 6 | LR 2.0 quá lớn cho tabular/MLP | `option.py:18` | 🟡 Trung bình | Dễ |

---

## 🛠️ Kế hoạch sửa để cải thiện accuracy

### Bước 1 — Sửa nhanh (không đổi architecture)
```bash
# Chạy với learning rate nhỏ hơn
python fl_main.py --dataset tabular --device -1 --learning_rate 0.1 --epochs_global 60

# Hoặc tăng rounds để LR schedule có cơ hội kích hoạt
python fl_main.py --dataset tabular --device -1 --epochs_global 60 --epochs_local 20
```

### Bước 2 — Sửa logic eval (1 dòng code)

```python
# fl_main.py, dòng 147: đổi args.numclass → args.task_size
acc_global = model_global_eval(model_g, test_dataset, task_id, args.task_size, args.device)
```

### Bước 3 — Tăng capacity MLP

```python
# myNetwork.py: tăng hidden dimension
class MLP_FeatureExtractor(nn.Module):
    def __init__(self, in_dim=32, hidden=128):  # ← 32 → 128
        self.body = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64),  nn.ReLU()
        )
        self.fc = nn.Linear(64, 1)
```

---

## 💡 Dự đoán sau khi sửa

| Cải tiến | Accuracy kỳ vọng |
|---|---|
| Baseline (hiện tại) | ~15–17% |
| Sau sửa eval logic | ~40–50% (task 0 riêng lẻ) |
| Sau tăng MLP capacity | ~55–65% (task 0) |
| Sau tune LR + schedule | ~60–70% (task 0) |
| Tất cả cải tiến kết hợp + nhiều rounds | ~50–60% (trung bình mọi task) |

---

*Phân tích lần cuối: 2026-04-01*
