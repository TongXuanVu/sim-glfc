# Tài liệu kỹ thuật: sim-glfc với Tabular Dataset

> Tài liệu này giải thích chi tiết từng module trong repository `sim-glfc` sau khi đã được cải tiến để hỗ trợ bộ dữ liệu tabular (dạng vector đặc trưng 1D thay thế cho ảnh).

---

## 1. Tổng quan kiến trúc

```
sim-glfc/
├── federated_data/           # Dữ liệu train mỗi client - task
│   ├── client0_task1.pt      # client 0, task 1
│   ├── client0_task2.pt
│   ├── ...
│   └── client4_task6.pt      # 5 clients × 6 tasks = 30 file
├── 30_test_data.pt           # Dữ liệu test toàn cục
└── src/
    ├── fl_main.py            # ⭐ Điểm vào chính (main entry point)
    ├── GLFC.py               # ⭐ Model học liên tục tại client
    ├── ProxyServer.py        # ⭐ Proxy Server tại server
    ├── Fed_utils.py          # Các hàm tiện ích federated
    ├── myNetwork.py          # Kiến trúc mạng (MLP, ResNet, LeNet)
    ├── FederatedTabularDataset.py  # ⭐ Dataloader cho tabular data
    ├── option.py             # Tham số dòng lệnh
    ├── iCIFAR100.py          # Dataloader cho CIFAR100 (gốc)
    ├── mini_imagenet.py      # Dataloader cho Mini-ImageNet
    ├── tiny_imagenet.py      # Dataloader cho Tiny-ImageNet
    ├── proxy_data.py         # Dataset cho Proxy Server
    ├── ResNet.py             # Kiến trúc ResNet-18 + CBAM
    └── plot_results.py       # Script vẽ đồ thị kết quả
```

---

## 2. Bối cảnh bài toán: GLFC là gì?

**GLFC** = **Global-Local Forgetting Compensation** — một phương pháp học liên tục (**Continual Learning**) trong môi trường **Federated Learning**.

### Vấn đề cần giải quyết

| Vấn đề | Giải thích |
|---|---|
| **Catastrophic Forgetting** | Khi học task mới, model quên task cũ |
| **Non-IID Data** | Mỗi client có phân phối dữ liệu khác nhau |
| **Privacy** | Clients không chia sẻ dữ liệu trực tiếp |
| **Incremental Tasks** | Dữ liệu đến theo từng đợt (task) |

### Giải pháp GLFC
- Mỗi client giữ **exemplar set** — tập mẫu đại diện từ các task cũ
- Server dùng **Proxy Server** để giám sát và tái tạo dữ liệu từ **prototype gradients**
- Dùng **entropy signal** để phát hiện khi task mới xuất hiện

---

## 3. Luồng chạy tổng quát

```
fl_main.py (vòng lặp global rounds)
│
├─ [Mỗi global round]
│   ├─ 1. Chọn clients tham gia
│   ├─ 2. local_train() → mỗi client huấn luyện cục bộ
│   │       ├─ beforeTrain(): chuẩn bị dữ liệu task hiện tại
│   │       ├─ update_new_set(): cập nhật exemplar set
│   │       ├─ train(): huấn luyện model
│   │       └─ proto_grad_sharing(): chia sẻ prototype gradient
│   │
│   ├─ 3. participant_exemplar_storing(): clients còn lại cập nhật exemplar
│   ├─ 4. FedAvg(): tổng hợp model từ các clients
│   ├─ 5. proxy_server.dataloader(): server tái tạo dữ liệu từ gradient
│   └─ 6. model_global_eval(): đánh giá model toàn cục
```

---

## 4. Chi tiết từng module

---

### 4.1 `fl_main.py` — Điểm vào chính

**Chức năng**: Điều phối toàn bộ quá trình federated learning.

**Luồng thực thi**:
1. **Khởi tạo**: Parse tham số, thiết lập seed, tạo model toàn cục `model_g`
2. **Tạo clients**: Mỗi client được biểu diễn bằng một `GLFC_model` instance
3. **Vòng lặp global**: Lặp qua `epochs_global` round, mỗi round xử lý 1 task

**Cơ chế quản lý task**:
```python
task_id = ep_g // args.tasks_global
```
Ví dụ với `tasks_global=6`, `epochs_global=12`:
- Rounds 0–5 → `task_id = 0`
- Rounds 6–11 → `task_id = 1`

**Phân loại clients** (chỉ dùng khi không phải tabular):
- `old_client_0`: Client cũ không có dữ liệu task mới
- `old_client_1`: Client cũ có dữ liệu task mới
- `new_client`: Client mới (chỉ có dữ liệu task này)

**Với tabular dataset**: 5 clients cố định, tất cả cùng tham gia mỗi round.

---

### 4.2 `GLFC.py` — Model học liên tục tại client

**Chức năng**: Xử lý toàn bộ logic học liên tục tại mỗi client.

#### Các method quan trọng:

**`beforeTrain(task_id_new, group)`**
- Chuẩn bị dataloader cho task hiện tại
- Xác định `current_class`: danh sách các class thuộc task này
- Với tabular: tự động lấy danh sách class từ file `.pt`

**`update_new_set()`**
- Dùng **entropy signal** để phát hiện task mới
- Nếu phát hiện task mới → cập nhật exemplar set từ task cũ
- Sau đó kết hợp exemplar + data mới vào dataloader

**`entropy_signal(loader)`** — Cơ chế phát hiện task mới
```
Nếu entropy tăng đột ngột > 1.2 so với task cũ → Task mới xuất hiện!
```
Tại sao? Khi gặp dữ liệu lạ (task mới), model chưa học → đầu ra phân tán đều → entropy cao.

**`train(ep_g, model_old)`** — Huấn luyện local
- Loss gồm 2 thành phần:
  - `loss_cur`: Cross-entropy cho classes hiện tại
  - `loss_old`: Knowledge distillation từ model cũ (chống forgetting)

```
loss = 0.5 × loss_cur + 0.5 × loss_old
```

**`_construct_exemplar_set(images, m)`**
- Chọn `m` mẫu đại diện nhất cho mỗi class
- Chọn theo tiêu chí gần mean feature nhất trong không gian feature

**`prototype_mask()`**
- Tìm prototype (mẫu đại diện) cho mỗi class trong task hiện tại
- Tính gradient của model khi xử lý prototype này
- Gửi gradient lên server (thay vì gửi dữ liệu thật → **bảo vệ privacy**)

---

### 4.3 `ProxyServer.py` — Máy chủ trung gian

**Chức năng**: Server không chỉ nhận model weights mà còn **tái tạo dữ liệu giả** từ prototype gradients để tự giám sát model.

#### Luồng hoạt động:

```
Client gửi prototype gradients
        ↓
Server giải gradient ngược (gradient inversion)  
        ↓
Server có được dữ liệu giả (synthetic data)
        ↓
Server dùng dữ liệu giả để monitor model toàn cục
        ↓
Server chọn model tốt nhất trong 2 phiên liên tục
```

**`reconstruction()`** — Tái tạo dữ liệu từ gradient
- Dùng **L-BFGS optimizer** để tối ưu `dummy_data`
- Mục tiêu: tìm `dummy_data` sao cho gradient của nó ≈ gradient nhận được từ client
- Với tabular: `dummy_data` là vector 1D, kích thước `(1, 32)`
- Với ảnh: `dummy_data` là ảnh, kích thước `(1, 3, 32, 32)`

**`monitor()`** — Kiểm tra chất lượng model
- Dùng dữ liệu tái tạo để đánh giá model
- Server giữ lại 2 model tốt nhất: `best_model_1`, `best_model_2`

**`model_back()`**
- Trả về 2 model tốt nhất cho clients dùng làm reference (chống forgetting)

---

### 4.4 `Fed_utils.py` — Tiện ích Federated Learning

**Các hàm chính:**

**`setup_seed(seed)`**
- Thiết lập random seed để đảm bảo reproducibility

**`model_to_device(model, parallel, device)`**
- Di chuyển model lên GPU/CPU
- `device=-1` → CPU; `device=0,1,...` → CUDA GPU

**`local_train(clients, index, model_g, task_id, ...)`**
- Điều phối quá trình huấn luyện cục bộ của một client
- Client copy model từ server, train, rồi trả về weights và gradients

**`participant_exemplar_storing(clients, num, ...)`**
- Cập nhật exemplar set cho **tất cả** clients (kể cả những client không được chọn train round này)
- Quan trọng để mọi client đều có exemplar mới nhất

**`FedAvg(models)`** — Federated Averaging
```
w_global = (w_client_1 + w_client_2 + ... + w_client_k) / k
```
Tổng hợp weights từ nhiều clients bằng cách lấy trung bình.

**`model_global_eval(model_g, test_dataset, task_id, ...)`**
- Đánh giá model toàn cục trên test dataset
- Lấy tất cả classes từ task 0 đến task hiện tại để test

---

### 4.5 `myNetwork.py` — Kiến trúc mạng nơ-ron

#### Các class:

**`network(numclass, feature_extractor)`** — Model chính
- Kết hợp feature extractor + lớp phân loại `fc`
- `Incremental_learning(numclass)`: mở rộng lớp `fc` khi có class mới, giữ nguyên weights cũ

**`LeNet`** — Encoder cho ảnh (gốc)
- Input: ảnh `(3, 32, 32)` → Hidden → Output: `num_classes` logits

**`MLP_FeatureExtractor`** *(mới, thêm cho tabular)*
```
Input(32) → Linear(32) → ReLU → Linear(32) → ReLU → Output(32)
```

**`MLP_Encoder`** *(mới, thêm cho tabular)*
```
Input(32) → Linear(32) → ReLU → Linear(32) → ReLU → Output(num_classes)
```

**`weights_init(m)`**
- Khởi tạo weights ngẫu nhiên từ `[-0.5, 0.5]`

---

### 4.6 `FederatedTabularDataset.py` — Dataloader tabular *(mới)*

**Chức năng**: Load và quản lý dữ liệu 1D tabular theo từng client và task.

**Cấu trúc dữ liệu input**:
```
federated_data/client{i}_task{j}.pt
→ tuple(data_tensor, label_tensor)
→ data_tensor.shape = (N, 32)    # N mẫu, mỗi mẫu 32 features
→ label_tensor.shape = (N,)       # N nhãn lớp
```

**Các method chính**:

| Method | Chức năng |
|---|---|
| `set_task(task_id)` | Thông báo task hiện tại (0-indexed → 1-indexed file) |
| `load_task(task_id)` | Load file `.pt` theo client và task ID |
| `getTrainData(classes, exemplar_set, ...)` | Load data huấn luyện + exemplar cũ |
| `getTestData(classes)` | Load data test từ `30_test_data.pt` |
| `get_image_class(label)` | Lấy tất cả mẫu của 1 class cụ thể |

---

### 4.7 `option.py` — Tham số dòng lệnh

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--dataset` | `cifar100` | Loại dataset: `cifar100`, `tiny_imagenet`, `tabular` |
| `--numclass` | `10` | Số class trong task đầu tiên |
| `--task_size` | `10` | Số class mỗi task tăng thêm |
| `--num_clients` | `30` | Tổng số clients ban đầu |
| `--local_clients` | `10` | Số clients được chọn mỗi round |
| `--epochs_global` | `100` | Tổng số global rounds |
| `--tasks_global` | `10` | Số rounds mỗi task |
| `--epochs_local` | `20` | Số epoch huấn luyện local mỗi round |
| `--learning_rate` | `2.0` | Learning rate |
| `--memory_size` | `2000` | Kích thước exemplar memory |
| `--device` | `0` | ID GPU (`-1` = CPU) |
| `--seed` | `2021` | Random seed |

---

### 4.8 `iCIFAR100.py` / `ResNet.py` — Hỗ trợ ảnh (gốc)

- **`iCIFAR100`**: Kế thừa từ CIFAR100 PyTorch, thêm các method `getTrainData()`, `getTestData()`, `get_image_class()`.
- **`resnet18_cbam()`**: ResNet-18 backbone + **CBAM** (Convolutional Block Attention Module) — cơ chế attention giúp model tập trung vào vùng quan trọng trong ảnh.

---

## 5. Cơ chế học liên tục — Chi tiết

```
Task 0: Client học class {0,...,5}       → cập nhật exemplar cho class 0-5
Task 1: Client học class {6,...,11}      
        + replay exemplar class 0-5      → không quên task cũ
Task 2: Client học class {12,...,17}     
        + replay exemplar class 0-11     → không quên task cũ và cũ hơn
...
```

**Exemplar selection** (chọn mẫu đại diện):
- Chọn `m = memory_size / số_class_đã_học` mẫu mỗi class
- Mẫu được chọn theo **herding strategy**: ưu tiên mẫu có feature gần **mean** của class nhất

---

## 6. Cơ chế bảo vệ Privacy

GLFC **KHÔNG** gửi dữ liệu thô lên server. Thay vào đó:

```
Client                    Server
  |                          |
  |-- prototype gradient --> |  (gradients của 1 mẫu đại diện)
  |                          | 
  |                          | [Gradient Inversion Attack]
  |                          | → tái tạo dữ liệu giả
  |                          | → dùng để monitor model
```

Dữ liệu tái tạo chỉ dùng để giám sát, không huấn luyện lại model.

---

## 7. Cách chạy

### Chạy với tabular dataset (dữ liệu mới)

```powershell
cd c:\Users\Admin\Desktop\glfc\sim-glfc\src

# Chạy trên CPU
python fl_main.py --dataset tabular --device -1

# Chạy thêm epochs
python fl_main.py --dataset tabular --device -1 --epochs_global 60

# Giảm epoch local để chạy nhanh hơn
python fl_main.py --dataset tabular --device -1 --epochs_local 5 --epochs_global 12
```

### Xem kết quả

```powershell
# Log nằm tại:
type .\training_log\glfc\seed2021\log_tar_6.txt

# Vẽ đồ thị:
python plot_results.py
```

### Chạy với CIFAR100 (cấu hình gốc)

```powershell
python fl_main.py --dataset cifar100 --numclass 10 --task_size 10 --epochs_global 100 --tasks_global 10
```

---

## 8. Kết quả thực nghiệm (12 rounds)

| Round | Task | Accuracy |
|-------|------|----------|
| 0     | 0    | 2.35%    |
| 1     | 0    | 2.35%    |
| 2     | 0    | 2.35%    |
| 3     | 0    | 2.35%    |
| 4     | 0    | **15.18%** |
| 5     | 0    | **17.50%** |
| 6     | 1    | 17.25%   |
| 7     | 1    | 14.38%   |
| 8     | 1    | 14.82%   |
| 9     | 1    | 15.89%   |
| 10    | 1    | 15.41%   |
| 11    | 1    | **15.77%** |

> **Nhận xét**: Accuracy tăng từ ~2% lên ~16% sau khi model bắt đầu hội tụ (từ round 4). Có sụt giảm nhỏ khi chuyển sang task 1 do catastrophic forgetting, sau đó ổn định lại.

---

*Tài liệu cập nhật lần cuối: 2026-04-01*
