# Giải thích từng Module trong repo `sim-glfc`
> Đối chiếu trực tiếp với bài báo **B.1.GLFC.pdf**

---

## 📌 1. Bài báo nói gì? — Tổng quan GLFC

**GLFC** = **Global-Local Forgetting Compensation**

Bài báo giải quyết bài toán **Federated Class-Incremental Learning (FCIL)** — tức là học liên tục (Continual Learning) trong môi trường Federated Learning, nơi:

| Thách thức | Diễn giải |
|---|---|
| **Catastrophic Forgetting** (local) | Khi client học task mới, model quên các class cũ |
| **Catastrophic Forgetting** (global) | Khi server tổng hợp model từ nhiều clients, model toàn cục cũng bị quên |
| **Non-IID data** | Mỗi client chỉ có một tập dữ liệu thiên lệch (không phân phối đều) |
| **Privacy** | Client không được phép chia sẻ dữ liệu thô lên server |

### Giải pháp GLFC (2 thành phần chính):

```
┌─────────────────────────────────────────────────────┐
│  LOCAL:   Global-Local Forgetting Compensation       │
│           → Client dùng knowledge distillation      │
│           → Client giữ exemplar set (replay)         │
├─────────────────────────────────────────────────────┤
│  GLOBAL:  Proxy Server + Gradient Inversion          │
│           → Server tái tạo dữ liệu từ gradient      │
│           → Server monitor và chọn model tốt nhất   │
└─────────────────────────────────────────────────────┘
```

---

## 📌 2. Kiến trúc tổng thể

```
sim-glfc/
├── federated_data/                  # Dữ liệu phân tán (5 clients × 6 tasks)
│   ├── client0_task1.pt             # Dữ liệu client 0, task 1
│   └── ...
├── 30_test_data.pt                  # Test set toàn cục
└── src/
    ├── fl_main.py          ← Điều phối toàn bộ vòng lặp FL
    ├── GLFC.py             ← Logic học liên tục tại client
    ├── ProxyServer.py      ← Proxy server + gradient inversion
    ├── Fed_utils.py        ← FedAvg, local_train, eval
    ├── myNetwork.py        ← Kiến trúc mạng (MLP, LeNet, ResNet)
    ├── FederatedTabularDataset.py  ← Dataloader tabular
    ├── proxy_data.py       ← Dataset cho Proxy Server
    ├── option.py           ← CLI arguments
    └── ResNet.py           ← ResNet-18 + CBAM (backbone gốc)
```

---

## 📌 3. Luồng chạy tổng quát

```
fl_main.py
│
└── for ep_g in range(epochs_global):          # Mỗi global round
    │
    ├── task_id = ep_g // tasks_global          # Xác định task hiện tại
    ├── model_old = proxy_server.model_back()   # Lấy 2 best model từ server
    │
    ├── for c in clients_index:                 # Mỗi client được chọn
    │   └── local_train(models[c], ...)
    │       ├── beforeTrain()      # Chuẩn bị data task hiện tại
    │       ├── update_new_set()   # Phát hiện task mới + cập nhật exemplar
    │       ├── train()            # Huấn luyện (loss_cur + loss_old)
    │       └── proto_grad_sharing()  # Gửi gradient prototype lên server
    │
    ├── participant_exemplar_storing()  # Cập nhật exemplar cho mọi client
    ├── FedAvg(w_local)                # Tổng hợp weights
    ├── proxy_server.dataloader(pool_grad)
    │   ├── reconstruction()            # Gradient inversion → synthetic data
    │   └── monitor()                   # Đánh giá model với synthetic data
    │
    └── model_global_eval()             # Đánh giá accuracy toàn cục
```

---

## 📌 4. Chi tiết từng Module

---

### 🔵 Module 1: `fl_main.py` — Điểm vào chính

**Tương ứng bài báo:** Algorithm 1 (GLFC training loop)

**Chức năng:** Điều phối toàn bộ quá trình Federated Learning qua các global rounds.

#### Khởi tạo (lines 18–82)

```python
# Tabular mode: override các tham số mặc định
if args.dataset == 'tabular':
    args.num_clients = 5        # 5 clients cố định
    args.tasks_global = 6       # 6 tasks
    args.numclass = 34          # 34 classes tổng cộng
    args.task_size = 6          # mỗi task 6 classes mới
    feature_extractor = MLP_FeatureExtractor(in_dim=32, hidden=32)
else:
    feature_extractor = resnet18_cbam()   # ResNet-18 cho ảnh
```

#### Vòng lặp global (lines 95–155)

```python
for ep_g in range(args.epochs_global):
    task_id = ep_g // args.tasks_global   # task 0: round 0–5, task 1: round 6–11...
    
    # Lấy 2 best model từ Proxy Server làm reference cho knowledge distillation
    model_old = proxy_server.model_back()  # → [best_model_1, best_model_2]
    
    # Chọn clients tham gia
    if args.dataset == 'tabular':
        clients_index = list(range(num_clients))  # Tabular: tất cả 5 clients
    else:
        clients_index = random.sample(range(num_clients), args.local_clients)  # Image: random subset
```

> **Điểm khác biệt tabular vs image:** Với ảnh, số clients tăng dần theo task (thêm `new_client` mỗi task). Với tabular, 5 clients cố định tham gia mọi round.

---

### 🔵 Module 2: `GLFC.py` — Logic học liên tục tại client

**Tương ứng bài báo:** Section 3 (Local Forgetting Compensation) + Section 4 (Global)

**Class chính:** `GLFC_model`

#### 2.1 `beforeTrain(task_id_new, group)` — Chuẩn bị data

```python
def beforeTrain(self, task_id_new, group):
    if task_id_new != self.task_id_old:          # Task mới lần đầu
        if type(self.train_dataset).__name__ == 'FederatedTabularDataset':
            data, targets = self.train_dataset.load_task(task_id_new + 1)
            self.current_class = torch.unique(targets).tolist()  # Lấy danh sách class
        else:
            # Image: chọn ngẫu nhiên 6 class từ task range
            self.current_class = random.sample([x for x in range(self.numclass - self.task_size, self.numclass)], 6)
    
    self.train_loader = self._get_train_and_test_dataloader(self.current_class, False)
```

**Ý nghĩa:** Mỗi client nhìn thấy một tập class riêng của mình trong task hiện tại (non-IID).

#### 2.2 `update_new_set()` — Phát hiện task mới + cập nhật exemplar

```python
def update_new_set(self):
    self.signal = self.entropy_signal(self.train_loader)   # Phát hiện task mới?
    
    if self.signal and (self.last_class != None):
        # Task mới ĐƯỢC phát hiện → cập nhật exemplar set từ task CŨ
        self.learned_numclass += len(self.last_class)
        m = int(self.memory_size / self.learned_numclass)   # m mẫu/class
        self._reduce_exemplar_sets(m)                        # Cắt bớt exemplar cũ
        for i in self.last_class:
            images = self.train_dataset.get_image_class(i)
            self._construct_exemplar_set(images, m)          # Xây dựng exemplar mới
    
    # Mix exemplar cũ vào dataloader → replay để chống forgetting
    self.train_loader = self._get_train_and_test_dataloader(self.current_class, True)
```

#### 2.3 `entropy_signal(loader)` — Phát hiện task mới bằng entropy

**Tương ứng bài báo:** Section 3.2 (Task Detection)

```python
def entropy_signal(self, loader):
    # Tính entropy trung bình của toàn bộ data hiện tại
    # entropy(p) = -Σ p_i * log(p_i)
    overall_avg = torch.mean(all_ent).item()
    
    if overall_avg - self.last_entropy > 1.2:
        res = True   # Task mới! Entropy tăng đột ngột
    
    self.last_entropy = overall_avg
    return res
```

**Tại sao entropy báo hiệu task mới?**  
Khi gặp data của task mới (class chưa học), model không nhận ra → softmax phân tán đều → entropy cao. Ngưỡng `1.2` là hyperparameter từ bài báo.

#### 2.4 `_compute_loss()` — Hàm loss chính

**Tương ứng bài báo:** Equation (1) và (2) trong paper

```python
def _compute_loss(self, indexs, imgs, label):
    output = self.model(imgs)
    target = get_one_hot(label, self.numclass, self.device)
    
    # Weighted Binary Cross-Entropy cho classes hiện tại
    w = self.efficient_old_class_weight(output, label)
    loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(output, target))
    
    if self.old_model is not None:
        # Knowledge Distillation: distill từ old model cho các class cũ
        distill_target = target.clone()
        old_target = torch.sigmoid(self.old_model(imgs))     # Soft labels từ old model
        distill_target[..., :old_task_size] = old_target     # Ghi đè phần class cũ
        loss_old = F.binary_cross_entropy_with_logits(output, distill_target)
        
        return 0.5 * loss_cur + 0.5 * loss_old    # loss = 50% current + 50% distillation
    
    return loss_cur
```

**2 thành phần loss:**
- `loss_cur`: Học class mới (Cross-Entropy)
- `loss_old`: Không quên class cũ (Knowledge Distillation từ `old_model`)

#### 2.5 `efficient_old_class_weight()` — Trọng số thích ứng

**Tương ứng bài báo:** Adaptive weighting scheme

```python
def efficient_old_class_weight(self, output, label):
    # g = |sigmoid(output) - one_hot(label)|  ← độ khó của mẫu
    g = torch.abs(pred.detach() - target)
    
    # w1: trọng số cho các mẫu thuộc class đã học cũ
    # w2: trọng số cho các mẫu thuộc class mới
    # → Cân bằng giữa class cũ và mới để tránh bias
```

**Ý nghĩa:** Những mẫu mà model đang predict sai nhiều sẽ được gán trọng số cao hơn → tập trung học các mẫu khó.

#### 2.6 `_construct_exemplar_set(images, m)` — Herding strategy

**Tương ứng bài báo:** Section 3.3 (Exemplar Management)

```python
def _construct_exemplar_set(self, images, m):
    class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
    
    for i in range(m):
        # Chọn mẫu sao cho mean của exemplar đã chọn ≈ mean toàn class
        x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
        x = np.linalg.norm(x, axis=1)
        index = np.argmin(x)    # Mẫu có feature gần mean nhất
        now_class_mean += feature_extractor_output[index]
        exemplar.append(images[index])
```

**Herding:** Chọn `m` mẫu đại diện nhất (gần trung tâm class nhất trong feature space).

#### 2.7 `prototype_mask()` — Gửi gradient lên server (Privacy-preserving)

**Tương ứng bài báo:** Section 4 (Global Forgetting Compensation)

```python
def prototype_mask(self):
    for i in self.current_class:
        # 1. Tìm prototype: mẫu gần mean của class nhất
        pro_index = np.argmin(dis)
        proto.append(images[pro_index])
    
    for i in range(len(proto)):
        # 2. Tinh chỉnh prototype qua 50 bước gradient descent
        for ep in range(iters):   # iters = 50
            outputs = proto_model(data)
            loss_cls = F.binary_cross_entropy_with_logits(outputs, target)
            opt.zero_grad(); loss_cls.backward(); opt.step()
        
        # 3. Tính gradient của encode_model trên prototype tinh chỉnh
        outputs = self.encode_model(data)
        loss_cls = criterion(outputs, label)
        dy_dx = torch.autograd.grad(loss_cls, self.encode_model.parameters())
        proto_grad.append(list(dy_dx))   # Gửi gradient (KHÔNG gửi dữ liệu thật!)
```

**Cơ chế privacy:** Client chỉ gửi **gradient** (không phải dữ liệu thô). Server sau đó tái tạo dữ liệu giả từ gradient này.

---

### 🔵 Module 3: `ProxyServer.py` — Máy chủ trung gian

**Tương ứng bài báo:** Section 4 (Global Forgetting Compensation via Proxy Server)

**Class chính:** `proxyServer`

#### 3.1 `dataloader(pool_grad)` — Luồng tổng

```python
def dataloader(self, pool_grad):
    self.pool_grad = pool_grad
    if len(pool_grad) != 0:
        self.reconstruction()           # Gradient Inversion → synthetic data
        self.monitor_dataset.getTestData(self.new_set, self.new_set_label)
        self.best_model_1 = self.best_model_2   # Shift: best cũ lùi một bước
    
    cur_perf = self.monitor()           # Đánh giá model hiện tại với synthetic data
    if cur_perf >= self.best_perf:
        self.best_model_2 = copy.deepcopy(self.model)   # Lưu best model mới
```

**Cơ chế 2 best models:** Server giữ `best_model_1` (từ round trước) và `best_model_2` (round hiện tại). Clients dùng 2 model này làm reference để distillation, giúp model ổn định hơn.

#### 3.2 `gradient2label()` — Suy ra nhãn từ gradient

```python
def gradient2label(self):
    for w_single in self.pool_grad:
        # Trick: argmin của tổng gradient layer cuối = nhãn thật
        pred = torch.argmin(torch.sum(w_single[-2], dim=-1), dim=-1)
        pool_label.append(pred.item())
```

**Lý thuyết:** Với Cross-Entropy loss, gradient của lớp FC cuối có pattern đặc biệt cho phép suy ra nhãn chính xác 100% mà không cần brute-force.

#### 3.3 `reconstruction()` — Gradient Inversion Attack

**Tương ứng bài báo:** Section 4.1 (Gradient Inversion for Data Reconstruction)

```python
def reconstruction(self):
    pool_label = self.gradient2label()   # Suy ra nhãn từ gradient
    
    for label_i in range(self.numclass):
        # Khởi tạo dummy_data ngẫu nhiên
        if self.dataset_type == 'tabular':
            dummy_data = torch.randn((1, 32)).requires_grad_(True)   # Vector 1D
        else:
            dummy_data = torch.randn((1, 3, 32, 32)).requires_grad_(True)  # Ảnh 32×32
        
        # L-BFGS optimizer để tìm dummy_data tối ưu
        optimizer = torch.optim.LBFGS([dummy_data], lr=0.1)
        
        for iters in range(self.Iteration):   # 250 iterations
            def closure():
                pred = recon_model(dummy_data)
                dummy_loss = criterion(pred, label_pred)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, recon_model.parameters(), create_graph=True)
                
                # Minimize: ||gradient(dummy) - gradient(real)||²
                grad_diff = sum(((gx - gy)**2).sum() for gx, gy in zip(dummy_dy_dx, grad_truth_temp))
                grad_diff.backward()
                return grad_diff
            
            optimizer.step(closure)
            
            # Thu thập 20 mẫu cuối (augmentation)
            if iters >= self.Iteration - self.num_image:
                augmentation.append(dummy_data.clone().squeeze(0).detach().cpu().numpy())
```

**Gradient Inversion:** Tìm input `x*` sao cho:
```
x* = argmin ||∇L(model, x*, y) - ∇L_received||²
```
Dùng **L-BFGS** (optimizer bậc 2) vì hiệu quả hơn SGD cho bài toán optimization này.

#### 3.4 `monitor()` — Kiểm tra chất lượng model

```python
def monitor(self):
    # Dùng synthetic data (tái tạo từ gradient) để đánh giá model
    correct, total = 0, 0
    for step, (imgs, labels) in enumerate(self.monitor_loader):
        outputs = self.model(imgs)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
    accuracy = 100 * correct / total
    return accuracy
```

**Vai trò:** Server không có dữ liệu thật nhưng vẫn có thể đánh giá model thông qua dữ liệu tái tạo.

---

### 🔵 Module 4: `Fed_utils.py` — Tiện ích Federated Learning

**Tương ứng bài báo:** Algorithm 1 (steps: local training, FedAvg, evaluation)

#### 4.1 `local_train()` — Huấn luyện cục bộ

```python
def local_train(clients, index, model_g, task_id, model_old, ep_g, old_client):
    clients[index].model = copy.deepcopy(model_g)    # Copy model từ server
    
    clients[index].beforeTrain(task_id, group)        # Chuẩn bị data
    clients[index].update_new_set()                   # Phát hiện task mới, cập nhật exemplar
    clients[index].train(ep_g, model_old)             # Train với combined loss
    
    local_model = clients[index].model.state_dict()   # Lấy weights sau train
    proto_grad = clients[index].proto_grad_sharing()  # Lấy gradient để gửi server
    
    return local_model, proto_grad
```

#### 4.2 `FedAvg(models)` — Federated Averaging

```python
def FedAvg(models):
    w_avg = copy.deepcopy(models[0])
    for k in w_avg.keys():          # Mỗi layer
        for i in range(1, len(models)):
            w_avg[k] += models[i][k]
        w_avg[k] = torch.div(w_avg[k], len(models))   # Trung bình
    return w_avg
```

**FedAvg gốc (McMahan 2017):** Đơn giản nhất, unweighted average. GLFC không weight theo số mẫu.

#### 4.3 `participant_exemplar_storing()` — Đồng bộ exemplar cho mọi client

```python
def participant_exemplar_storing(clients, num, model_g, old_client, task_id, clients_index):
    for index in range(num):
        clients[index].model = copy.deepcopy(model_g)    # Cập nhật model mới
        if index not in clients_index:                    # Client KHÔNG được chọn train
            clients[index].beforeTrain(task_id, ...)
            clients[index].update_new_set()              # Vẫn phải cập nhật exemplar!
```

**Lý do quan trọng:** Mọi client (kể cả không được chọn train round này) đều cần cập nhật exemplar để khi được chọn ở round sau, họ đã có exemplar đúng.

#### 4.4 `model_global_eval()` — Đánh giá model toàn cục

```python
def model_global_eval(model_g, test_dataset, task_id, task_size, device):
    # Test trên TẤT CẢ class từ task 0 đến task hiện tại
    test_dataset.getTestData([0, task_size * (task_id + 1)])
    ...
    accuracy = 100 * correct / total
    return accuracy
```

---

### 🔵 Module 5: `myNetwork.py` — Kiến trúc mạng

**Tương ứng bài báo:** Section 3.1 (Model Architecture)

#### 5.1 `network` — Model chính (Incremental)

```python
class network(nn.Module):
    def __init__(self, numclass, feature_extractor):
        self.feature = feature_extractor              # Backbone (MLP hoặc ResNet)
        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass)   # Classifier
    
    def Incremental_learning(self, numclass):
        # Mở rộng fc layer khi có class mới, GIỮ NGUYÊN weights cũ
        weight = self.fc.weight.data        # Lưu weights cũ
        bias = self.fc.bias.data
        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight   # Khôi phục weights cũ
        self.fc.bias.data[:out_feature] = bias
```

**Incremental learning:** Khi task mới đến, thêm neuron mới vào lớp FC mà không xóa neuron cũ.

#### 5.2 `MLP_FeatureExtractor` — Backbone cho tabular *(custom)*

```python
class MLP_FeatureExtractor(nn.Module):
    def __init__(self, in_dim=32, hidden=32):
        self.body = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(),     # Layer 1
            nn.Linear(32, 32), nn.ReLU()      # Layer 2
        )
        self.fc = nn.Linear(32, 1)   # Dummy: chỉ để lấy in_features=32
    
    def forward(self, x):
        return self.body(x)   # Output: (batch, 32) feature vectors
```

#### 5.3 `MLP_Encoder` — Encoder cho Gradient Inversion *(custom)*

```python
class MLP_Encoder(nn.Module):
    # Dùng cho ProxyServer để tính gradient inversion
    # Cấu trúc khác với FeatureExtractor: có lớp FC cuối → num_classes
    def forward(self, x):
        out = self.body(x)
        out = self.fc(out)    # Output: (batch, num_classes) logits
        return out
```

#### 5.4 `LeNet` — Encoder gốc cho ảnh

```python
class LeNet(nn.Module):
    # 3 Conv layers với Sigmoid activation
    # Input: (3, 32, 32) → Output: num_classes logits
    # Dùng cho Gradient Inversion với ảnh (CIFAR100)
```

---

### 🔵 Module 6: `FederatedTabularDataset.py` — Dataloader tabular *(mới thêm)*

**Không có trong bài báo gốc** — Đây là phần mở rộng của repo để hỗ trợ dữ liệu tabular.

#### Cấu trúc dữ liệu input

```
federated_data/
├── client0_task1.pt   → (data_tensor: [N,32], label_tensor: [N])
├── client0_task2.pt
├── ...
└── client4_task6.pt   # 5 clients × 6 tasks = 30 files
```

#### Các method chính

```python
class FederatedTabularDataset(Dataset):
    
    def load_task(self, task_id):
        # Load file .pt theo client và task
        filepath = f'client{self.client_id}_task{task_id}.pt'
        data, targets = torch.load(filepath)
        return data, targets
    
    def set_task(self, task_id):
        self.current_task = task_id + 1   # 0-indexed → 1-indexed filename
    
    def getTrainData(self, classes, exemplar_set=[], exemplar_label_set=[]):
        # Kết hợp: data task hiện tại + exemplar từ các task cũ
        if len(exemplar_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]    # Exemplar cũ
        data, targets = self.load_task(self.current_task)
        datas.append(data)          # Thêm data mới
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
    
    def getTestData(self, classes):
        # Load từ 30_test_data.pt, filter theo class range
        data, targets = torch.load(self.test_file)
        for label in range(classes[0], classes[1]):
            subset_data = data[targets == label]
            ...
    
    def get_image_class(self, label):
        # Lấy tất cả mẫu của class label (dùng cho exemplar selection)
        return self.TrainData[self.TrainLabels == label]
    
    def __getitem__(self, index):
        # Trả về (index, tensor_float32, label_long) cho DataLoader
        return index, torch.tensor(img, dtype=torch.float32), torch.tensor(target, dtype=torch.long)
```

---

### 🔵 Module 7: `proxy_data.py` — Dataset cho Proxy Server

**Tương ứng bài báo:** Section 4.2 (Monitor Dataset)

```python
class Proxy_Data():
    def getTestData(self, new_set, new_set_label):
        # new_set: list of lists of synthetic samples (từ gradient inversion)
        # new_set_label: nhãn tương ứng
        self.TestData, self.TestLabels = self.concatenate(datas, labels)
    
    def getTestItem(self, index):
        data = self.TestData[index]
        
        if len(data.shape) == 1:
            # Tabular: trả về tensor trực tiếp
            img = torch.tensor(data, dtype=torch.float32)
        else:
            # Image: convert từ numpy → PIL → apply transform
            img = Image.fromarray(data)
            if self.test_transform:
                img = self.test_transform(img)
        
        return img, target
```

**Vai trò:** Wraps dữ liệu tái tạo (synthetic) từ `ProxyServer.reconstruction()` thành `Dataset` chuẩn để `DataLoader` có thể dùng trong `monitor()`.

---

### 🔵 Module 8: `option.py` — Cấu hình CLI

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--dataset` | `cifar100` | `cifar100` / `tiny_imagenet` / `tabular` |
| `--numclass` | `10` | Số class trong **task đầu tiên** |
| `--task_size` | `10` | Số class **tăng thêm** mỗi task |
| `--num_clients` | `30` | Tổng số clients ban đầu |
| `--local_clients` | `10` | Số clients chọn tham gia mỗi round |
| `--epochs_global` | `100` | Tổng số global rounds |
| `--tasks_global` | `10` | Số rounds mỗi task |
| `--epochs_local` | `20` | Số epoch huấn luyện local mỗi round |
| `--learning_rate` | `2.0` | Learning rate (khá lớn vì dùng SGD) |
| `--memory_size` | `2000` | Tổng kích thước exemplar memory |
| `--device` | `0` | GPU ID (`-1` = CPU) |
| `--seed` | `2021` | Random seed |

> **Tabular override:** `fl_main.py` tự override `num_clients=5`, `tasks_global=6`, `numclass=34`, `task_size=6` khi `--dataset tabular`.

---

### 🔵 Module 9: `ResNet.py` — Backbone cho ảnh (gốc)

**Tương ứng bài báo:** Appendix (Implementation Details)

```
ResNet-18 backbone:
   Input (3×32×32)
   → BasicBlock × 8 (với CBAM attention)
   → AdaptiveAvgPool
   → Output: 512-dim feature vector
```

**CBAM** (Convolutional Block Attention Module):
- **Channel Attention:** Học xem kênh nào quan trọng
- **Spatial Attention:** Học xem vùng spatial nào quan trọng
- Giúp model tập trung vào đặc trưng thực sự có ích

---

## 📌 5. Cơ chế Privacy — Gradient Inversion

```
CLIENT (có dữ liệu thật)          SERVER (không có dữ liệu thật)
─────────────────────────────     ──────────────────────────────
1. Tìm prototype (mẫu đại diện)
2. Fine-tune prototype 50 steps
3. Tính gradient của encode_model
                     ──── gradient ────→
                                   4. gradient2label(): suy ra nhãn
                                   5. Khởi tạo dummy_data ngẫu nhiên
                                   6. L-BFGS 250 iterations:
                                      minimize ||grad(dummy) - grad_received||²
                                   7. Được synthetic data ≈ prototype thật
                                   8. monitor(): đánh giá model với synthetic data
```

**Tại sao không phải tấn công hoàn toàn?** Đây là **cooperative gradient inversion** — client CHỦ ĐỘNG gửi gradient của prototype để server có thể monitor. Đây là thiết kế có chủ ý của GLFC, không phải lỗ hổng bảo mật.

---

## 📌 6. Dòng chảy dữ liệu hoàn chỉnh (Tabular mode)

```
federated_data/client0_task1.pt
        ↓
FederatedTabularDataset.load_task(1)
        ↓ (data: [N,32], labels: [N])
GLFC_model.beforeTrain() → current_class = [0,1,2,3,4,5]
        ↓
GLFC_model.update_new_set()
  ├── entropy_signal() → phát hiện task mới không?
  ├── _construct_exemplar_set() → chọn m mẫu/class
  └── getTrainData(classes, exemplar_set) → mix data + exemplar
        ↓
GLFC_model.train()
  ├── loss_cur = BCEWithLogits(output, one_hot_label)    [class mới]
  └── loss_old = BCEWithLogits(output, distill_from_old_model)  [class cũ]
        ↓
GLFC_model.prototype_mask()
  ├── Tìm prototype → fine-tune → tính gradient
  └── Trả về proto_grad (list of gradient tensors)
        ↓
proxyServer.reconstruction(proto_grad)
  ├── gradient2label() → suy ra nhãn
  └── L-BFGS 250 iter → synthetic tabular vector (1, 32)
        ↓
proxyServer.monitor() → accuracy trên synthetic data
  └── Lưu best_model_1, best_model_2
        ↓
FedAvg(w_local) → model_g mới
        ↓
model_global_eval() → test accuracy trên 30_test_data.pt
```

---

## 📌 7. Tóm tắt đối chiếu Paper ↔ Code

| Khái niệm trong Paper | File trong Code | Function cụ thể |
|---|---|---|
| Local Forgetting Compensation | `GLFC.py` | `_compute_loss()`: `loss_cur + loss_old` |
| Task Detection | `GLFC.py` | `entropy_signal()` |
| Exemplar Management (Herding) | `GLFC.py` | `_construct_exemplar_set()`, `_reduce_exemplar_sets()` |
| Adaptive Weighting | `GLFC.py` | `efficient_old_class_weight()` |
| Prototype Gradient Sharing | `GLFC.py` | `prototype_mask()` → `proto_grad_sharing()` |
| Global Forgetting Compensation | `ProxyServer.py` | `reconstruction()` + `monitor()` |
| Gradient Inversion | `ProxyServer.py` | `reconstruction()` với L-BFGS |
| Label Recovery from Gradient | `ProxyServer.py` | `gradient2label()` |
| Best Model Tracking | `ProxyServer.py` | `best_model_1`, `best_model_2` |
| FedAvg | `Fed_utils.py` | `FedAvg()` |
| Incremental Classifier | `myNetwork.py` | `network.Incremental_learning()` |

---

*Tài liệu tạo ngày: 2026-04-01*
