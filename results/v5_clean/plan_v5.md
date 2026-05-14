# Benchmark v3 — Diagnostic Report & v5 Upgrade Plan

> **Document Type:** Technical Upgrade Plan
> **Version:** 5.0 (Draft)
> **Date:** 2026-05-12
> **Status:** Draft — for review before implementation
> **Inputs:** benchmark_v3.py source code (1,080 lines), 8 result CSV files, coordinator state

---

## 1. Executive Summary

Benchmark v3 đã chạy thành công phần lớn nhưng có **5 vấn đề nghiêm trọng** khiến kết quả không đáng tin cậy và thiếu sót. Plan v5 này phân tích chi tiết root cause từng vấn đề và đề ra hành động cụ thể cho từng model.

### 1.1 Status hiện tại

```
Table A (Batch):
  ✅ sklearn_IF       — chạy OK, AUC-PR ~0.10-0.57
  ✅ sklearn_OCSVM   — chạy OK, AUC-PR ~0.53-0.89  [nhưng KHÔNG có trong thống kê chính thức]
  ✅ sklearn_LOF     — chạy OK, rank #1
  ❌ LSTM-AE         — 990 runs FAILED (GPU pin_memory error)
  ✅ CA-DIF-EIA      — chạy OK, rank #3
  ✅ METER-SCD       — chạy OK, rank #4

Table B (Streaming):
  ❌ sHST-River      — chạy OK nhưng hoàn toàn thất bại (threshold=1.0, AUC-PR≈0.524=random)
  ⚠️  MemStream       — easy OK (~0.82), hard thất bại (~0.20), KHÔNG consume labels
  ❌ IForestASD      — rank cuối, high variance, AUC-PR ~0.06-0.30
  ❌ CA-DIF-EIA      — KHÔNG xuất hiện trong streaming results
```

### 1.2 Cấp độ ưu tiên

| Priority | Issue | Model | Effort | Impact |
|----------|-------|-------|--------|--------|
| P0 | Fix GPU error | LSTM-AE | Low | Unblocks DL baseline |
| P0 | Implement CA-DIF-EIA streaming | CA-DIF-EIA (B) | High | Core contribution |
| P0 | Fix threshold disaster | sHST-River | Low | Unblocks streaming table |
| P1 | Debug MemStream label consumption | MemStream | Medium | Correct BAR score |
| P1 | Fix IForestASD sliding window | IForestASD | Medium | Streaming table accuracy |
| P1 | Add OCSVM to statistics | sklearn_OCSVM | Low | Complete Table A |
| P1 | Compute all difficulty stats | All | Low | Full picture |
| P2 | Refine benchmark protocol | Various | Medium | Scientific rigor |

---

## 2. Root Cause Analysis Per Model

---

### 2.1 sklearn_LOF — Rank #1 (Good)

**Kết quả:** AUC-PR batch rank #1 ở mọi difficulty.

**Điểm mạnh cần giữ:**
- Density-based, dùng local outlier factor
- Neighborhood-based approach phù hợp với taxi data structure
- Easy AUC-PR ~0.55-0.58, Hard vẫn rank 1

**Điều cần cải thiện cho v5:**
- Thử `n_neighbors` grid search: hiện hardcoded 20. Thử [10, 15, 20, 30, 50].
- Thử `contamination` grid: hiện hardcoded 0.05. Với contamination thực tế ~0.83%, thử [0.01, 0.03, 0.05, 0.1].
- **Tại sao quan trọng:** LOF là một trong những base line mạnh nhất. Việc tune hyperparameters có thể cải thiện thêm 5-10% AUC-PR, đặc biệt ở hard level.

**Action Items:**
- [ ] Grid search n_neighbors và contamination trên fold 0-1 (initial training partition)
- [ ] So sánh LOF batch vs LOF với context-weighted KNN distance (tương tự CA-DIF-EIA context grid)

---

### 2.2 sklearn_IF — Rank #2 (Good, có room to grow)

**Kết quả:** Rank #2 ổn định, nhưng AUC-PR drop mạnh ở hard (0.43→0.10).

**Root cause của sự sụt giảm ở hard:**
- Isolation Forest dùng axis-parallel splits — tạo "ghost regions" khi anomaly nằm ở diagonal boundaries hoặc non-axis-aligned regions.
- Hard anomalies (meter_mult 1.5-3x) tạo ra subtle deviations không nằm trong sparse axis-aligned regions.
- IF đo **global isolation** — không có sense về local density.

**Action Items:**
- [ ] Tăng `n_estimators`: hiện 200 → thử [200, 300, 500]. Nhiều trees hơn cải thiện stability của path length estimates.
- [ ] Thử `max_samples`: hiện mặc định. Thử [256, 512, 1024]. Với taxi data cỡ ~1M records/train fold, subsampling strategy quan trọng.
- [ ] Thử `bootstrap=True` thay vì subsampling cố định.
- [ ] Không cần thay đổi gì cho benchmark protocol vì IF đã là solid baseline.

---

### 2.3 sklearn_OCSVM — "Ghost entry" (Critical bug)

**Kết quả:** Có trong raw data (benchmark_results_batch.csv) nhưng **KHÔNG** có trong `cd_ranks_batch.csv` hay `statistical_tests_batch.csv`.

**Root cause:** Xem code, không có bug rõ ràng trong OCSVM class. Vấn đề nằm ở **statistical analysis pipeline**.

Kiểm tra code statistical analysis (dòng 939-945):
```python
for m_i in range(1, 12):           # folds 1-11
    for a_i, an in enumerate(group_algos):
        sub = df_d[(df_d['fold'] == m_i) & (df_d['algorithm'] == an)]
        sm[m_i-1, a_i] = sub['AUC_PR'].mean()
```

Tất cả 6 algorithms trong BATCH_ALGOS được xử lý. Grep cho thấy OCSVM có data. Vấn đề có thể là:
1. OCSVM chỉ chạy với `ablation='control'` trong job loop (dòng 844-873), không chạy với `ablation='treatment'`
2. Hoặc OCSVM results bị ghi đè/lọc bởi logic nào đó

Thực tế nhìn vào dòng 858-861:
```python
for algo_cls in BATCH_ALGOS + STREAM_ALGOS:  # <- cả 6 batch algorithms
    for seed_v in SEEDS:
        bench_jobs.append(...)
```

Và dòng 844: `for X_train, scaler, ablation in [(X_train_A, scaler_A, 'control'), (X_train_B, scaler_B, 'treatment')]:`

Nghĩa là **mỗi algorithm chạy 2 lần** (control + treatment), mỗi lần 10 seeds × 3 difficulties × 11 folds = 330 rows/algorithm.

OCSVM có data trong raw CSV. Có thể statistical analysis code không đọc đúng file, hoặc có filtering logic loại bỏ OCSVM.

**Action Items:**
- [ ] Kiểm tra `benchmark_results_batch.csv` — grep OCSVM: `rg "sklearn_OCSVM" results/v3/benchmark_results_batch.csv | wc -l`
- [ ] Nếu có data: fix statistical analysis code để include OCSVM
- [ ] Nếu không có data: debug job generation — OCSVM phải nằm trong BATCH_ALGOS
- [ ] Thử `kernel='poly', degree=3` thay vì `kernel='rbf'` — polynomial kernel có thể capture taxi fare non-linear relationships tốt hơn
- [ ] Thử `gamma='auto'` thay vì `gamma='scale'`

---

### 2.4 CA-DIF-EIA (Batch) — Rank #3 (OK, nhưng chưa phải sức mạnh thật)

**Kết quả:** Rank #3 ổn định, ablation cho thấy Context-aware Grid có lợi (+3-6%).

**Root cause tại sao rank #3:**
- CA-DIF-EIA batch implementation **không khác gì sklearn_IF** ngoài thresholding:
  ```python
  class CADIFEia:
      def fit(self, X):
          self.if_ = IsolationForest(...)
          self.if_.fit(X)
          raw = -self.if_.score_samples(X)
          self.thresh_ = float(np.percentile(raw, 97))
      def decision_function(self, X):
          return -self.if_.score_samples(X)  # <- giống hệt sklearn_IF
  ```
- **"Context-aware" không có trong batch implementation!** Không có context feature weighting, không có DIF (Deep Isolation Forest) — chỉ là sklearn_IF với threshold 97th percentile.
- "Deep Isolation Forest" (RNN-based random projection trước khi isolation) chưa được implement.

**Action Items — CA-DIF-EIA Batch:**
- [ ] Implement **DIF (Deep Isolation Forest)**: trước khi feed vào IsolationForest, pass qua 2-3 layers của Random Neural Networks (weights ngẫu nhiên cố định, không train). Đây là phần "Deep" trong tên.
- [ ] Implement **Context-aware weighting**: sau khi có isolation scores, nhân với context feature weights learned từ data (hoặc fixed domain knowledge weights).
- [ ] Ablation trong v5 cần test 3 configs: (1) vanilla IF, (2) DIF only (deep projection), (3) CA-DIF-EIA (DIF + context weighting).

---

### 2.5 CA-DIF-EIA (Streaming) — HOÀN TOÀN VẮNG MẶT (P0)

**Kết quả:** Không xuất hiện trong benchmark_results_streaming.csv, bar_score_results.csv. Coordinator không ghi completed phase nào liên quan đến streaming CA-DIF-EIA.

**Root cause:** 
- CA-DIF-EIA streaming variant **chưa được implement** trong code.
- BATCH_ALGOS và STREAM_ALGOS là 2 danh sách tách biệt. CA-DIF-EIA chỉ nằm trong BATCH_ALGOS.
- Không có class `CADIFEiaStreaming` trong code.
- **Đây là model quan trọng nhất** — nó là đóng góp chính của nghiên cứu.

**Action Items — CA-DIF-EIA Streaming (CRITICAL):**
- [ ] Implement class `CADIFEiaStreaming`:
  ```
  1. Warm-up: fit IF trên first 20% train data
  2. ADWIN-U: monitor anomaly score stream (không phải labels) cho drift detection
  3. Nếu drift detected AND label_budget > 0: request 1 label, update model
  4. Nếu no drift: continue scoring (no update)
  ```
- [ ] ADWIN-U cần monitor theo mean và variance của anomaly score stream, KHÔNG phải raw input features
- [ ] Drift detection threshold: cần tunning. ADWIN delta = 0.002 có thể quá nhạy hoặc quá thụ động
- [ ] Context weighting trong streaming: weight = f(hour, day_of_week, location_cluster) — pre-computed không cần online computation
- [ ] Label budget consumption strategy: ưu tiên drift windows, không phải random sampling
- [ ] Add CA-DIF-EIA streaming vào STREAM_ALGOS

**Design chi tiết cho ADWIN-U module:**
```
ADWIN-U parameters:
  - delta: confidence level for drift detection (thử 0.001, 0.002, 0.005, 0.01)
  - window_size: max size of score window (thử 500, 1000, 2000)
  - min_window: minimum window size before checking (thử 100, 200, 500)
  
Drift detection logic:
  1. Compute running mean μ and variance σ² of score window
  2. On each new score s_new:
     a. Add s_new to window, evict oldest if full
     b. If window size >= min_window:
        - Split window into two halves
        - Compute μ1, σ1² and μ2, σ2²
        - If |μ1 - μ2| > α × √(σ1²/n1 + σ2²/n2) for some α: DRIFT
  3. α tuning: thử [1.5, 2.0, 2.5, 3.0]
```

---

### 2.6 METER-SCD — Rank #4 (Cần streaming mode)

**Kết quả:** Rank #4, chỉ hơn LSTM-AE (thất bại) và tương đương CA-DIF-EIA.

**Root cause:**
- METER-SCD trong code **không phải là METER-SCD thật**:
  ```python
  class METERSCD:
      def fit(self, X):
          self.if_ = IsolationForest(...)
          self.if_.fit(X)
          self.thresh_ = float(np.percentile(-self.if_.score_samples(X), 95))
  ```
- Đây chỉ là IsolationForest với threshold 95th percentile. Không có hypernetwork, không có SCD module, không có concept drift adaptation.
- METER-SCD batch degenerate về IF cơ bản.

**Action Items:**
- [ ] **Thận trọng:** METER-SCD (Zhu et al., VLDB 2024) là một nghiên cứu thực, không phải open source. Không thể implement đúng nếu không có paper details.
- [ ] Giữ nguyên như một "simplified SCD module" baseline
- [ ] Tập trung nguồn lực vào CA-DIF-EIA streaming thay vì cải thiện METER-SCD giả lập

---

### 2.7 LSTM-AE — 990 runs FAILED (P0, Easy fix)

**Kết quả:** Toàn bộ 990 runs lỗi. GPU error trên mọi fold.

**Root cause:**
```
cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
```

**Phân tích code (dòng 355-419):**
```python
class LSTMAE:
    def fit(self, X):
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)
        seq = torch.FloatTensor(Xs.reshape(-1, 1, Xs.shape[1])).to(self._device)  # <- line 376
        
        # ... training ...
        
        # DataLoader với pin_memory (KHÔNG thấy trong code nhưng error gợi ý có)
```

Lỗi `pin_memory` xảy ra khi PyTorch DataLoader được gọi với `pin_memory=True` trên tensors đã nằm trên GPU. Error message đặc trưng:
```
RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
```

Tuy nhiên, trong code đọc được, **không thấy** `DataLoader` được tạo với `pin_memory=True`. Có thể:
1. Error đến từ một DataLoader ở đâu đó trong training loop
2. Hoặc error đến từ PyTorch internals khi PyTorch version không tương thích

Xem kỹ hơn dòng 391-392:
```python
ds    = TensorDataset(seq, seq)
dl    = DataLoader(ds, batch_size=256, shuffle=True)
```

Không có `pin_memory=True`. Nhưng `seq` đã `.to(self._device)` ở dòng 376. DataLoader mặc định `pin_memory=False`, không phải vấn đề.

**Tuy nhiên**, error "cannot pin 'torch.cuda.FloatTensor'" có thể xảy ra khi:
1. `TensorDataset` chứa tensors đã ở GPU, và PyTorch cố gắng pin memory
2. PyTorch version không tương thích với CUDA version

**Action Items:**
- [ ] Fix 1: Không `.to(self._device)` trước khi tạo DataLoader. Để DataLoader tự xử lý:
  ```python
  seq = torch.FloatTensor(Xs.reshape(-1, 1, Xs.shape[1]))  # CPU
  ds = TensorDataset(seq, seq)
  dl = DataLoader(ds, batch_size=256, shuffle=False)
  # Chuyển sang GPU trong training loop
  for bx, by in dl:
      bx, by = bx.to(self._device), by.to(self._device)
  ```
- [ ] Fix 2: Set `torch.backends.cudnn.benchmark = False` để tránh pinning issues
- [ ] Fix 3: Thêm fallback — nếu GPU fails, chạy trên CPU (hiện đã có fallback ở dòng 404-406 nhưng error xảy ra ở training chứ không phải inference)
- [ ] Fix 4: Thử `num_workers=0` trong DataLoader (multiprocessing + CUDA + pin_memory = phổ biến gây lỗi)

---

### 2.8 sHST-River — THẢM HỌA Threshold (P0)

**Kết quả:** AUC-PR ≈ 0.524 (random baseline) trên mọi fold, mọi seed, mọi difficulty.

**Root cause — phân tích triple:**
1. **Custom implementation thay vì river library:**
   ```python
   class sHST_River:
       def decision_function(self, X):
           # scores = normalized ratio count
           ratio = (Xr[:, tf] > sp).sum(axis=1) / self.depth
           scores += np.clip(ratio / 0.000977, 0, 1)
           return scores / self.n_trees
   ```
   Đây là **custom Half-Space Trees**, không phải `river.anomaly.HalfSpaceTrees`. River library là well-tested, đây là self-made.

2. **Custom predict —KHÔNG SỬ DỤNG decision_function score đúng cách:**
   ```python
   def predict(self, X):
       d = self.decision_function(X)
       return np.where(d > np.percentile(d, 95), -1, 1)  # <- self-percentile
   ```
   Đây là **self-referential thresholding**: lấy 95th percentile của test scores rồi apply lên chính test scores. Với streaming model không update, test score distribution giống hệt train → threshold ≈ constant. Với HST scoring scheme, scores ≈ uniform/constant → percentile(95) ≈ max → threshold ≈ max → mọi record below threshold → normal → FPR=1.0.

3. **Streaming evaluation KHÔNG thực sự streaming:**
   Xem code dòng 858: `for algo_cls in BATCH_ALGOS + STREAM_ALGOS`. Cả 3 streaming algorithms đều được đánh giá trong cùng batch evaluation loop (`evaluate()` function, dòng 486). Không có `fit_partial()` — mỗi test record được xử lý batch, không phải online.

**Action Items:**
- [ ] **Option A (Preferred):** Sử dụng `river.anomaly.HalfSpaceTrees` thay vì custom implementation. Đây là well-tested, maintained library.
  ```python
  from river.anomaly import HalfSpaceTrees
  hst = HalfSpaceTrees(
      n_trees=25, depth=10, window_size=250,
      seed=seed, threshold=0.5
  )
  for x in X_test_stream:
      score = hst.score_one(x)
      hst.update(x)  # online update
  ```
- [ ] **Option B (If Option A unavailable):** Fix custom implementation:
    - Change `predict` để dùng **fixed threshold từ training data** thay vì self-percentile
    - Hoặc dùng **sliding window quantile** đúng cách: maintain buffer of scores từ recent window, compute threshold từ buffer
    - Implement actual streaming: `for x in X_test: algo.partial_fit(x); score = algo.decision_function_one(x)`
- [ ] **CRITICAL:** Tách biệt streaming evaluation — streaming algorithms PHẢI dùng `fit_partial` / `score_one` / `update_one` API, không phải batch `fit()` + `decision_function()`.

---

### 2.9 MemStream — Không Consume Labels + Hard Failure (P1)

**Kết quả:** 
- Easy: AUC-PR ~0.82 (rất tốt)
- Medium: AUC-PR ~0.47-0.57
- Hard: AUC-PR ~0.17-0.24 (thất bại)
- AUC-PR GIỐNG HỆT ở mọi label budget → **thuật toán không consume labels**

**Root cause — 3 vấn đề:**

1. **MemStream KHÔNG update trong code hiện tại:**
   ```python
   class MemStream_:
       def fit(self, X):
           # Chỉ fill buffer và memory
           for x in X:
               if len(self.buffer) >= self.buffer_size:
                   evicted = self.buffer.pop(0)
                   if len(self.memory) >= self.memory_size:
                       self.memory.pop(0)
                   self.memory.append(evicted)
               self.buffer.append(x.astype(np.float32))
       def decision_function(self, X):
           # Chỉ compute distance to memory
           mem = np.array(self.memory)
           d = np.linalg.norm(X[:, np.newaxis] - mem, axis=2)
           return sd[:, :k].mean(axis=1)
   ```
   Không có `partial_fit()` / `update_one()` method. Đây là **batch memory initialization**, không phải streaming update. Khi test data đến, memory vẫn chỉ chứa training data — không có adaptation.

2. **`decision_function` không có streaming update:**
   Mỗi khi model nhận record mới, nó nên update memory. Nhưng trong batch evaluation, `decision_function(X)` nhận cùng lúc N records — không có cơ hội update giữa các records.

3. **Self-percentile thresholding trong predict:**
   ```python
   def predict(self, X):
       d = self.decision_function(X)
       return np.where(d > np.percentile(d, 95), -1, 1)
   ```
   Cùng vấn đề như sHST-River — threshold tự tham chiếu.

**Action Items:**
- [ ] Implement `update_one(self, x, label=None)` method:
  ```
  1. If label is available and label == anomaly: add to memory with higher weight
  2. If no label: use density-based detection
     - If distance_to_memory > threshold: potential anomaly
     - Update memory adaptively
  3. Add x to buffer, evict oldest if full
  4. If evicted x is in memory (or close): replace with representative
  ```
- [ ] Implement proper streaming `score_one(x)` method
- [ ] Fix threshold: use fixed threshold from training, or streaming quantile
- [ ] **Implement label consumption:** When drift detected AND label budget > 0: consume label, update memory
- [ ] **Tăng buffer_size và memory_size:** Hiện buffer=500, memory=200. Với test stream ~600K records, cần lớn hơn:
  - buffer_size: thử [1000, 2000, 5000]
  - memory_size: thử [500, 1000, 2000]
  - decay_factor: thử [0.95, 0.99, 0.999] để forget old patterns

**Why MemStream easy works but hard fails:**
- Easy: anomaly distortions rất lớn (fare x10-20) → distance to normal memory >> threshold → easy to detect
- Hard: subtle distortions (fare x1.5-3) → distance barely above threshold → memory confuses with legitimate variation

**Action Items (Fine-tuning for hard difficulty):**
- [ ] Thử Mahalanobis distance thay vì Euclidean: Mahalanobis accounting for feature correlations sẽ phân biệt tốt hơn subtle anomalies từ legitimate variation
- [ ] Per-feature weighting: some features (fare_per_mile, speed) more discriminative than others (hour, day)
- [ ] Multiple memory banks: separate memory for different time contexts (rush hour vs night vs weekend)

---

### 2.10 IForestASD — Cuối bảng + High Variance (P1)

**Kết quả:** Rank #3 (cuối) trong streaming. AUC-PR hard ~0.06 (tệ hơn random!). Seed sensitivity cực cao.

**Root cause — quadruple:**

1. **`decision_function` không hoạt động đúng:**
   ```python
   def decision_function(self, X: np.ndarray) -> np.ndarray:
       scores = np.zeros(len(Xf))
       for x in Xf:
           depth_sum = 0.0
           for fi, sp in self.trees_:
               depth_sum += 0 if x[fi] < sp else 1
           scores[np.where(np.all(Xf == x, axis=1))[0]] = depth_sum / len(self.trees_)
           #                        ^^^^^^^^^^^^^^^^^^^^^^^^ <- BUG! np.all(Xf == x, axis=1) returns True for first match only
           # If there are duplicate Xf rows, only first gets scored
       return scores
   ```
   Đây là **major bug**: `np.where(np.all(Xf == x, axis=1))` trả về tất cả indices trùng với x, nhưng nếu Xf chứa duplicate records (rất phổ biến trong taxi data), chỉ first match được assign score. Unscored records → score=0 → không contribute vào evaluation.

2. **`_partial_fit` không rebuild trees đúng cách:**
   ```python
   def _partial_fit(self, x):
       if len(self.buffer) >= self.window_size:
           self.buffer.pop(0)
       self.buffer.append(x)
       if len(self.buffer) < self.max_samples:
           return
       buf = np.array(self.buffer[-self.max_samples:])
       self.trees_ = []
       for _ in range(self.n_trees):
           idx = self._rng.choice(len(buf), ...)  # <- random sampling từ buffer
           sample = buf[idx]
           feat_i = self._rng.randint(0, feat_dim)
           split = self._rng.uniform(sample[:, feat_i].min(), ...)
           self.trees_.append((feat_i, split))
   ```
   Mỗi khi `_partial_fit` được gọi và buffer đầy, nó rebuild ALL trees từ random samples. Nhưng:
   - Không có label feedback — không biết x là normal hay anomaly
   - Rebuilding trees với sliding window không đủ代表性 — window chỉ 256 records
   - `n_trees=100` nhưng mỗi tree chỉ tốt khi được train trên diverse sample

3. **`window_size = 256` quá nhỏ:**
   Với taxi data distributions thay đổi theo thời gian, 256 records không đủ để capture seasonal patterns. Context window (1 tuần) nên ≥ 10,000 records.

4. **`predict` threshold cố định = 0.5:**
   ```python
   def predict(self, X):
       d = self.decision_function(X)
       return np.where(d > 0.5, -1, 1)
   ```
   0.5 là midpoint của depth. Nhưng depth scores không phân phối đều — có thể skew về 0.8 hoặc 0.2. Dùng fixed 0.5 không adaptive.

**Action Items:**
- [ ] **Bug fix #1:** Rewrite `decision_function`:
  ```python
  def decision_function(self, X: np.ndarray) -> np.ndarray:
      if len(self.trees_) == 0:
          return np.full(len(X), 0.5)
      Xf = X.astype(np.float32)
      scores = np.zeros(len(Xf))
      for idx, x in enumerate(Xf):
          depth_sum = 0.0
          for fi, sp in self.trees_:
              depth_sum += 0 if x[fi] < sp else 1
          scores[idx] = depth_sum / len(self.trees_)
      return scores
  ```
  Direct indexing thay vì `np.where(np.all(...))`.

- [ ] **Feature importance:** Trong mỗi tree, track feature usage frequency. Features ít dùng nên được removed khỏi split candidates để tăng efficiency.

- [ ] **Tăng window_size:** window_size = 256 → [1000, 2000, 5000]. NYC taxi có strong daily patterns (1440 minutes/day). Cần ít nhất 1-2 ngày context.

- [ ] **Tăng max_samples:** max_samples = 256 → [512, 1024, 2048]. Đa dạng hơn trong mỗi tree construction.

- [ ] **Adaptive threshold:** Thay vì fixed 0.5, compute threshold from score distribution:
  ```python
  def predict(self, X):
      d = self.decision_function(X)
      # Option 1: from training scores
      # Option 2: from recent window of scores
      threshold = np.percentile(d, 95)  # top 5% as anomaly
      return np.where(d > threshold, -1, 1)
  ```

- [ ] **Weighted sampling:** Thay vì uniform random sampling cho tree construction, dùng exponential decay weights cho recent samples:
  ```python
  weights = np.exp(-np.arange(len(buf)) / decay_window)
  weights /= weights.sum()
  idx = self._rng.choice(len(buf), size=min(self.max_samples, len(buf)), p=weights)
  ```

---

## 3. Benchmark Protocol Improvements (v5)

### 3.1 True Streaming Evaluation

**Vấn đề hiện tại:** Cả 3 streaming algorithms đều dùng batch `evaluate()` function — không có online update.

**Fix:**
- Tách riêng streaming evaluation loop:
  ```python
  def evaluate_streaming(algo, X_train_warmup, X_test, y_test, seed):
      # Warm-up
      algo.fit(X_train_warmup)
      
      # Streaming: one record at a time
      labels_consumed = 0
      label_budget = ...
      scores = []
      
      for i, x in enumerate(X_test):
          score = algo.score_one(x)
          scores.append(score)
          
          # Label consumption for CA-DIF-EIA streaming
          if hasattr(algo, 'should_update') and algo.should_update() and label_budget > 0:
              if y_test[i] is not None:
                  algo.update_one(x, y_test[i])
                  label_budget -= 1
                  labels_consumed += 1
      
      return scores, labels_consumed
  ```

### 3.2 Threshold Strategy

**Vấn đề hiện tại:** Self-referential thresholds (dùng percentile của test scores để classify test) → circular reasoning.

**Fix — Phân biệt 3 threshold types:**
```
Type A (Batch):        Threshold từ training scores (known labels)
Type B (Streaming):      Adaptive sliding-window quantile từ recent scores  
Type C (Streaming+LB):  Threshold từ labeled subset + apply to unlabeled
```

### 3.3 Fold Configuration

**Vấn đề hiện tại:** Fold 0-10 nhưng coordinator state ghi "12 folds". Có inconsistency.

**Fix:**
- Consistent: folds 0-11 (12 folds) hoặc folds 1-12 (12 folds)
- Report fold index và month name rõ ràng
- Thêm sanity check: verify fold i = train Jan..Month(i-1), test Month(i)

### 3.4 Statistical Analysis

**Vấn đề hiện tại:** Chỉ có hard difficulty được phân tích thống kê. Easy và medium không.

**Fix:**
- Compute Friedman + Wilcoxon + CD cho cả 3 difficulties
- Thêm Friedman p-value ở đầu mỗi group để xác nhận có difference đáng kể không

### 3.5 OCSVM Statistical Inclusion

**Fix:**
- Debug tại sao OCSVM bị exclude từ cd_ranks_batch.csv
- Kiểm tra statistical_tests_batch.csv header — nó chỉ có 12 dòng (hard difficulty pairs), nghĩa là statistical tests CHỈ được compute cho hard
- Cần compute cho cả easy và medium

---

## 4. v5 Implementation Roadmap

### Phase 1: Critical Fixes (1-2 hours)
1. Fix LSTM-AE GPU error (pin_memory fix)
2. Add sklearn_OCSVM vào statistical analysis
3. Compute stats cho easy và medium
4. Fix IForestASD decision_function bug

### Phase 2: Streaming Evaluation Fixes (4-6 hours)
1. Implement true streaming evaluation loop
2. Fix sHST-River: dùng river.anomaly.HST hoặc fix custom implementation
3. Fix MemStream: add update_one(), label consumption
4. Implement CA-DIF-EIA streaming (ADWIN-U + context weighting)

### Phase 3: Model Enhancements (6-8 hours)
1. Implement DIF (Deep Isolation Forest) for CA-DIF-EIA batch
2. Add context-aware weighting to CA-DIF-EIA batch
3. Tune LOF hyperparameters
4. Tune IForestASD window sizes
5. MemStream Mahalanobis distance

### Phase 4: Full Rerun + Analysis (8-10 hours)
1. Run full v5 benchmark: 12 folds × all algorithms × all seeds × all difficulties
2. Full statistical analysis (all difficulties)
3. BAR Score với CA-DIF-EIA streaming
4. Complete ablation study (3 configs: IF, DIF, CA-DIF-EIA)

---

## 5. Expected Outcomes (v5)

| Model | v3 AUC-PR (Hard) | v5 Expected | Improvement |
|-------|-------------------|-------------|--------------|
| sklearn_LOF | ~0.10 (hard case) | ~0.10-0.12 | +0-20% |
| sklearn_IF | ~0.10 | ~0.12-0.15 | +20-50% |
| sklearn_OCSVM | ~0.53 | ~0.53-0.60 | +0-13% |
| CA-DIF-EIA (batch) | ~0.09 | ~0.15-0.20 | +67-122% |
| LSTM-AE | FAILED | ~0.60-0.70 | Unblocked |
| METER-SCD | ~0.09 | ~0.09-0.10 | +0-11% |
| sHST-River | ~0.524 (random) | ~0.60-0.70 | +15-34% |
| MemStream | ~0.20 | ~0.40-0.50 | +100-150% |
| IForestASD | ~0.06 | ~0.30-0.40 | +400-567% |
| CA-DIF-EIA (stream) | N/A | ~0.70-0.80 | New |

---

## 6. Notes

- METER-SCD là proprietary research (VLDB 2024). Không nên cố gắng implement đầy đủ — giữ như simplified baseline.
- CA-DIF-EIA streaming là đóng góp chính. Tập trung nguồn lực vào Phase 2.
- Benchmark protocol improvements (Section 3) nên được merge vào benchmark_v5.py, KHÔNG sửa benchmark_v3.py (giữ nguyên v3 như historical record).
- LSTM-AE chỉ cần chạy được để có DL baseline. Không cần optimize vì DL không phải contribution của nghiên cứu.

---

*Document Status: Draft for review*
*Next step: User reviews plan, identifies priorities, and we begin implementation.*
