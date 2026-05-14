# Benchmark v5 — Ket Qua Phan Tich

## 1. Tong Quan

**Thoi gian:** 7.3 phut
**Moi truong:** NVIDIA RTX 3090 Ti, 18 cores
**Mau:** 6 thang (Jan-Jun 2024), 10K train, 10K test/fold
**Seeds:** 3 (42, 123, 456)
**Khoang cach:** 3 (easy, medium, hard)
**Tong jobs:** 405 (9 algorithms x 3 difficulties x 5 folds x 3 seeds)

## 2. Bang Xep Hang Overall (AUC-PR)

| Hang | Algorithm | AUC-PR | Std | AUC-ROC |
|------|-----------|--------|-----|---------|
| 1 | **sklearn_OCSVM** | 0.229 | 0.185 | 0.875 |
| 2 | MemStream | 0.151 | 0.152 | 0.956 |
| 3 | LSTM-AE (GPU) | 0.125 | 0.155 | 0.937 |
| 4 | sklearn_LOF | 0.114 | 0.117 | 0.754 |
| 5 | CA-DIF-EIA (streaming) | 0.068 | 0.094 | 0.769 |
| 6 | CA-DIF-EIA | 0.067 | 0.093 | 0.772 |
| 7 | sklearn_IF | 0.033 | 0.045 | 0.926 |
| 8 | sHST-River | 0.023 | 0.050 | 0.804 |
| 9 | IForestASD | 0.011 | 0.022 | 0.702 |

## 3. Ket qua theo Khoang Cach

### Easy (meter_mult 10-20x, speed 50-95mph)
| Algorithm | AUC-PR | F1 | AUC-ROC | Recall |
|-----------|--------|-----|---------|--------|
| sklearn_OCSVM | **0.376** | 0.187 | 0.891 | 0.710 |
| MemStream | 0.323 | **0.231** | **0.982** | 0.600 |
| LSTM-AE | 0.316 | 0.211 | 0.968 | 0.547 |
| sklearn_LOF | 0.230 | 0.194 | 0.801 | 0.523 |
| CA-DIF-EIA | 0.163 | 0.153 | 0.836 | 0.414 |
| CA-DIF-EIA (stream) | 0.163 | 0.152 | 0.826 | 0.421 |
| sklearn_IF | 0.058 | 0.104 | 0.963 | 0.417 |
| sHST-River | 0.033 | 0.043 | 0.903 | 0.356 |
| IForestASD | 0.022 | 0.049 | 0.808 | 0.323 |

### Medium (meter_mult 4-8x, speed 30-60mph)
| Algorithm | AUC-PR | F1 | AUC-ROC | Recall |
|-----------|--------|-----|---------|--------|
| sklearn_OCSVM | **0.185** | 0.150 | 0.887 | 0.667 |
| MemStream | 0.097 | **0.181** | **0.973** | 0.578 |
| sklearn_LOF | 0.090 | 0.160 | 0.779 | 0.499 |
| LSTM-AE | 0.039 | 0.078 | 0.950 | 0.278 |
| sklearn_IF | 0.034 | 0.034 | 0.942 | 0.563 |
| sHST-River | 0.034 | 0.022 | 0.828 | 0.354 |
| CA-DIF-EIA | 0.033 | 0.074 | 0.785 | 0.281 |
| CA-DIF-EIA (stream) | 0.032 | 0.077 | 0.785 | 0.312 |
| IForestASD | 0.011 | 0.017 | 0.721 | 0.331 |

### Hard (meter_mult 1.5-3x, speed 20-40mph)
| Algorithm | AUC-PR | F1 | AUC-ROC | Recall |
|-----------|--------|-----|---------|--------|
| sklearn_OCSVM | **0.126** | 0.059 | 0.847 | 0.601 |
| MemStream | 0.034 | **0.073** | **0.914** | 0.260 |
| sklearn_LOF | 0.023 | 0.068 | 0.681 | 0.244 |
| LSTM-AE | 0.020 | 0.050 | 0.893 | 0.522 |
| sklearn_IF | 0.006 | 0.023 | 0.873 | 0.441 |
| CA-DIF-EIA (stream) | 0.007 | 0.020 | 0.695 | 0.231 |
| CA-DIF-EIA | 0.006 | 0.022 | 0.696 | 0.225 |
| sHST-River | 0.002 | 0.009 | 0.681 | 0.340 |
| IForestASD | 0.002 | 0.006 | 0.578 | 0.249 |

## 4. Phan Tich Chi Tiet

### 4.1 OCSVM la Overall Winner
sklearn_OCSVM dat AUC-PR cao nhat (0.229 overall), danh gia tot nhat tren tat ca 3 muc kho khan. Dac biet:
- **Easy:** 0.376 AUC-PR, 71% recall
- **Medium:** 0.185 AUC-PR
- **Hard:** 0.126 AUC-PR (van con > sklearn_IF 0.006)

Lý do: OCSVM tim boundary tu phan ap dung cua data, hoac hoc chinh xac phan phoi cua normal data, phu hop voi taxi fraud (diem outliers can duoc phat hien).

### 4.2 MemStream la Streaming Winner
MemStream dat 0.151 AUC-PR overall trong nhom streaming, voi:
- AUC-ROC rat cao (0.956 overall) — phan biet tot giua normal/anomaly
- F1 cao nhat trong nhom (0.162 overall)
- Xep thu 2 overall, sau OCSVM

### 4.3 CA-DIF-EIA: Khong Co Improvement so voi Baseline
**Van de lon nhat:** CA-DIF-EIA (batch) = 0.067, CA-DIF-EIA (streaming) = 0.068 — giong nhau va khong tot hon sklearn_IF (0.033).

Nguyen nhan:
1. **Context weights khong hieu qua:** Feature weights hoc tu 3000 mau chi la proxy yeu cho correlation voi scores, khong phai actual context
2. **DIF projection khong cai thien:** Random projection (W1, b1, W2, b2) chi la linear transform, khong co learning — equivalent voi PCA + IF
3. **Threshold tai 97th percentile:** Qua cao cho imbalanced data

### 4.4 LSTM-AE: Tot tren Easy, Giam nhanh
- Easy: 0.316 AUC-PR (thu 3)
- Medium: 0.039
- Hard: 0.020

Anomaly subtler thi reconstruction error giam, kha nang phan biet giarm.

### 4.5 Streaming vs Batch
| Method | Batch | Streaming |
|--------|-------|----------|
| IF-based | sklearn_IF 0.033 | IForestASD 0.011, sHST-River 0.023 |
| Mem | N/A | MemStream 0.151 |
| OCSVM | sklearn_OCSVM 0.229 | N/A |

Streaming methods (ngoai MemStream) deu kem hon batch counterparts.

## 5. Loi Phan Tich

### 5.1 AUC-PR Thap (Precision-Recall Trade-off)
AUC-PR dao dong 0.01-0.38, thap hon AUC-ROC (0.58-0.98). Nguyen nhan:
- **Imbalanced data:** 5000 anomalies / 10000 samples = 50% anomaly rate trong test set
- **Precision thap:** voi threshold tai 97th percentile, nhieu false positives
- **Recall khong du cao:** threshold qua cao lam miss nhieu true anomalies

### 5.2 Contamination Parameter
sklearn.IF/LOF/OCSVM dung `contamination=0.05` nhung actual anomaly rate trong test la 5% (2500/10500). Gia tri tot nhat cho contamination:
- Easy: anomaly rat rat cao, contamination nen tang
- Hard: anomaly rat thap, contamination nen giam

### 5.3 CA-DIF-EIA Failure Modes
1. **Projection qua don gian:** 2-layer ReLU MLP khong hoc duoc — equivalent voi random projection
2. **Context weights hoc tu small sample:** chi 3000 mau, correlation voi scores khong stable
3. **Threshold khong adaptive:** co dinh tai 97th percentile, khong theo kip concept drift

## 6. Khuyen Nghi Cai Thien

### 6.1 OCSVM + Adaptive Threshold
Thay vi co dinh 97th percentile, dung:
- Grid search threshold tren validation set
- Adaptive threshold theo contamination rate cua tung difficulty

### 6.2 CA-DIF-EIA v2
Neu muon CA-DIF-EIA tot hon:
- **DIF that su:** Thay random projection bang trained autoencoder bottleneck
- **Context-aware split:** Chia data theo temporal context (gio, ngay trong tuan) truoc khi detect
- **Adaptive threshold:** Hoc threshold tu labeled subset

### 6.3 MemStream Enhancement
- Tang `memory_size` tu 500 len 2000 (hien tai: 500)
- Them drift detection de reset memory khi concept thay doi

## 7. Ket Luan

| Criteria | Winner | Score |
|----------|--------|-------|
| Overall AUC-PR | sklearn_OCSVM | 0.229 |
| Streaming AUC-PR | MemStream | 0.151 |
| AUC-ROC | MemStream | 0.956 |
| F1 Overall | MemStream | 0.162 |
| Speed (score_ms) | sklearn_OCSVM | 145ms |
| GPU-accelerated | LSTM-AE | 0.125 |

**Key findings:**
1. **OCSVM > Isolation Forest** cho imbalanced taxi fraud detection
2. **MemStream** la best streaming algorithm, nhat la khi combine voi drift detection
3. **CA-DIF-EIA chua co improvement** so voi sklearn_IF — can refactor DIF that su voi learned representations
4. **LSTM-AE** hoat dong tot tren GPU nhung giam nhanh khi anomaly subtler
5. **IForestASD/sHST-River** khong tot cho this task — can larger windows va trained components
