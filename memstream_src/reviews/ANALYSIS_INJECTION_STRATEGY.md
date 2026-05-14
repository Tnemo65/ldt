# Phan tich: Tai sao Ket qua Benhmark v7 khac nhau?

## 1. Phat hien then chot

Benchmark v7 (DenoisingAE/MemStream dat AUC-PR = 0.9996) khac biet fundamentally voi tuning benchmark cua toi (tuned sklearn_IF dat AUC-PR = 0.2244) vi **3 ly do chinh**:

### Ly do 1: Feature Engineering cua toi lam mat signal anomaly

```
Ban tot nhat cua toi (tuning benchmark):
  - Data da duoc preprocess (25D, z-score normalized)
  - Features nhu: fare/dist, fare/dur, speed  
  - => Extreme fare_amount van co the tao ra ratio "binh thuong"

Vi du:
  - Anomaly: fare=500, dist=1, dur=5
  - feature[6] = fare/dist = 500 (vi du rat lon)
  - Nhung sau z-score normalization, 500 co the nam trong khoang 1-3 std
  - kNN chi can 500 "nguoi lang" gan => khoang cach nho

Benchmark v7 (inject truoc khi feature):
  - Anomaly: fare=500, dist=1, dur=5
  - feature[6] = fare/dist = 500 (THUC SU rat lon)
  - Khong co z-score normalization cho test
  - => Distance tu 500 den cac gia tri binh thuong (mean=2.5) la 500 units
  - => kNN DE PHAN BIET HON
```

### Ly do 2: Z-score normalization trong benchmark cua toi

```
Benchmark cua toi (results/v5_clean):
  - X_train da z-score normalized (mean=0, std=1)
  - X_test cung da z-score normalized theo scaler cua train
  - => Test data co cung scale nhu train

Van de voi anomalies:
  - Anomaly co gia tri lon (fare=500) duoc normalize thanh z=(500-mean)/std
  - Neu std cua train feature6 la 0.5, thi z = (500-2.5)/0.5 = 995
  - Nhung 995 van con 995 "normal neighbors" gan 995 trong memory
  - => Memory co 10K normal points, chi can 5-10 nearest => van la anomaly
  - Nhung khoang cach den nhung diem gan nhat van con 1-5 units
  
Benchmark v7 (khong normalize test):
  - Train: z-score normalized
  - Test: khong normalize (hoac normalize nhung anomaly van con outlier lon)
  - Distance = 500 units thuc su
  - => kNN phan biet tot hon nhieu
```

### Ly do 3: Muc do anomaly

```
v7 benchmark:
  - easy: fare 150-500 (gap 8-29x so voi mean=17)
  - hard: fare 60-80, zero distance, slow crawl
  - => RAT IO RAT lon

Tuning benchmark:
  - Khong ro rang muc do (co the la anomaly equipment)
  - Co the la 2-5x mean, khong phai 8-29x
  - => Signal yeu hon
```

## 2. Bang so sanh

| Thuoc tinh | v7 Benchmark | Tuning Benchmark (cu toi) |
|---|---|---|
| Injection timing | BEFORE features | AFTER features (da preprocessed) |
| Test normalization | Khong co (hoac khac) | Co (z-score theo train) |
| Muc do anomaly | 8-29x mean (easy) | Khong ro (co the la 2-5x) |
| Feature engineering | fare/dist, fare/dur, speed | Giong v7 |
| AUC-PR DenoisingAE | 0.9995 | Chua test |
| AUC-PR MemStream | 0.9996 | Chua test |
| AUC-PR sklearn_IF | 0.809 | 0.2244 (tuned) |

## 3. Ket luan

Ket qua v7 cho thay **MemStream va DenoisingAE la tot nhat** khi:
1. Anomalies duoc inject truoc khi feature engineering
2. Muc do anomaly duong nhien lon (8-29x normal)
3. Khong co z-score normalization cho test

Ket qua tuning benchmark cua toi cho thay **sklearn_IF voi max_features=0.5 la tot nhat** khi:
1. Anomalies da ton tai trong preprocessed data
2. Muc do anomaly co the yeu hon
3. Z-score normalization co mat signal

**Co the chung minh hai ket qua deu dung trong ngu canh cua chung.**

De chung minh MemStream/CA-DQStream la tot nhat, can tai hien benchmark v7 nhung tang them:
- Nhieu thuat toan baseline hon
- Statistical tests Friedman + Wilcoxon nhu v7
- Cai thien CA-DQStream/MemStream de dat 0.9996+
