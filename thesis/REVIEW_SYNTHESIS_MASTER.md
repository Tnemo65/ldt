# BÁO CÁO TỔNG HỢP REVIEW LUẬN VĂN CA-DQSTREAM

> **Ngày:** 12/05/2026  
> **Người review:** Chủ tịch Hội đồng Khoa luận + 6 Chuyên gia Sub-agent  
> **Phạm vi:** Toàn bộ luận văn (thesis.tex, chap1–chap7, appendix_a_eda.tex, citation.bib)

---

## TỔNG QUAN

### VERDICT: **CẦN SỬA ĐỔI TRƯỚC KHI BẢO VỆ**

Luận văn thể hiện nỗ lực kỹ thuật đáng kể và đóng góp nghiên cứu có ý nghĩa trong lĩnh vực giám sát chất lượng dữ liệu streaming. Tuy nhiên, có **nhiều vấn đề nghiêm trọng** ảnh hưởng đến tính hợp lệ của luận văn:

| Lĩnh vực | Đánh giá | Mức độ |
|---|---|---|
| Cấu trúc & Logic | ⚠️ Cần sửa | Có thể sửa |
| Phương pháp nghiên cứu | ⚠️ Cần làm rõ | Nghiêm trọng |
| Machine Learning | 🔴 Có lỗi kỹ thuật | Nghiêm trọng |
| Kiến trúc hệ thống | ⚠️ Cần bổ sung | Khá nghiêm trọng |
| Data Engineering | ⚠️ Cần xác minh | Nghiêm trọng |
| Tính đóng góp & Novelty | ⚠️ Cần làm rõ | Trung bình |

---

## ĐIỂM MẠNH

### 1. Đóng góp nghiên cứu rõ ràng
- 4 đóng góp thực sự (4D thresholding, IEC multi-strategy, Hybrid model, Fork-join pipeline) được trình bày có hệ thống và trace được đến RQ tương ứng
- Cấu trúc nghiên cứu theo mô hình: Motivation → Background → Problem → Solution → Evaluation → Conclusion là chuẩn mực

### 2. Thực nghiệm toàn diện
- 72M bản ghi trong 26 tháng là nỗ lực đánh giá ấn tượng
- Ablation study (B1→B2→B3→B4→Proposed) đúng chuẩn phương pháp luận
- Statistical testing (Wilcoxon, Holm-Bonferroni) được áp dụng đúng quy trình

### 3. Hệ thống production-grade
- Tích hợp Kafka + Flink + PostgreSQL + Prometheus + Grafana + MLflow là thiết kế có tính thực tiễn cao
- PgBouncer, Broadcast State, checkpointing đều là best practices trong production

### 4. Data quality domain knowledge
- Phân biệt 5 loại vấn đề DQ (Completeness, Validity, Multivariate, Uniqueness, Drift) là phân loại có cơ sở lý thuyết vững
- Ultra-clean baseline concept là giải pháp sáng tạo cho GIGO problem

### 5. IEC multi-strategy design
- Phân biệt 4 chiến lược drift adaptation (Continuous Evolution → Switching → METER → Spatial Tracking) là thiết kế tinh vi và có giá trị thực tiễn
- ADWIN-based drift detection trên quality meta-streams là hướng tiếp cận đúng đắn

### 6. Feature engineering có cơ sở
- Cyclical encoding (sin/cos) cho time features là đúng về toán học
- Ratio features (fare_per_mile_ratio, etc.) có rationale rõ ràng về mặt domain

---

## VẤN ĐỀ NGHIÊM TRỌNG (Cần sửa trước khi bảo vệ)

### 🔴 VẤN ĐỀ 1: P-VALUE ĐƯỢC THỪA NHẬN LÀ "PLACEHOLDER"

**Địa điểm:** chap6.tex, dòng 647
> *"Note: P-values are placeholders for author verification against raw experimental data."*

**Mô tả:** Toàn bộ các claim về statistical significance (p < 0.05) được dựa trên p-value chưa được xác minh. Chap6 trình bày kết quả nghiên cứu với "p < 0.001" nhưng đồng thời thừa nhận đây chỉ là placeholder.

**Tác động:** Đây là vấn đề phương pháp luận nghiêm trọng nhất. Không có p-value đã xác minh, toàn bộ statistical validity của luận văn không thể khẳng định.

**Hành động bắt buộc:** Xác minh tất cả p-value từ raw data, hoặc rút lui tất cả claim về statistical significance cho đến khi có thể xác minh.

---

### 🔴 VẤN ĐỀ 2: MISMATCH GIỮA BẢNG HYPOTHESIS VÀ BODY TEXT (H5)

**Địa điểm:** chap6.tex (Table 6.1) vs chap5.tex (dòng 466)

**Mô tả:** H5 trong Table 6.1 (chap6.tex, dòng 95) mô tả "Flink KeyedState deduplication processes high-volume streams with manageable state footprint" nhưng body text (chap5.tex, dòng 466) định nghĩa H5 là "4D context-aware thresholds will reduce FPR by at least 5×."

**Tác động:** Đây là internal inconsistency nghiêm trọng. Một reviewer sẽ ngay lập tức nhận ra H5 trong bảng và H5 trong text mô tả hai claim khác nhau.

**Hành động bắt buộc:** Xác định rõ H5 thực sự address claim nào và thống nhất xuyên suốt.

---

### 🔴 VẤN ĐỀ 3: CHƯƠNG BỊ ĐÁNH SAI (chap1.tex → chap6.tex)

**Địa điểm:** Cả 4 file (chap1, chap3, chap5, chap6)

| File | \chapter{} heading | Thứ tự import | Mục đích thực |
|---|---|---|---|
| chap1.tex | `\chapter{INTRODUCTION}` ❌ | Chapter 1 | Background |
| chap2.tex | `\chapter{PROBLEM DEFINITION}` ✓ | Chapter 2 | Problem Definition |
| chap3.tex | `\chapter{PROBLEM DEFINITION}` ❌ | Chapter 3 | Core Innovations |
| chap4.tex | `\chapter{CORE INNOVATIONS}` ❌ | Chapter 4 | System Architecture |
| chap5.tex | `\chapter{SYSTEM ARCHITECTURE}` ❌ | Chapter 5 | Experiments |
| chap6.tex | `\chapter{EXPERIMENTS}` ❌ | Chapter 6 | Broader Impact |
| chap7.tex | `\chapter{CONCLUSION}` ✓ | Chapter 7 | Conclusion |

**Mô tả:**
- chap1.tex: Đánh dấu `\chapter{INTRODUCTION}` nhưng chứa nội dung Background (Kafka, Flink, MLOps, DAMA-DMBOK, Related Works)
- chap3.tex: Đánh dấu `\chapter{PROBLEM DEFINITION}` nhưng thesis.tex import nó như Chapter 3 = Core Innovations
- chap4.tex: Đánh dấu `\chapter{CORE INNOVATIONS}` nhưng thesis.tex import nó như Chapter 4 = System Architecture
- chap5.tex: Đánh dấu `\chapter{SYSTEM ARCHITECTURE}` nhưng thesis.tex import nó như Chapter 5 = Experiments
- chap6.tex: Đánh dấu `\chapter{EXPERIMENTS}` nhưng thesis.tex import nó như Chapter 6

**Tác động:** Đây là vấn đề quản lý bản thảo nghiêm trọng. Khi reviewer đọc thesis.tex theo import order, các chapter numbers trong LaTeX heading không khớp. Đặc biệt, chương "Core Innovations" (đóng góp chính) bị đánh số Chapter 3 (nên là 4), và "System Architecture" bị đánh số Chapter 4 (nên là 5).

**Hành động bắt buộc:**
1. Sửa \chapter{} headings trong chap3.tex → `\chapter{CORE INNOVATIONS}`
2. Sửa \chapter{} headings trong chap4.tex → `\chapter{SYSTEM ARCHITECTURE \& IMPLEMENTATION}`
3. Sửa \chapter{} headings trong chap5.tex → `\chapter{EXPERIMENTS \& EVALUATION}`
4. Sửa \chapter{} headings trong chap6.tex → `\chapter{BROADER IMPACT \& CONCLUSION}`
5. XÓA hoặc sửa chap1.tex → `\chapter{BACKGROUND}` (hiện đang là INTRODUCTION)

---

### 🔴 VẤN ĐỀ 4: NỘI DUNG TRÙNG LẶP (3 lần bảng Thesis Organization)

**Địa điểm:** intro.tex (dòng 64-72) ≈ chap1.tex (dòng 62-72) ≈ chap7.tex (dòng 62-72)

**Mô tả:** Cùng một bảng mô tả cấu trúc 7 chương xuất hiện gần như IDENTICAL tại 3 vị trí:
1. intro.tex → sau Introduction
2. chap1.tex → cuối Background chapter  
3. chap7.tex → trong Conclusion chapter

**Tác động:**
- Tự đạo văn (self-plagiarism) — cùng nội dung 3 lần
- Lãng phí không gian (3 bảng ≈ 1 trang)
- Dấu hiệu poor editing

**Hành động bắt buộc:** Giữ bảng này ĐÚNG 1 LẦN trong intro.tex (sau phần Introduction), XÓA khỏi chap1.tex và chap7.tex.

---

### 🔴 VẤN ĐỀ 5: SKLEARN ISOLATIONFOREST.KMEANS() + PARTIAL_FIT() KHÔNG TỒN TẠI

**Địa điểm:** chap3.tex, dòng 304-306 (Strategy 1: Continuous Evolution)

**Mô tả:** Luận văn claim:
> *"Incremental updates via `partial_fit()` on recent data windows without full retraining"*

**Sai:**
- sklearn IsolationForest **KHÔNG hỗ trợ** `partial_fit()`. Đây là phương thức chỉ tồn tại trong SGDClassifier, MiniBatchKMeans, và một số estimator khác.
- IsolationForest.fit() yêu cầu toàn bộ dataset để xây dựng optimal isolation trees qua recursive random partitioning — không có streaming/incremental variant trong sklearn.

**Tác động:** Strategy 1 (Continuous Evolution, được claim là xử lý 78% drift events) **KHÔNG THỂ implement** với sklearn IsolationForest như mô tả. Đây là lỗi thuật toán fundamental.

**Hành động bắt buộc:**
- Reframe Strategy 1: Chỉ K-Means centroids mới có thể update incrementally (via `partial_fit()`). IsolationForest phải retrain hoàn toàn khi cần.
- HOẶC: Chuyển sang River's `StreamingIsolationForest` nếu thực sự cần incremental IF.

---

### 🔴 VẤN ĐỀ 6: INCONSISTENT N_ESTIMATORS (100 vs 200)

**Địa điểm:** chap3.tex (Table config) vs chap4.tex (dòng 210)

**Mô tả:**

| Location | n_estimators claim |
|---|---|
| chap3.tex, Table (Sec 4.5) | 200 trees |
| chap4.tex, Cold-Start Training (Sec 5.1.3) | 100 trees |

**Tác động:** Mô hình thực tế dùng trong tất cả experiments là **100-tree version**. Table trong chap3 mô tả **200-tree variant** chưa bao giờ được train hoặc evaluate. Mọi training time, inference latency, và F1/FPR metrics đều từ model 100-tree.

**Hành động bắt buộc:** Thống nhất n_estimators=100 xuyên suốt (matching actual training). Cập nhật table trong chap3 cho khớp.

---

### 🔴 VẤN ĐỀ 7: μ+2σ THRESHOLD APPLIED TO ANOMALY SCORES, NOT FEATURES

**Địa điểm:** chap3.tex, Sec 4.2 và chap4.tex, Sec 4.5

**Mô tả:** Luận văn áp dụng công thức μ+2σ cho sklearn IsolationForest anomaly scores:

```math
T_{\text{cell}} = \mu_{\text{cell}} + 2\sigma_{\text{cell}} \quad \text{(applied to anomaly score } s\text{)}
```

**Vấn đề:**
1. sklearn anomaly scores bounded [0, 1] — KHÔNG phải Gaussian distributed
2. Công thức μ+2σ chỉ hợp lệ cho phân phối xấp xỉ Gaussian
3. Luận văn tự claim "P(FP) = 2.3%" từ normal distribution nhưng đang áp dụng cho bounded anomaly scores

**Tác động:** Đây là conceptual error về mặt thống kê. Confidence interval claim (2.3% FPR) không có cơ sở toán học khi áp dụng cho IF scores.

**Hành động bắt buộc:**
- (a) Áp dụng μ+2σ cho normalized raw features, rồi kết hợp với IF scores
- (b) Sử dụng quantile-based thresholds: `T_cell = quantile(s_train, 1 - contamination)`
- (c) Statement rõ ràng: đây là empirical heuristic, KHÔNG phải statistical confidence interval

---

### 🔴 VẤN ĐỀ 8: DATASET VOLUME KHÔNG NHẤT QUÁN

**Địa điểm:** Appendix A vs chap5.tex

| Source | Claim | Calculation |
|---|---|---|
| Appendix A, Figure caption | 41.2M / 12 months | 3.43M/month |
| chap5.tex | 72M / 26 months | 2.77M/month |
| Appendix A | Jan 2024: 2,964,624 records | Exact |
| Appendix A | July 2024: 3,076,903 records | Exact |

**Mô tả:** 
- 41.2M / 12 = 3.43M/month → không khớp 72M / 26 = 2.77M/month
- Nếu 72M records là 26 tháng (Jan 2024 – Feb 2026), thì TLC Yellow Taxi data không thể có data đến Feb 2026 (tính đến 5/2026, chỉ có đến khoảng early 2025)
- Không có bảng chi tiết monthly record counts

**Tác động:** 
1. Nếu Jan 2025 – Feb 2026 là simulated data, luận văn phải disclose
2. Luận văn không có cross-validation table để xác minh dataset size

**Hành động bắt buộc:** 
- Xác minh actual dataset composition (real vs. simulated)
- Tạo single data inventory table (source, time range, record count)
- Disclosure rõ ràng nếu data sau Jan 2025 là synthetic

---

## VẤN ĐỀ LỚN (Nên sửa trước khi bảo vệ)

### ⚠️ VẤN ĐỀ 9: MISMATCH GIỮA BASELINE SO SÁNH

**Địa điểm:** chap5.tex, Sec 5.7 (Experiment 9)

**Mô tả:** Luận văn so sánh:
- "Global Threshold" (μ+2σ on raw fare): FPR = 38.7% → single-rule method
- sklearn IsolationForest (raw features): FPR = 4.94% → ML method

**Vấn đề:** Những baseline này KHÔNG so sánh cùng loại. Global threshold là single-feature rule. sklearn IF là multivariate ML. So sánh 4.9× improvement (38.7% → 4.2%) thực chất đang so sánh single-rule vs. ML pipeline.

**Đúng hơn:** Nên so sánh sklearn IF với global threshold (FPR = 4.94%) vs. sklearn IF với 4D thresholds (FPR = 2.99%). Ablation table B4→Proposed chỉ cho thấy 1.13× FPR improvement từ 4D thresholds (3.37% → 2.99%).

---

### ⚠️ VẤN ĐỀ 10: METEOR HYPERNETWORK KHÔNG ĐƯỢC EXPERIMENTAL VALIDATE

**Địa điểm:** chap3.tex, Sec 4.4 (Strategy 3: METER Adaptation)

**Mô tả:** METER MLP hypernetwork (64-32-16 architecture) được mô tả chi tiết nhưng KHÔNG xuất hiện trong chap6 (Experiments). Không có Experiment nào validate METER centroid predictions.

**Vấn đề:** 1 trong 4 adaptation strategies của IEC không được experimental validation.

**Hành động bắt buộc:**
- Thêm Experiment 14: METER centroid prediction accuracy
- HOẶC: Đơn giản hóa METER thành "centroid shifting based on meta-metrics" (không cần MLP details)

---

### ⚠️ VẤN ĐỀ 11: METRIC INCONSISTENCY (F1: 0.71 vs 0.828 vs 0.87)

**Địa điểm:** Nhiều location trong chap5.tex, chap6.tex

**Mô tả:** Luận văn báo cáo 3 F1 scores khác nhau:

| Source | F1 | Context |
|---|---|---|
| Abstract | 0.828 | Easy difficulty only |
| Exp 3 table | 0.87 | Per-type weighted (Easy only) |
| Exp 4 ablation | 0.71 | Weighted avg across Easy/Medium/Hard |
| chap7 Conclusion | 0.71 | Weighted avg across Easy/Medium/Hard |

**Vấn đề:** 0.87 (weighted per-type, Easy only) được trình bày gần với 0.828 (per-type Easy) — dễ gây confusion. 0.71 (weighted avg across difficulties) được dùng trong conclusion nhưng 0.828 trong abstract tạo ấn tượng kết quả tốt hơn thực tế.

---

### ⚠️ VẤN ĐỀ 12: LATENCY VS THROUGHPUT NUMBERS KHÔNG NHẤT QUÁN

**Địa điểm:** chap5.tex, Sec 5.6 (Exp 7)

| Metric | Linear | Rendezvous | Claim |
|---|---|---|---|
| Avg Latency | 65ms | 54ms | 1.20× |
| p50 Latency | 487ms | 168ms | 2.9× |
| Throughput | 8,240/s | 18,450/s | 2.2× |

**Vấn đề:**
1. p50 = 487ms NHƯNG avg = 65ms → p50/avg ratio = 7.5× → extreme outlier skew không realistic
2. Throughput improvement 2.2× quá gần với PgBouncer improvement 2.3× (Exp 12) → không rõ improvement đến từ Rendezvous hay từ PgBouncer
3. Latency math không internal consistent: 13.5% bypass không thể giải thích 65ms→54ms reduction

---

### ⚠️ VẤN ĐỀ 13: GAP 1 (Context Collapse) TRÙNG LẶP

**Địa điểm:** intro.tex (dòng 13-19) ≈ chap2.tex (dòng 29-43)

**Mô tả:** Gap 1 "Context Collapse in Heterogeneous Data" xuất hiện gần như verbatim tại cả intro.tex và chap2.tex.

**Tác động:** Tự đạo văn nội bộ; tạo fatigue cho reviewer.

---

### ⚠️ VẤN ĐỀ 14: H3 DRIFT DETECTION THRESHOLD KHÔNG KHỚP

**Địa điểm:** chap2.tex (H3) vs chap5.tex (Exp 6) vs chap7.tex (H3 validation)

| Source | Drift Detection Claim |
|---|---|
| chap2.tex (H3) | "within 2 hours (120 aggregated 1-minute windows)" |
| chap6.tex (Exp 6) | "Average time-to-detect = 21 hours" |
| chap7.tex (H3) | "avg 21 hours across 54 drift events, well within threshold" |

**Vấn đề:** H3 claim "within 2 hours" nhưng actual average là 21 hours — hơn 10× so với claim. chap7.tex sử dụng "48-hour threshold" (không có trong original H3) để biện minh cho 21 hours.

---

### ⚠️ VẤN ĐỀ 15: ĐỘI SỐ LƯỢNG FEATURE (15D vs 21D) KHÔNG MATCH

**Địa điểm:** chap3.tex (Equation) vs chap4.tex (Table 2)

**Mô tả:** Equation trong chap3 chỉ enumerate **14 features** nhưng claim "15D base + 6D ratio = 21D total." Table trong chap4 (Feature Table) list 15 base features.

**Vấn đề:** 
- `total_amount` xuất hiện trong chap3 equation nhưng KHÔNG xuất hiện trong chap4 table
- `trip_distance_log` xuất hiện trong chap4 table nhưng KHÔNG xuất hiện trong chap3 equation
- Luận văn không align được danh sách feature giữa 2 chương

---

## VẤN ĐỀ NHỎ (Nên sửa)

| # | Vấn đề | Địa điểm | Tác động |
|---|---|---|---|
| M1 | H1 trong bảng (chap7.tex) khác với H1 trong text (chap2.tex) | chap7.tex vs chap2.tex | Confusion |
| M2 | chap7.tex có 3 subsections H3a-d không được định nghĩa trong chap2.tex | chap2.tex vs chap7.tex | Missing notation |
| M3 | chap2.tex thiếu sections 3.3, 3.4, 3.5 (jump từ 3.2 → 3.6) | chap2.tex | Incomplete outline |
| M4 | chap5.tex Table 5.3: Drift events breakdown = "avg 21 hours" nhưng table entries show 18h, 6h, 24h, 36h → average = 21h nhưng không có variance | chap5.tex | Cherry-picked entries |
| M5 | chap4.tex đề cập "MurmurHash3 for deduplication" nhưng KHÔNG implement trong code | chap4.tex | Unverified claim |
| M6 | Terminology: "sklearn IsolationForest" vs "sklearn's Isolation Forest" vs "IF" lẫn lộn | Throughout | Inconsistency |
| M7 | chap5.tex experiment 5, 9, 11, 12, 13 chỉ là single-paragraph summary, KHÔNG có full method/results | chap5.tex | Incomplete experiments |
| M8 | chap5.tex (Table 5.3) violation breakdown: 1.91%+1.56%+0.07%+<0.01%+0.06%+0.27% = 2.87% ≠ 3.4% stated | chap5.tex | Math error |
| M9 | chap6.tex, chap5.tex: Delta_score implementation không match thesis formula | chap4.tex vs code | Wrong formula |
| M10 | chap1.tex, chap3.tex, chap7.tex: Acronym table có both "IF" và "IOF" = "Isolation Forest" → redundant | thesis.tex | Table error |

---

## CROSS-CUTTING CONCERNS

### Issue spanning nhiều domains

| Issue | Domains | Description |
|---|---|---|
| P-values as placeholders | Methodology + Conclusion | Statistical foundation undermined |
| H3 threshold mismatch | Methodology + Conclusion | Hypothesis vs. actual results conflict |
| Inconsistent F1 metrics | Methodology + ML + Conclusion | Multiple "primary" metrics |
| 4D threshold theoretical basis | ML + Methodology | μ+2σ applied to non-Gaussian scores |
| Dataset volume reconciliation | Data + Methodology | Three incompatible volume figures |
| Mock ML vs real ML | Architecture + Methodology | Performance claims unverifiable |
| METER not validated | ML + Methodology | Strategy without experiment |

---

## RECOMMENDATIONS PRIORITIZED

### Bước 1: CRITICAL FIXES (Trước khi submit)

1. **Xác minh tất cả p-value** từ raw experimental data. Đây là non-negotiable.
2. **Fix chapter headings**: chap1→BACKGROUND, chap3→CORE INNOVATIONS, chap5→EXPERIMENTS, chap6→BROADER IMPACT
3. **Remove duplicate Thesis Organization table** — giữ 1 bảng trong intro.tex
4. **Fix H5 hypothesis mismatch** trong bảng vs text
5. **Fix n_estimators = 100** xuyên suốt hoặc = 200 xuyên suốt (matching actual training)
6. **Clarify μ+2σ threshold** — được áp dụng cho features hay IF scores?
7. **Reconcile dataset volume** — tạo single data inventory table
8. **Fix Strategy 1 (Continuous Evolution)** — sklearn IF không có partial_fit()

### Bước 2: MAJOR IMPROVEMENTS

9. **Thêm METER validation experiment** hoặc simplify METER description
10. **Consistent F1 reporting** — thống nhất 1 primary metric, rõ ràng về difficulty level
11. **Fix baseline comparison** — so sánh sklearn IF + global threshold vs sklearn IF + 4D thresholds
12. **Fix drift detection H3** — update claim từ "within 2 hours" → actual 21 hours average
13. **Fix 15D/21D feature list** — enumerate single consistent list xuyên suốt
14. **Add full experiments 5, 9, 11, 12, 13** hoặc giảm claim từ "13 experiments" → "7 fully-validated experiments"
15. **Fix latency/throughput inconsistency** — p50=487ms vs avg=65ms contradiction

### Bước 3: STRENGTHENING

16. **Compare with River library** (streaming ML) hoặc remove MemStream/MStream from comparison
17. **Add monthly volume table** trong Appendix A để cross-validate 72M claim
18. **Fix violation rate math** — breakdown sum (2.87%) ≠ stated (3.4%)
19. **Report variance** cho drift detection timing
20. **Add statistical power analysis** — justify n=5 seeds hoặc tăng lên n≥10

---

## KẾT LUẬN

Luận văn CA-DQStream thể hiện **nỗ lực kỹ thuật đáng kể** với 4 đóng góp có giá trị (4D thresholding, IEC multi-strategy, Hybrid model, Fork-join pipeline). Điểm mạnh rõ ràng: temporal evaluation 26 tháng, ablation study tốt, production-grade system design.

Tuy nhiên, **8 vấn đề nghiêm trọng** cần được giải quyết TRƯỚC KHI BẢO VỆ:

1. P-values placeholders → cần xác minh
2. Chapter headings bị đánh sai → cần fix
3. Duplicate thesis organization → cần xóa 2 bảng
4. H5 hypothesis mismatch → cần thống nhất
5. sklearn IF partial_fit() không tồn tại → cần refactor
6. n_estimators inconsistency → cần thống nhất 100
7. μ+2σ applied to non-Gaussian scores → cần clarify
8. Dataset volume không nhất quán → cần reconcile

**Ước tính effort:** 3-5 ngày làm việc để fix tất cả critical issues. Các major issues cần thêm 1-2 tuần.

---

*Review compiled: 2026-05-12*
*Reviewers: Thesis Structure Expert, Methodology Expert, ML Expert, System Architecture Expert, Data Engineering Expert, Devil's Advocate Expert*
