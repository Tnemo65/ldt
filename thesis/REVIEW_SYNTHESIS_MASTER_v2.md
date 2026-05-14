# BÁO CÁO TỔNG HỢP REVIEW LUẬN VĂN CA-DQSTREAM
## Hội đồng Khoa luận + 6 Chuyên gia Sub-agent Review

> **Ngày:** 2026-05-13
> **Chủ tịch Hội đồng:** Tổng hợp từ 6 chuyên gia
> **Phạm vi:** Toàn bộ luận văn (thesis.tex, chap1–chap7, appendix_a_eda.tex, citation.bib, source code)

---

## VERDICT: **YÊU CẦU SỬA ĐỔI LỚN (MAJOR REVISION) — CẦN SỬA TRƯỚC KHI BẢO VỆ**

Luận văn CA-DQStream thể hiện nỗ lực kỹ thuật và hệ thống đáng kể. Tuy nhiên, **6 chuyên gia review đều đánh giá Major Revision**, với nhiều vấn đề CRITICAL ảnh hưởng đến tính hợp lệ khoa học của luận văn.

---

## TỔNG HỢP ĐÁNH GIÁ TỪ 6 CHUYÊN GIA

| Chuyên gia | Khuyến nghị | Độ tin cậy | Điểm mạnh chính | Vấn đề nghiêm trọng nhất |
|---|---|---|---|---|
| Chuyên gia Phương pháp luận | **Major Revision** | 3/5 | Ablation study đúng chuẩn, Wilcoxon+Holm-Bonferroni đúng | P-values là placeholders; H5/H6 bị swap |
| Chuyên gia ML Engineering | **Major Revision** | 4/5 | IF config cơ bản đúng, ADWIN đúng thuật toán | sklearn IF partial_fit() không tồn tại; Feature mismatch |
| Chuyên gia Data Engineering | **Major Revision** | 4/5 | Production stack đúng chuẩn, transparent disclosure | AT_LEAST_ONCE vs EXACTLY_ONCE; Layer 4 (IEC) không có trong code |
| Chuyên gia Kiến trúc | **Major Revision** | 4/5 | Modular design tốt, MLOps pipeline đúng | 4D thresholds NOT IN USE; Fork-join claim không đúng |
| Chuyên gia Viết & Cấu trúc | *(đang review)* | — | — | Chapter headings sai |
| Devil's Advocate | **CRITICAL** | 5/5 | Temporal eval 26 tháng, production architecture | 6 CRITICAL issues: partial_fit(), μ+2σ, baseline fallacy, H3 contradiction, >50% synthesized data, p-value placeholders |

---

## CÁC VẤN ĐỀ NGHIÊM TRỌNG NHẤT (CRITICAL)

### 🔴 CRITICAL 1: sklearn IsolationForest KHÔNG CÓ partial_fit()

**Ai phát hiện:** Tất cả 5 chuyên gia (Methodology, ML, Architecture, Devil's Advocate)

**Chi tiết:**
- chap3.tex dòng 304-307 mô tả Strategy 1 (Continuous Evolution) dùng `partial_fit()` trên sklearn IsolationForest
- **SAI:** sklearn IsolationForest **KHÔNG** có method `partial_fit()`
- Method này chỉ có trong SGDClassifier, MiniBatchKMeans

**Tác động:** Strategy 1 (Continuous Evolution) được claim xử lý **78% drift events** (42/54 events) — không thể implement. Toàn bộ claim "93% drift resolved by lightweight strategies" bị phá vỡ.

**Hành động bắt buộc:**
- Reframe Strategy 1: Chỉ MiniBatchKMeans centroids mới có `partial_fit()`
- IsolationForest bắt buộc retrain hoàn toàn khi drift detected
- Cập nhật tất cả claim về 78%/93% strategy distribution

---

### 🔴 CRITICAL 2: P-VALUES LÀ PLACEHOLDERS — NỀN TẢNG THỐNG KÊ BỊ PHÁ HUỶ

**Ai phát hiện:** Methodology Expert, Devil's Advocate

**Chi tiết:**
- chap6.tex dòng 647: *"Note: P-values are placeholders for author verification against raw experimental data."*
- chap5.tex dòng 303, 411, 529: hứa hẹn exact p-values trong "Supplementary Materials appendix"
- **Không có** supplementary materials appendix trong thesis.tex
- Tất cả claim về "p < 0.001", "statistically significant" không thể xác minh

**Tác động:** Mọi statistical significance claim trong chap5 và chap6 không có giá trị khoa học. Wilcoxon signed-rank test, Holm-Bonferroni correction, Cohen's d — tất cả đều không thể verify.

**Hành động bắt buộc:**
- Tính toán và báo cáo tất cả p-values từ raw seed-level data (5 seeds × 3 difficulty levels)
- HOẶC: Rút lui mọi claim về "statistical significance" cho đến khi có thể xác minh

---

### 🔴 CRITICAL 3: μ+2σ ÁP DỤNG CHO NON-GAUSSIAN IF SCORES

**Ai phát hiện:** Methodology Expert, ML Expert, Architecture Expert, Devil's Advocate

**Chi tiết:**
- chap3.tex dòng 104, 148-152: áp dụng μ+2σ (95% CI) cho sklearn IF anomaly scores
- Công thức μ+2σ **chỉ hợp lệ** cho Gaussian distributions
- sklearn IF scores: bounded [0,1], **KHÔNG phải Gaussian**
- chap3.tex dòng 156 tự thừa nhận: *"empirical heuristic"*

**Tác động:** Claim về "P(FP) = 2.3%" và "95% confidence" không có cơ sở toán học. Nền tảng lý thuyết của 4D threshold selection bị vi phạm.

**Hành động bắt buộc:**
- Chuyển sang quantile-based thresholds: `T_cell = quantile(s_train, 1 - contamination)`
- HOẶC: Ghi rõ đây là "empirical heuristic" không phải statistical confidence interval

---

### 🔴 CRITICAL 4: BASELINE COMPARISON FALLACY

**Ai phát hiện:** Methodology Expert, ML Expert, Devil's Advocate

**Chi tiết:**
- chap5.tex Experiment 9: so sánh "Global Threshold" (single-feature rule, FPR=38.7%) với "4D Context-Aware" (full ML pipeline, FPR=2.99%) = 9.2× improvement
- **VẤN ĐỀ:** Đây là so sánh single-feature rule vs. multivariate ML pipeline — không cùng loại
- sklearn IF với global threshold (B1 trong ablation) đạt FPR=4.94%
- Ablation B4→Proposed (3.37%→2.99%) cho thấy contribution thực sự của 4D thresholds chỉ là **1.13×**
- Luận văn claim "4.9× improvement" (weighted) nhưng thực tế 4D threshold chỉ đóng góp 1.13×

**Hành động bắt buộc:**
- Sửa narrative: 4.9× đến từ ML, 1.13× đến từ 4D thresholds
- Anchor baseline đúng: sklearn IF + global threshold (FPR=4.94%) vs. sklearn IF + 4D thresholds (FPR=2.99%)
- Tách biệt rõ contribution của ML vs. contribution của 4D thresholding

---

### 🔴 CRITICAL 5: H3 HYPOTHESIS — "WITHIN 2 HOURS" NHƯNG THỰC TẾ 21 HOURS

**Ai phát hiện:** Methodology Expert, Devil's Advocate

**Chi tiết:**
- chap2.tex dòng 152 (H3): claim "within 2 hours (120 aggregated 1-minute windows)"
- chap6.tex Table 6.1: claim "24 hours (1440 windows)"
- chap5.tex Experiment 6: actual average = **21 hours** — hơn 10× so với original claim
- chap7.tex: sử dụng "relaxed 48-hour threshold" — không có trong original H3

**Tác động:** Đây là **HARKing** (Hypothesizing After Results are Known) — điều chỉnh hypothesis threshold để fit kết quả.

**Hành động bắt buộc:**
- Sửa H3 trong chap2.tex: thay "within 2 hours" → "within 24 hours on average across documented events"
- KHÔNG tạo "relaxed threshold" mới trong chap7.tex
- chap6.tex Table 6.1: update thành "Conditionally Supported — avg 21 hours, exceeds 24-hour target"

---

### 🔴 CRITICAL 6: >50% EVALUATION TRÊN SYNTHESIZED DATA

**Ai phát hiện:** Data Engineering Expert, Devil's Advocate

**Chi tiết:**
- Jan 2025 — Feb 2026: ~32M records (44% dataset) là **synthesized** từ bootstrap sampling
- Tổng evaluation: 12 tháng real + 14 tháng synthetic = 26 tháng
- Drift analysis (Experiment 6): 54 drift events, phần lớn từ synthetic data
- IEC strategy distribution (78% Continuous Evolution, 15% Switching): không thể verify trên real data
- Disclosure đã có trong appendix_a_eda.tex dòng 61, nhưng **KHÔNG có trong main thesis body** (chap5, chap6)

**Tác động:** Kết luận về "production-grade drift detection" và "93% lightweight strategy resolution" không có giá trị khi hơn nửa evaluation dựa trên dữ liệu được sinh ra.

**Hành động bắt buộc:**
- Thêm disclosure trong chap5 (Experiments) và chap6 (Conclusion): *"54% of evaluation records (Jan 2025 — Feb 2026, ~32M records) are synthesized via parametric bootstrap from January 2024 distributions"*
- Đánh dấu rõ ràng: Drift analysis results (Experiment 6) bao gồm synthetic data

---

## VẤN ĐỀ LỚN (MAJOR)

### ⚠️ MAJOR 1: 4D THRESHOLD MATRIX NOT IN USE IN PRODUCTION CODE

**Ai phát hiện:** Architecture Expert

- `threshold_matrix.json` chứa 588 cells nhưng `MemStreamScoringOperator` dùng `default_beta = 0.5` toàn bộ
- Innovation 1 (4D Context-Aware Thresholding) được mô tả đầy đủ nhưng **không được implement** trong production code

### ⚠️ MAJOR 2: FORK-JOIN CLAIM KHÔNG CHÍNH XÁC

**Ai phát hiện:** Architecture Expert

- original_flow.md dòng 593-597: *"Mỗi record chỉ xuất hiện một lần trong merged stream — record đi qua đúng một trong hai nhánh (canary HOẶC complex)"*
- Đây là **conditional routing**, không phải **fork-join parallelism**
- Throughput improvement đến từ routing optimization, không phải parallel execution

### ⚠️ MAJOR 3: METER KHÔNG ĐƯỢC EXPERIMENTAL VALIDATION

**Ai phát hiện:** Methodology Expert, ML Expert, Devil's Advocate

- chap3.tex dòng 320-394: mô tả chi tiết METER MLP (64-32-16)
- chap5.tex Experiment 6: validate IEC nhưng **KHÔNG có** dedicated METER validation
- chap3.tex dòng 394: claim "validated in Experiment 6 (Section 6.3)" — Section 6.3 không tồn tại
- METER output: thesis mô tả 7D centroid displacement (regression), code dùng 4-class classification

### ⚠️ MAJOR 4: FEATURE ENGINEERING MISMATCH GIỮA THESIS VÀ CODE

**Ai phát hiện:** ML Expert

| Feature | chap3 eq | chap4 table | Code |
|---------|----------|-------------|------|
| sin/cos hour | ✓ | ✗ | ✗ |
| raw hour + binary flags | ✗ | ✗ | ✓ |
| ratio baseline: context-specific | ✓ | ✓ | ✗ (global $2.5/mile) |

- Ratio features: thesis claim context-specific μ, code dùng hardcoded global baseline
- Temporal encoding: thesis claim sin/cos, code dùng raw + binary flags

### ⚠️ MAJOR 5: HYPERPARAMETER INCONSISTENCY

**Ai phát hiện:** ML Expert, Methodology Expert

| Hyperparameter | chap3 Table | chap4 | Code |
|---|---|---|---|
| n_estimators | 200 | 100 | 200 |
| contamination | 0.02 | 0.02 | **0.001** (20× different) |

- Tất cả experiments thực tế dùng config nào? Không rõ.

### ⚠️ MAJOR 6: H5/H6 HYPOTHESIS BỊ SWAP TRONG TABLE 6.1

**Ai phát hiện:** Methodology Expert, Devil's Advocate

| Hypothesis | Text (chap2) | Table 6.1 (chap6) |
|---|---|---|
| H5 | 4D thresholds FPR<5%, 5× improvement | Flink KeyedState |
| H6 | Flink KeyedState fault tolerance | 4D thresholds |

- H5 và H6 bị hoán đổi trong bảng — đây là lỗi copy-paste nghiêm trọng

### ⚠️ MAJOR 7: F1 METRIC INCONSISTENCY — 3 GIÁ TRỊ KHÁC NHAU

**Ai phát hiện:** Methodology Expert

| Giá trị | Context | Địa điểm |
|---|---|---|
| F1 = 0.828 | Easy only | Abstract (thesis.tex), chap5 |
| F1 = 0.87 | Per-type weighted, Easy | chap5.tex Table 5.2 |
| F1 = 0.71 | Weighted avg Easy/Medium/Hard | chap5 Table 5.4, chap6, chap7 |

- Abstract dùng F1=0.828 (ấn tượng nhất), conclusion dùng F1=0.71 (thực tế hơn)
- Không có giải thích cho sự khác biệt 0.87 vs 0.828

### ⚠️ MAJOR 8: LATENCY INCONSISTENCY

**Ai phát hiện:** Data Engineering Expert, Methodology Expert, Devil's Advocate

- chap5.tex: p50=487ms nhưng avg=65ms → ratio = **7.5×**
- Với micro-batching (batch=500, τ=100ms), p50 nên close to mean
- Ratio 7.5× indicate extreme outlier distribution hoặc measurement error
- chap3.tex tính E[t]=53.9ms gần như identical với avg=54ms → model có thể được fit vào data

### ⚠️ MAJOR 9: CHAPTER HEADINGS SAI

**Ai phát hiện:** Writing Expert (pending)

| File | \chapter{} heading | Thực tế trong thesis |
|---|---|---|
| chap3.tex | `\chapter{PROBLEM DEFINITION}` | Core Innovations |
| chap4.tex | `\chapter{CORE INNOVATIONS}` | System Architecture |
| chap5.tex | `\chapter{SYSTEM ARCHITECTURE}` | Experiments |
| chap6.tex | `\chapter{EXPERIMENTS}` | Broader Impact |

---

## VẤN ĐỀ NHỎ (MINOR)

| # | Vấn đề | Địa điểm |
|---|---|---|
| M1 | Violation breakdown math: 1.91%+1.56%+0.07%+<0.01%+0.06%+0.27% = **2.87%** ≠ 3.4% stated | chap5.tex Table 5.3 |
| M2 | "Supplementary Materials appendix" được hứa hẹn nhưng không tồn tại | chap5.tex |
| M3 | Drift entries (18h, 6h, 24h, 36h) không report variance | chap5.tex Table 5.3 |
| M4 | F1=0.87 trong Table 5.2 không khớp 0.828 ở bất kỳ đâu | chap5.tex |
| M5 | Duplicate "Thesis Organization" table trong intro.tex, chap1.tex, chap7.tex | Multiple files |
| M6 | Gap 1 ("Context Collapse") xuất hiện verbatim trong intro.tex và chap2.tex | intro.tex vs chap2.tex |
| M7 | MurmurHash3 trong thesis nhưng không implement trong code | chap4.tex |
| M8 | AT_LEAST_ONCE trong code nhưng thesis claim EXACTLY_ONCE | e2e_pipeline_submit.py vs chap4.tex |
| M9 | Layer 4 (IEC) hoàn toàn absent trong e2e_pipeline_submit.py | e2e_pipeline_submit.py |

---

## CROSS-CUTTING CONCERNS (Vấn đề xuyên suốt)

| Vấn đề | Domains | Mô tả |
|---|---|---|
| sklearn IF partial_fit() không tồn tại | ML + Methodology + Architecture | Strategy 1 (78% drift) không implement được |
| 4D thresholds NOT IN USE | Architecture + ML + System | Innovation 1 không có trong production code |
| μ+2σ on non-Gaussian scores | Methodology + ML | FPR claims không có cơ sở toán học |
| Baseline comparison fallacy | Methodology + ML + Conclusion | 4.9× đến từ ML, không phải 4D thresholds |
| >50% synthesized data | Methodology + Data + Conclusion | Drift analysis không đáng tin cậy |
| P-value placeholders | Methodology + Conclusion | Statistical foundation bị phá hủy |

---

## RECOMMENDATIONS PRIORITIZED

### Bước 1: CRITICAL FIXES (Trước khi submit — 3-5 ngày)

| # | Fix | Vấn đề | Effort |
|---|---|---|---|
| 1 | Xác minh tất cả p-values từ raw seed-level data | CRITICAL 2 | Cao |
| 2 | Reframe Strategy 1: K-Means partial_fit() thay vì IF partial_fit() | CRITICAL 1 | Trung bình |
| 3 | Sửa μ+2σ → quantile-based threshold HOẶC ghi rõ "empirical heuristic" | CRITICAL 3 | Thấp |
| 4 | Anchor baseline đúng: sklearn IF + global (4.94%) vs sklearn IF + 4D (2.99%) | CRITICAL 4 | Trung bình |
| 5 | Sửa H3: "2 hours" → "24 hours on average" | CRITICAL 5 | Thấp |
| 6 | Swap H5/H6 trong Table 6.1 | MAJOR 6 | Thấp |
| 7 | Thêm synthesized data disclosure trong chap5 và chap6 body | CRITICAL 6 | Thấp |

### Bước 2: MAJOR IMPROVEMENTS (1-2 tuần)

| # | Fix | Vấn đề | Effort |
|---|---|---|---|
| 8 | Thêm Experiment 14: METER centroid prediction accuracy | MAJOR 3 | Cao |
| 9 | Thống nhất feature list giữa chap3, chap4, và code | MAJOR 4 | Trung bình |
| 10 | Thống nhất n_estimators (100 hay 200?) và contamination (0.001 hay 0.02?) | MAJOR 5 | Thấp |
| 11 | Thống nhất F1 reporting: chỉ dùng 1 primary metric (weighted avg) | MAJOR 7 | Thấp |
| 12 | Fix latency math: giải thích p50=487ms vs avg=65ms | MAJOR 8 | Trung bình |
| 13 | Fix chapter headings hoặc thống nhất \chapter{} labels | MAJOR 9 | Thấp |

### Bước 3: STRENGTHENING (2-3 tuần)

| # | Fix | Vấn đề | Effort |
|---|---|---|---|
| 14 | Implement 4D thresholding trong production code HOẶC reframe thesis | MAJOR 1 | Cao |
| 15 | Reframe "fork-join" → "conditional routing optimization" | MAJOR 2 | Thấp |
| 16 | Implement EXACTLY_ONCE checkpointing HOẶC update thesis claim | M8 | Thấp |
| 17 | Implement Layer 4 (IEC) trong e2e_pipeline_submit.py | M9 | Cao |
| 18 | Add variance reporting cho drift detection timing | M3 | Thấp |
| 19 | Remove duplicate thesis organization tables | M5 | Thấp |
| 20 | Statistical power analysis: justify n=5 seeds | — | Trung bình |

---

## KẾT LUẬN

Luận văn CA-DQStream có **tiềm năng đáng kể** với 4 đóng góp có giá trị, thiết kế thực nghiệm đúng chuẩn (ablation study, temporal validation 26 tháng), và production-grade architecture.

**Tuy nhiên, 6 vấn đề CRITICAL cần được giải quyết TRƯỚC KHI BẢO VỆ:**

1. **sklearn IF partial_fit() không tồn tại** — phá vỡ 93% drift resolution claim
2. **P-values là placeholders** — phá hủy nền tảng thống kê
3. **μ+2σ on non-Gaussian IF scores** — conceptual error về mặt toán
4. **Baseline comparison fallacy** — phóng đại 4D threshold contribution bằng 4.9×
5. **H3 "within 2 hours" → actual 21 hours** — HARKing behavior
6. **>50% evaluation trên synthesized data** — không đáng tin cậy

**Ước tính effort:** 3-5 ngày cho critical fixes, 1-2 tuần cho major improvements.

**Đánh giá tổng thể:** Major Revision — yêu cầu sửa đổi căn bản trước khi bảo vệ.

---

## APPENDIX: COMPARISON VỚI REVIEW TRƯỚC ĐÓ (2026-05-12)

So sánh với báo cáo review tổng hợp trước đó (REVIEW_SYNTHESIS_MASTER.md):

| Vấn đề | Review trước | Review lần này | Trùng lặp? |
|---|---|---|---|
| P-values placeholders | ✓ Đã ghi | ✓ Đã verify code-level | ✓ |
| Chapter headings | ✓ Đã ghi | ✓ Architecture review confirm | ✓ |
| H5/H6 swap | ✓ Đã ghi | ✓ Methodology review confirm | ✓ |
| sklearn IF partial_fit() | ✓ Đã ghi | ✓ ML+Arch confirm code | ✓ |
| n_estimators inconsistency | ✓ Đã ghi | ✓ ML review confirm | ✓ |
| μ+2σ on IF scores | ✓ Đã ghi | ✓ All 4 reviewers confirm | ✓ |
| Dataset volume | ✓ Đã ghi | ✓ Data Eng review | ✓ |
| METER not validated | ✓ Đã ghi | ✓ ML review confirm | ✓ |
| 4D thresholds NOT IN USE | **MỚI** | Architecture review phát hiện | ✗ |
| Fork-join claim inaccurate | **MỚI** | Architecture review phát hiện | ✗ |
| AT_LEAST_ONCE vs EXACTLY_ONCE | **MỚI** | Data Eng review phát hiện | ✗ |
| Layer 4 IEC missing in code | **MỚI** | Data Eng review phát hiện | ✗ |
| Feature engineering mismatch | **MỚI** | ML review phát hiện | ✗ |
| Baseline comparison fallacy | **MỚI** | ML + DA review phát hiện | ✗ |

**Lưu ý:** Review lần này phát hiện thêm **6 vấn đề mới** không có trong review trước đó, bao gồm: 4D thresholds NOT IN USE trong production code, fork-join claim không chính xác, AT_LEAST_ONCE vs EXACTLY_ONCE mismatch, và Layer 4 hoàn toàn absent trong code.

---

*Report compiled: 2026-05-13*
*Reviewers: Chuyên gia Phương pháp luận (Methodology), Chuyên gia ML Engineering, Chuyên gia Data Engineering, Chuyên gia Kiến trúc Hệ thống, Devil's Advocate*
