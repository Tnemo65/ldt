# COMPREHENSIVE THESIS REVIEW REPORT
## CA-DQStream: A Context-Aware Framework for Streaming Data Quality Monitoring
### Author: Le Dac Thinh | Supervisor: Associate Professor Nguyen Ngoc Hoa
### Review Date: May 10, 2026

---

> **Reviewer Perspective:** Chủ tịch Hội đồng Khóa luận & Chuyên gia Documentation
> **Review Methodology:** Multi-agent parallel analysis using 3 specialized subagents (structure/review, citation/LaTeX audit, peer-review/methodology)

---

# TỔNG HỢP ĐÁNH GIÁ (EXECUTIVE SUMMARY)

## Overall Assessment: **MINOR REVISIONS** (Cần chỉnh sửa nhỏ trước khi bảo vệ)

Thesis CA-DQStream thể hiện độ sâu kỹ thuật ấn tượng và phạm vi thí nghiệm toàn diện. Các đóng góp khoa học cốt lõi (4D thresholds, Rendezvous pipeline, IEC, Hybrid model) được trình bày rõ ràng và đánh giá qua 10 thí nghiệm chính. Tuy nhiên, có một số vấn đề nghiêm trọng về labeling, đánh số, và tính nhất quán cần được sửa trước khi bảo vệ.

---

# PHẦN 1: CÁC VẤN ĐỀ NGHIÊM TRỌNG (CRITICAL ISSUES)

---

## CRITICAL-001: Sai Ngày/Số Lượng Dataset trong Cold-Start

| Thuộc tính | Giá trị |
|---|---|
| **Severity** | **CRITICAL — Lỗi thực tế** |
| **Location** | `chap4.tex`, dòng 84 |
| **Files Affected** | `chap4.tex`, `chap2.tex` |

### Evidence

**chap2.tex** (dòng 39–44) định nghĩa chính xác các dataset splits:
```
Training (Baseline): January 2024, 2,964,624 records
Primary Test: July 2024, 3,076,903 records
```

**chap4.tex** (dòng 84) tuyên bố:
> "CA-DQStream solves this via a **3-Stage Sequential Funnel** applied *offline* to the training dataset (January 2024, 3.08M raw records)"

**chap4.tex** Table (dòng 125–131) hiển thị:
```
Raw data:                    3,076,903    ---
After physical filter:       2,997,418    2.58%
Ultra-clean baseline:       2,969,106    3.48% total
```

### Vấn đề

| Property | chap4.tex (text) | Correct (chap2.tex) |
|---|---|---|
| Month | January 2024 | January 2024 ✓ |
| Record count | **3.08M (3,076,903)** | **2,964,624** |

Table header "Raw data: 3,076,903" khớp với **July 2024** (Primary Test), KHÔNG PHẢI January 2024 (Training).

**Đây là lỗi thực tế nghiêm trọng**: Text nói January 2024 nhưng Table cho thấy July 2024 data (3,076,903 records). Cold-Start pipeline được mô tả là dùng training dataset nhưng thực tế dùng test dataset.

### Recommendation

**Ưu tiên cao nhất — Phải fix trước khi bảo vệ:**

Tùy chọn A: Regenerate table từ January 2024 data (2,964,624 records)
- Training set filter: 2,964,624 → ~2,880,000 → ~2,860,000
- Text và Table đồng nhất về January 2024

Tùy chọn B: Đổi text thành "July 2024" — nhưng điều này phá vỡ consistency với chap2.tex

**Khuyến nghị: Tùy chọn A** — Regenerate table từ January 2024 data.

---

## CRITICAL-002: Author Field Chứa Title (Broken Citation)

| Thuộc tính | Giá trị |
|---|---|
| **Severity** | **CRITICAL — Build/Citation Error** |
| **Location** | `citation.bib`, dòng 401–409 |
| **Entry Key** | `contextdq2022` |

### Evidence

```bib
@article{contextdq2022,
  author = {{Context in Data Quality Management: A Systematic Literature Review}},
  title = {Context in Data Quality Management: A Systematic Literature Review},
  year = {2022},
  eprint = {2204.10655},
}
```

### Vấn đề

Trường `author` chứa title thay vì tên tác giả thực. Citation trong reference list sẽ hiển thị sai.

### Recommendation

Tìm author names thực của arXiv:2204.10655 và fix:

```bib
@article{contextdq2022,
  author = {Author1 Name and Author2 Name and Author3 Name},
  title = {Context in Data Quality Management: A Systematic Literature Review},
  year = {2022},
  eprint = {2204.10655},
  archiveprefix = {arXiv},
  primaryclass = {cs.DB},
}
```

---

# PHẦN 2: CÁC VẤN ĐỀ CHÍNH (MAJOR ISSUES)

---

## MAJOR-001: Đánh Số Experiment — RQ3 Section Thiếu Experiment 5

| Thuộc tính | Giá trị |
|---|---|
| **Severity** | MAJOR |
| **Location** | `chap5.tex` |
| **Files Affected** | `chap5.tex`, `thesis.tex`, `chap2.tex` |

### Evidence

**chap5.tex** định nghĩa experiments theo thứ tự:

| Location | Experiment | Section |
|---|---|---|
| Dòng 52 | Experiment 1 | RQ1 |
| Dòng 91 | Experiment 2 | RQ1 |
| Dòng 147 | Experiment 3 | RQ2 |
| Dòng 184 | Experiment 4 | RQ2 |
| **Dòng 272** | **Experiment 6** (nhảy từ 4 lên 6!) | **RQ3** |
| Dòng 324 | Experiment 7 | RQ4 |
| Dòng 361 | Experiment 8 | RQ4 |
| Dòng 407 | Experiment 9 | RQ5 |
| Dòng 455 | Experiment 10 | RQ5 |
| **Dòng 496** | **Experiment 5** (Business Rules) | **Additional Validation** |

**chap2.tex** định nghĩa: `RQ3: Does IEC Multi-Strategy Adaptation... [Exp 5, 6]`

### Vấn đề

1. **RQ3 section header tuyên bố `[Exp 5, 6]` nhưng chỉ có Experiment 6** — Experiment 5 không có trong section này
2. **Experiment 5 (Business Rules) bị đặt sai chỗ** — nó nên thuộc RQ1 (Layer 2a), không phải "Additional Validation"
3. Không có experiment nào validate "ADWIN Drift Detection" một cách độc lập — Experiment 6 cover toàn bộ IEC với 4 strategies

### Recommendation

```
Option A: Renumber
- Move Experiment 5 vào RQ1 section (trước Experiment 2)
- Experiment 6 (ADWIN) giữ nguyên
- Sửa header RQ3 thành [Exp 6]

Option B: Relabel
- Giữ nguyên numbering
- Sửa RQ3 header thành [Exp 6] (xóa Exp 5)
- Thêm footnote: Experiment 5 (Business Rules) xem Section 5.8
```

---

## MAJOR-002: Section Numbering Off-By-One (4 Chapters)

| Thuộc tính | Giá trị |
|---|---|
| **Severity** | MAJOR |
| **Location** | Tất cả chap*.tex files |
| **Files Affected** | `chap1.tex`, `chap3.tex`, `chap4.tex`, `chap5.tex`, `chap6.tex` |

### Evidence

| File | `\chapter{}` label | Actual Chapter # (TOC) | Internal Sections | Should Be |
|---|---|---|---|---|
| chap1.tex | Background & Related Works | Chapter 2 | 1.1–1.6 | **2.1–2.6** |
| chap2.tex | Problem Definition | Chapter 3 | 3.1–3.6 | **3.1–3.6 ✓** |
| chap3.tex | Core Innovations | Chapter 4 | 3.1–3.6 | **4.1–4.6** |
| chap4.tex | System Architecture | Chapter 5 | 4.1–4.6 | **5.1–5.6** |
| chap5.tex | Experiments & Evaluation | Chapter 6 | 5.1–5.9 | **6.1–6.9** |
| chap6.tex | Conclusion & Future Work | Chapter 7 | 6.1–6.6 | **7.1–7.6** |

### Root Cause

chap*.tex được viết với section numbers cứng (1.x, 3.x, 4.x...) như thể mỗi file bắt đầu chapter numbering từ 1. Nhưng khi assembled trong thesis.tex qua `\input{}`, chapter counter tiếp tục tăng, nên internal section prefixes không khớp.

**chap2.tex là chapter DUY NHẤT có internal numbering đúng** (sections 3.1–3.6 vì nó là Chapter 3).

### Recommendation

Chọn một trong hai approaches:

**Approach 1 (Recommended):** Thêm `\chapter{}` vào đầu mỗi chap*.tex và reset section numbering:
```latex
% chap3.tex
\chapter{Core Innovations}
\counterwithin{section}{chapter}  % Reset section to chapter number
```

**Approach 2:** Giữ nguyên nhưng update tất cả `\section{}` prefix để match actual chapter numbers.

---

## MAJOR-003: Section 1.6 (Thesis Organization) Off-By-One

| Thuộc tính | Giá trị |
|---|---|
| **Severity** | MAJOR |
| **Location** | `thesis.tex`, dòng 269–278 |
| **Files Affected** | `thesis.tex` |

### Evidence

**Section 1.6 tuyên bố:**
> "Chapter 1 covers background and related works"
> "Chapter 3 details the CA-DQStream framework design"
> "Chapter 4 presents experiments and evaluation results"
> "Chapter 5 concludes with contributions and future directions"

**TOC xác nhận cấu trúc 7-chapter:**
- Chapter 1: INTRODUCTION (inline)
- Chapter 2: BACKGROUND (chap1.tex)
- Chapter 3: PROBLEM DEFINITION (chap2.tex)
- Chapter 4: CORE INNOVATIONS (chap3.tex)
- Chapter 5: SYSTEM ARCHITECTURE (chap4.tex)
- Chapter 6: EXPERIMENTS (chap5.tex)
- Chapter 7: CONCLUSION (chap6.tex)

### Mismatches

| Section 1.6 | Actual (TOC) |
|---|---|
| "Chapter 1: Background" | **Chapter 2: Background** |
| "Chapter 3: Core Innovations" | **Chapter 4: Core Innovations** |
| "Chapter 4: Experiments" | **Chapter 6: Experiments** |
| "Chapter 5: Conclusion" | **Chapter 7: Conclusion** |

### Recommendation

Fix Section 1.6 để match TOC:
```latex
\section{Thesis Organization}
...
\begin{itemize}
    \item \textbf{Chapter 1: Introduction}
    \item \textbf{Chapter 2: Background and Related Works} % was "Chapter 1"
    \item \textbf{Chapter 3: Problem Definition and Research Questions}
    \item \textbf{Chapter 4: Core Innovations} % was "Chapter 3"
    \item \textbf{Chapter 5: System Architecture and Implementation}
    \item \textbf{Chapter 6: Experiments and Evaluation} % was "Chapter 4"
    \item \textbf{Chapter 7: Conclusion and Future Work} % was "Chapter 5"
\end{itemize}
```

---

## MAJOR-004: chap4.tex Chapter Summary — Sai Chapter Number

| Thuộc tính | Giá trị |
|---|---|
| **Severity** | MAJOR |
| **Location** | `chap4.tex`, dòng 1047 |

### Evidence
> "The next chapter (Chapter 5) presents experimental validation..."

chap4.tex = **Chapter 5** (System Architecture). "Next chapter" = chap5.tex = **Chapter 6** (Experiments), KHÔNG PHẢI Chapter 5.

### Recommendation
Đổi thành "(Chapter 6)".

---

## MAJOR-005: chap3.tex Innovation 2 — 13.5% Early Exit Rate Không Khớp

| Thuộc tính | Giá trị |
|---|---|
| **Severity** | MAJOR |
| **Location** | `chap3.tex` vs `chap2.tex` vs `chap5.tex` |

### Evidence

**chap3.tex Innovation 2 (dòng 145):**
> "This design enables: **Early exit optimization:** 13.5% of records flagged by Canary bypass Complex entirely"

**chap2.tex Layer 2 (Business Rules) Results (dòng 199–208):**
```
Negative fare:    1.91%
Zero distance:     1.56%
Speed >100mph:    0.07%
Dropoff<=Pickup:  <0.01%
Fare out bounds:  0.06%
Payment mismatch: 0.27%
Total:           ~3.87%
```

**chap2.tex cho Layer 1+2 (dòng 151):**
> "Layers 1 and 2 collectively cover approximately 97% of all violations"
> "This data quality summary motivates... Layer 2 handles... ~3.4% hard rule violations"

### Vấn đề

chap3.tex sử dụng 13.5% nhưng chap2.tex chỉ báo cáo ~3.4% violations cho Business Rules (Layer 2). 13.5% ≈ 13.5% × 3.08M ≈ 416,000 records. Nhưng 3.4% × 3.08M ≈ 105,000 records. Sự khác biệt là **4 lần**.

**Đâu là nguồn gốc 13.5%?** Có thể là:
- Tổng Layer 1 (9.1%) + Layer 2 (3.4%) = 12.5% ≈ 13.5%?
- Hay đây là kết quả từ experiment (chap5.tex Experiment 7)?

### Recommendation

Làm rõ nguồn gốc của 13.5%. Nếu đây là assumption, ghi chú rõ: "Estimated based on chap2.tex Layer 2 statistics." Nếu đây là measured result từ Experiment 7, cần reference forward citation.

---

## MAJOR-006: chap6.tex H6 — Content Không Khớp với chap2.tex

| Thuộc tính | Giá trị |
|---|---|
| **Severity** | MAJOR |
| **Location** | `chap2.tex` vs `chap6.tex` |

### Evidence

**chap2.tex H6:**
> "4D context-aware thresholds achieve FPR below 5% compared to 25--50% for global thresholds (at least 5$\times$ improvement)"

**chap6.tex H6:**
> "CA-DQStream maintains accuracy over 26 months with IEC recalibration"

chap6.tex H6 mô tả temporal stability, không phải FPR reduction. Đây là **content mismatch trong cùng một hypothesis**.

### Recommendation

Đổi chap6.tex H6 description để match chap2.tex:
> "4D context-aware thresholds achieve FPR below 5%, reducing false positive rate by at least 5$\times$ compared to global thresholds"

---

## MAJOR-007: 3 Giá Trị F1 Khác Nhau cho Cùng Một Model

| Thuộc tính | Giá trị |
|---|---|
| **Severity** | MAJOR |
| **Location** | `chap3.tex`, `chap5.tex`, `chap6.tex` |

### Evidence

| Source | F1 Value | Experiment |
|---|---|---|
| chap3.tex Chapter Summary | F1=0.87 | Không rõ nguồn |
| chap5.tex Experiment 3 | F1=0.87 (weighted average) | 50K standard anomalies |
| chap5.tex Experiment 4 | **F1=0.91** (proposed) | 50K EXTREME anomalies |
| chap6.tex Conclusion | F1=0.87 | Stated in closing |

### Vấn đề

1. chap3.tex (Core Innovations) — viết TRƯỚC experiments — sử dụng F1=0.87 như thể đó là kết quả đã biết. Tạo reference circular: innovation được mô tả với kết quả thí nghiệm TRƯỚC KHI thí nghiệm được thực hiện.
2. chap6.tex Conclusion sử dụng F1=0.87 thay vì F1=0.91 — đánh giá thấp hơn kết quả tốt nhất.
3. Experiment 3 (F1=0.87) vs Experiment 4 (F1=0.91) — "EXTREME" anomalies produce HIGHER F1? Điều này ngược intuitition (harder test nên lower F1).

### Recommendation

1. Update chap6.tex Conclusion → F1=0.91 (từ Experiment 4)
2. Thêm note trong chap3.tex Chapter Summary: "F1=0.87 from initial validation; see Chapter 5 for full results (F1=0.91)"
3. Giải thích tại sao "EXTREME" anomalies produce F1=0.91 > F1=0.87 (standard anomalies)

---

## MAJOR-008: Baseline 4 vs Proposed — Cùng Components?

| Thuộc tính | Giá trị |
|---|---|
| **Severity** | MAJOR |
| **Location** | `chap5.tex` Experiment 4 |

### Evidence

chap5.tex định nghĩa 5 baselines:

```
Baseline 1: 15D raw + global threshold
Baseline 2: 21D ratio + global threshold  
Baseline 3: 21D ratio + K-Means clustering
Baseline 4: 21D ratio + K-Means + 4D thresholds
Proposed:   21D ratio + K-Means + 4D thresholds
```

**Baseline 4 và Proposed có CÙNG components** (21D + K-Means + 4D thresholds). Chỉ khác tên gọi và marginal 0.01 F1 difference.

### Recommendation

1. Làm rõ sự khác biệt thực sự giữa Baseline 4 và Proposed
2. Nếu không có sự khác biệt, merge thành một row
3. Nếu có sự khác biệt (threshold calibration, feature weighting...), đặt tên cụ thể

---

## MAJOR-009: Nhiều So Sánh Không Có Statistical Correction

| Thuộc tính | Giá trị |
|---|---|
| **Severity** | MAJOR |
| **Location** | `chap5.tex` Experiment 4 |

### Evidence

Experiment 4 so sánh 5 baselines × 4 metrics = **20 simultaneous comparisons** mà không có correction cho multiple testing.

### Recommendation

1. Apply Benjamini-Hochberg FDR correction
2. Report t-statistic, degrees of freedom, confidence intervals
3. Consider Wilcoxon signed-rank test cho non-normal metrics

---

## MAJOR-010: Ground Truth Chỉ Dựa Trên Synthetic Anomalies

| Thuộc tính | Giá trị |
|---|---|
| **Severity** | MAJOR |
| **Location** | `chap5.tex` |

### Evidence

chap5.tex line 26: "Ground Truth: Synthetic anomalies injected into clean data"

Không có human expert labeling study cho production anomalies thực tế.

### Recommendation

1. Add paragraph về limitation này
2. Consider small-scale expert labeling
3. Add cross-validation (train Jan–Jun 2024, test Jul 2024)
4. Provide citations cho "taxi data analysis literature" claims

---

# PHẦN 3: CÁC VẤN ĐỀ NHỎ (MINOR ISSUES)

---

## MINOR-001: Duplicate `grab_engineering2024` vs `grab_streaming_dq`

| Severity | Location | Description |
|---|---|---|
| MINOR | `citation.bib:421,610` | Cùng nguồn Grab Engineering blog |

**Fix:** Consolidate into `grab_streaming_dq`, delete `grab_engineering2024`.

---

## MINOR-002: Author Key Có Non-ASCII Character

| Severity | Location | Description |
|---|---|---|
| MINOR | `citation.bib:372` | `s察ter2018automating` — Chinese character trong key |

**Fix:** Rename to `schelter2018automating`.

---

## MINOR-003: 49 Orphan Bibliography Entries

| Severity | Location | Description |
|---|---|---|
| MINOR | `citation.bib` | 49 entries không được cite trong text |

49 entries tăng bibliography size mà không contribute. Đa số là tool documentation (`mysql2024docs`, `flink_docs`, etc.) và survey papers không liên quan.

**Fix:** Remove all 49 orphan entries hoặc thêm citations vào text.

---

## MINOR-004: `@article` Nhưng Có `booktitle` (Nên Là `@inproceedings`)

| Severity | Location | Description |
|---|---|---|
| MINOR | `citation.bib` (5 entries) | `bifet2007learning`, `liu2008isolation`, `breunig2000lof`, `s察ter2018automating`, `scholkopf2001oc` |

**Fix:** Change `@article` → `@inproceedings`.

---

## MINOR-005: chap3.tex — Defensive Tone Trong Academic Writing

| Severity | Location | Description |
|---|---|---|
| MINOR | `chap4.tex:142` | "The term 'ultra-clean' is not marketing language" |

**Fix:** Remove defensive phrasing. Academic prose nên confident, không defensive.

---

## MINOR-006: chap4.tex — Comment Trong LaTeX Không Khớp Content

| Severity | Location | Description |
|---|---|---|
| MINOR | `chap4.tex:48` | `% 4.1 System Overview` nhưng content là Cold-Start |

**Fix:** Update comment thành `% 4.1.1 Cold-Start Model Training` hoặc tương tự.

---

## MINOR-007: chap3.tex Equation — Missing Space Trong Unit

| Severity | Location | Description |
|---|---|---|
| MINOR | `chap3.tex:242` | `53.9\text{ms (mean)}` → thiếu space |

**Fix:** `53.9\text{ ms (mean)}`

---

# PHẦN 4: BẢNG TÓM TẮT TẤT CẢ ISSUES

| ID | Category | Severity | Location | Description | Fix Priority |
|---|---|---|---|---|---|
| CRIT-001 | Factual | **CRITICAL** | chap4.tex:84 | Cold-Start: Jan 2024 nhưng table show Jul 2024 data | **Ngay** |
| CRIT-002 | Citation | **CRITICAL** | citation.bib:402 | `author` = title trong `contextdq2022` | **Ngay** |
| MAJ-001 | Numbering | MAJOR | chap5.tex | RQ3 section: thiếu Exp 5, Business Rules misplaced | Cao |
| MAJ-002 | Numbering | MAJOR | chap*.tex | Section numbering off-by-one in 4 chapters | Cao |
| MAJ-003 | Numbering | MAJOR | thesis.tex:269 | Section 1.6 off-by-one chapter labels | Cao |
| MAJ-004 | Numbering | MAJOR | chap4.tex:1047 | chap4=Chapter 5, "next chapter (Chapter 5)"→should be 6 | Cao |
| MAJ-005 | Factual | MAJOR | chap3.tex | 13.5% early exit rate không khớp với chap2.tex 3.4% | Cao |
| MAJ-006 | Consistency | MAJOR | chap2/6.tex | chap6.tex H6 khác content với chap2.tex H6 | Cao |
| MAJ-007 | Consistency | MAJOR | chap3/5/6.tex | 3 F1 values (0.87, 0.91, 0.87) cho cùng model | Cao |
| MAJ-008 | Comparison | MAJOR | chap5.tex:Exp4 | Baseline 4 vs Proposed: cùng components? | Cao |
| MAJ-009 | Statistics | MAJOR | chap5.tex:Exp4 | 20+ comparisons không có FDR correction | Cao |
| MAJ-010 | Methodology | MAJOR | chap5.tex | Chỉ synthetic anomalies, không có expert labeling | Cao |
| MIN-001 | Duplicate | MINOR | citation.bib | `grab_engineering2024` = `grab_streaming_dq` | Trung bình |
| MIN-002 | Typo | MINOR | citation.bib:372 | Non-ASCII char trong key `s察ter...` | Trung bình |
| MIN-003 | Orphan | MINOR | citation.bib | 49 orphan entries | Trung bình |
| MIN-004 | Type | MINOR | citation.bib (5) | `@article`→`@inproceedings` | Trung bình |
| MIN-005 | Writing | MINOR | chap4.tex:142 | Defensive tone: "not marketing language" | Thấp |
| MIN-006 | Comment | MINOR | chap4.tex:48 | LaTeX comment không khớp content | Thấp |
| MIN-007 | LaTeX | MINOR | chap3.tex:242 | Missing space trong `\text{ms (mean)}` | Thấp |

---

# PHẦN 5: FLOW DIAGRAM

```
thesis.tex: INTRODUCTION (Chapter 1, inline)
├── Section 1.1: Research Motivation
├── Section 1.2: Gaps in Current Research [Gap 1,2,3]
├── Section 1.3: Expected System [5 layers] ⚠ MAJ-003: chapter labels off-by-one
├── Section 1.4: Objectives
├── Section 1.5: Key Contributions
└── Section 1.6: Thesis Organization ⚠ MAJ-003: WRONG chapter numbers

    \input chap1 ────────────────────────────→ Chapter 2: BACKGROUND
    chap1.tex ⚠ MAJ-002: Sections 1.1–1.6 → should be 2.1–2.6
    ├── Streaming Data Processing (Kafka, Flink)
    ├── MLOps
    ├── DAMA-DMBOK
    └── Related Works
                                                       ↓
    \input chap2 ────────────────────────────→ Chapter 3: PROBLEM DEFINITION
    chap2.tex ✓ Sections 3.1–3.6 (only correct one)
    ├── Case Study: NYC Taxi Data
    ├── Research Gaps [Gap 1,2,3] ← ⚠ MAJ-006: chap6.tex H6 khác content
    ├── Research Questions [RQ1–RQ5]
    └── Research Hypotheses [H1–H6]
                                                       ↓
    \input chap3 ────────────────────────────→ Chapter 4: CORE INNOVATIONS
    chap3.tex ⚠ MAJ-002: Sections 3.1–3.6 → should be 4.1–4.6
    ├── Innovation 1: 4D Thresholding [RQ5]
    │   └── ⚠ MAJ-005: 13.5% early exit rate không khớp chap2.tex
    ├── Innovation 2: Rendezvous [RQ4]
    │   └── ⚠ MAJ-005: 13.5% early exit rate
    ├── Innovation 3: IEC [RQ3]
    └── Innovation 4: Hybrid K-Means [RQ2]
        └── ⚠ MAJ-007: F1=0.87 (trước experiment)
                                                       ↓
    \input chap4 ────────────────────────────→ Chapter 5: SYSTEM ARCHITECTURE
    chap4.tex ⚠ MAJ-002: Sections 4.1–4.6 → should be 5.1–5.6
    ├── System Overview
    ├── Cold-Start Model Training ⚠ CRIT-001: Jan 2024 text / Jul 2024 table!
    ├── Four-Layer Pipeline
    ├── Infrastructure
    ├── Optimization Techniques
    └── MLOps Integration ⚠ MAJ-004: "next chapter (Chapter 5)" → should be 6
                                                       ↓
    \input chap5 ────────────────────────────→ Chapter 6: EXPERIMENTS
    chap5.tex ⚠ MAJ-002: Sections 5.1–5.9 → should be 6.1–6.9
    ├── RQ1: Multi-Layer [Exp 1,2] ✓
    ├── RQ2: Hybrid Model [Exp 3,4]
    │   └── ⚠ MAJ-008: Baseline 4 = Proposed?
    ├── RQ3: IEC [Exp 5,6] ⚠ MAJ-001: Exp 5 MISSING from this section
    ├── RQ4: Rendezvous [Exp 7,8]
    ├── RQ5: 4D Thresholds [Exp 9,10]
    └── Additional Validation [Exp 5, 11,12,13] ⚠ MAJ-001: Exp 5 here (should be RQ1)
        └── ⚠ MAJ-009: Multiple comparisons without FDR correction
        └── ⚠ MAJ-010: Only synthetic anomalies
                                                       ↓
    \input chap6 ────────────────────────────→ Chapter 7: CONCLUSION
    chap6.tex ⚠ MAJ-002: Sections 6.1–6.6 → should be 7.1–7.6
    ├── Contributions Summary ⚠ MAJ-007: F1=0.87 → should be 0.91
    ├── Hypothesis Validation ⚠ MAJ-006: H6 content mismatch
    ├── Limitations
    ├── Future Work
    └── Broader Impact
```

---

# PHẦN 6: RECOMMENDATIONS THEO THỨ TỰ ƯU TIÊN

## Immediate (Trước khi nộp/bảo vệ)

1. **Fix CRIT-001** — Regenerate Cold-Start table từ January 2024 data (2,964,624 records)
2. **Fix CRIT-002** — Find actual author names cho `contextdq2022` và update author field

## High Priority (Trước khi bảo vệ)

3. **Fix MAJ-003** — Update Section 1.6 labels (Chapter 3→4, Chapter 4→6, Chapter 5→7)
4. **Fix MAJ-004** — chap4.tex Chapter Summary: "(Chapter 5)" → "(Chapter 6)"
5. **Fix MAJ-001** — Renumber/reorganize experiments: Move Exp 5 (Business Rules) vào RQ1, fix RQ3 section header
6. **Fix MAJ-002** — Add `\counterwithin{section}{chapter}` vào chap*.tex hoặc renumber sections
7. **Fix MAJ-007** — chap6.tex Conclusion: F1=0.87 → F1=0.91
8. **Fix MAJ-006** — chap6.tex H6: restore content từ chap2.tex

## Medium Priority (Sau khi bảo vệ, nếu có thời gian)

9. **Fix MAJ-005** — Explain source của 13.5% early exit rate
10. **Fix MAJ-009** — Add statistical correction cho multiple comparisons
11. **Fix MAJ-008** — Clarify Baseline 4 vs Proposed difference
12. **Fix MAJ-010** — Add cross-validation và discuss synthetic anomaly limitations
13. **Fix MIN-001, 002, 003, 004** — Cleanup bibliography

## Low Priority (Polish)

14. **Fix MIN-005** — Remove defensive tone
15. **Fix MIN-006** — Update LaTeX comment
16. **Fix MIN-007** — Add space trong unit

---

# PHẦN 7: POSITIVE FEEDBACK

Thesis có nhiều điểm mạnh đáng khen:

1. **Cấu trúc rõ ràng**: 7-chapter architecture với clear mapping từ RQ → Innovation → Experiment
2. **Độ sâu kỹ thuật**: Pipeline design (Rendezvous, fork-join), IEC multi-strategy, Hybrid model được trình bày chi tiết
3. **Thí nghiệm toàn diện**: 10 thí nghiệm chính + 3 additional, 72M records, 26 months
4. **Tài liệu đồ sộ**: 685 lines bib, 8 tables, 13 figures, 6 chapters
5. **Engineering quality**: LaTeX document class, natbib, hyperref, multi-column support
6. **Innovations well-motivated**: Mỗi innovation được ground trong real-world problems từ NYC Taxi dataset
7. **No orphan citations**: Tất cả citations được sử dụng (ngoại trừ 49 orphan entries trong bib)

---

*Report generated: May 10, 2026*
*Analysis performed by: Multi-agent parallel review (3 specialized subagents)*
*Total issues identified: 17 (2 Critical, 10 Major, 5 Minor)*
