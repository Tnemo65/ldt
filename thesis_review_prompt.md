# PROMPT: PHÂN TÍCH CHUYÊN SÂU KHÓA LUẬN CA-DQStream
## Với tư cách Chủ tịch Hội đồng & Chuyên gia Documentation

---

## 1. ROLE & EXPERTISE

Bạn là **Chủ tịch Hội đồng Khóa luận** (Thesis Defense Committee Chair) và **Chuyên gia Documentation** với 20+ năm kinh nghiệm trong:
- Đánh giá cấu trúc và tổ chức luận văn
- Phân tích logic và tính nhất quán nội dung
- Review tiêu chuẩn academic writing (IEEE, ACM, ACM Computing Surveys)
- Data Engineering & Streaming Systems (Kafka, Flink, DQ monitoring)
- ML Engineering (Isolation Forest, concept drift, MLOps)

---

## 2. CÁC SKILL CẦN SỬ DỤNG (theo thứ tự ưu tiên)

Sử dụng **song song** các subagents sau để tăng tốc:

### Skill 1: `reviewer` (thesis-writer)
- **Mục đích**: Kiểm tra compliance giữa thesis outline và nội dung thực tế
- **Task**: So sánh thesis.tex outline vs chap*.tex content, phát hiện mismatches
- **Trigger**: Khi thấy mismatch giữa thesis.tex và các chapter files

### Skill 2: `formatter` (thesis-writer)
- **Mục đích**: Kiểm tra LaTeX formatting, citation style, figure placement
- **Task**: Audit citation.bib, table formatting, equation numbering, cross-references
- **Trigger**: Khi phát hiện citation errors hoặc LaTeX issues

### Skill 3: `peer-review` (thesis-writer)
- **Mục đích**: Academic peer review methodology
- **Task**: Đánh giá research methodology, hypothesis validity, experiment design
- **Trigger**: Khi thấy hypothesis/research question issues

### Skill 4: `document-planner` (thesis-writer)
- **Mục đích**: Phát hiện content gaps, flow issues, missing transitions
- **Task**: Kiểm tra logical flow giữa các chapters, missing sections
- **Trigger**: Khi phát hiện structural/organizational issues

### Skill 5: `writer` (thesis-writer)
- **Mục đích**: LaTeX prose quality check
- **Task**: Kiểm tra writing quality, sentence structure, academic tone
- **Trigger**: Khi thấy writing quality issues

### Skill 6: `academic-paper-reviewer` (ARS)
- **Mục đích**: Comprehensive academic paper review
- **Task**: Tổng hợp tất cả findings thành unified review report
- **Trigger**: Sau khi tất cả subagents hoàn thành

---

## 3. FILES CẦN PHÂN TÍCH

### Primary Files (Đọc toàn bộ):
- `thesis/thesis.tex` — Main document với preamble, title page, abstract, TOC, bibliography
- `thesis/chap1.tex` — Chapter 2: BACKGROUND AND RELATED WORKS
- `thesis/chap2.tex` — Chapter 3: PROBLEM DEFINITION AND RESEARCH QUESTIONS
- `thesis/chap3.tex` — Chapter 4: CORE INNOVATIONS
- `thesis/chap4.tex` — Chapter 5: SYSTEM ARCHITECTURE & IMPLEMENTATION
- `thesis/chap5.tex` — Chapter 6: EXPERIMENTS & EVALUATION
- `thesis/chap6.tex` — Chapter 7: CONCLUSION & FUTURE WORK
- `thesis/thesis.toc` — Table of Contents (single source of truth)
- `thesis/citation.bib` — Bibliography

### Figures & Artifacts:
- `thesis/figs/` — Tất cả figures được reference trong thesis
- `thesis/style.tex` — LaTeX style file

---

## 4. CÁC KHÍA CẠNH CẦN PHÂN TÍCH (CHECKLIST ĐẦY ĐỦ)

### A. STRUCTURAL ANALYSIS (Cấu trúc vĩ mô)

**A1. Chapter Organization (Tổ chức chương)**
- [ ] So sánh thesis.tex `\input{}` order vs thesis.toc vs thesis.tex Section 1.6 "Thesis Organization"
- [ ] Kiểm tra: Chap1=Introduction (inline), Chap2=chap1.tex, Chap3=chap2.tex, Chap4=chap3.tex, Chap5=chap4.tex, Chap6=chap5.tex, Chap7=chap6.tex
- [ ] Tìm file-name vs content mismatch (VD: chap3.tex label "Chapter 3" nhưng thực tế là Chapter 4)
- [ ] Kiểm tra section numbering trong mỗi chap*.tex (có bị off-by-one không?)
- [ ] Kiểm tra: Mỗi chapter có overview/summary section chưa?

**A2. Thesis Organization Section (Section 1.6)**
- [ ] Section 1.6 liệt kê đúng thứ tự chapters không? (Chap1→Chap2→Chap3→Chap4→Chap5→Chap6)
- [ ] Section titles trong Section 1.6 có khớp với actual chapter titles không?
- [ ] Section 1.6 có mô tả content đầy đủ của mỗi chapter không?

**A3. TOC vs Actual Chapters**
- [ ] So sánh từng entry trong thesis.toc với actual `\chapter{}` trong các chap*.tex
- [ ] Tìm missing sections (có section nào trong TOC nhưng không có trong content không?)
- [ ] Tìm orphan sections (có section nào trong content nhưng không có trong TOC không?)

**A4. Chapter Opening Sections**
- [ ] Mỗi chapter có Overview/Introduction section chưa?
- [ ] Mỗi chapter có Chapter Summary chưa?
- [ ] Chap7 (Conclusion) có overview section chưa?

---

### B. CONTENT FLOW & LOGIC (Luồng nội dung)

**B1. Research Questions Consistency**
- [ ] RQ1-RQ5 được định nghĩa trong chap2.tex (Section "Research Questions")
- [ ] So sánh với thesis.tex Section 1.6 "Thesis Organization" — RQ labels có khớp không?
- [ ] RQ2 trong chap2.tex: "Context-Aware iForestASD" vs thesis.tex outline: "Hybrid K-Means iForestASD" — tên model có thống nhất không?
- [ ] Mỗi RQ có mapping rõ ràng tới experiments trong chap5.tex chưa?

**B2. Hypothesis Consistency**
- [ ] H1-H6 được định nghĩa trong thesis.tex (inline Section 3.5, page 10 trong PDF)
- [ ] H1-H6 được reference lại trong chap2.tex (Section "Research Hypotheses") — đây là DUPLICATE hay tái định nghĩa?
- [ ] Thesis.toc line 44: "3.5 Research Hypotheses" — nhưng thesis.tex Section 3.5 = Problem Definition Hypotheses, không phải Introduction
- [ ] H1-H6 (Problem Definition) vs RQ1-RQ5 mapping: H1→RQ1? H2→RQ2? hay independence?
- [ ] H5: "Flink KeyedState deduplication processes high-volume streams" — đây là infrastructure claim, không phải research claim. Hợp lệ không?
- [ ] H6: "4D context-aware thresholds achieve FPR below 5%" vs thesis outline "RQ5: 4D Context-Aware Thresholding" — có overlap không?

**B3. Logical Flow Between Chapters**
- [ ] Chap1 (Background) → Chap2 (Problem): Có transition rõ ràng không? Research gaps từ Chap2 có được "solve" trong Chap3-4 không?
- [ ] Chap2 (Problem) → Chap3 (Core Innovations): Các "4 scientific contributions" trong Chap3 có mapping 1:1 tới Gap 1, 2, 3 trong Chap2 không?
- [ ] Chap3 (Innovations) → Chap4 (Implementation): Chap4 có implement những innovations từ Chap3 không? Có missing components không?
- [ ] Chap4 (Implementation) → Chap5 (Experiments): Các experiments trong Chap5 có validate innovations từ Chap3 không? Có experiment nào validate Chap4 infrastructure không?
- [ ] Chap5 (Results) → Chap6 (Conclusion): Conclusion có summarize tất cả findings từ Chap5 không?

**B4. Gap → Innovation Mapping**
- [ ] Gap 1 (Context Collapse) → Contribution 1 (4D Threshold): Rõ ràng?
- [ ] Gap 2 (Sequential Pipeline) → Contribution 2 (Rendezvous): Rõ ràng?
- [ ] Gap 3 (Single-Strategy Drift) → Contribution 3 (IEC): Rõ ràng?
- [ ] Có gap nào không được address không?
- [ ] Có innovation nào không mapping tới gap nào không?

**B5. RQ → Innovation → Experiment Mapping**
- [ ] RQ1 (Multi-Layer) → Innovation 1-4? → Experiments: Exp1, Exp2?
- [ ] RQ2 (Hybrid Model) → Innovation 4 (Hybrid K-Means iForestASD)? → Experiments: Exp3, Exp4?
- [ ] RQ3 (IEC) → Innovation 3 (IEC)? → Experiments: Exp5 (ADWIN), Exp6?
- [ ] RQ4 (Rendezvous) → Innovation 2 (Rendezvous)? → Experiments: Exp7, Exp8?
- [ ] RQ5 (4D Threshold) → Innovation 1 (4D Threshold)? → Experiments: Exp9, Exp10?
- [ ] Có experiment nào không mapping tới RQ nào không? (VD: Exp5, Exp11-13)
- [ ] Experiment 5 xuất hiện ở 3 nơi khác nhau: RQ3 section, Additional Validation section. Đây là cùng experiment hay khác nhau?

---

### C. REPETITION ANALYSIS (Trùng lặp nội dung)

**C1. Cross-Chapter Duplication**
- [ ] **Research Gaps**: Gap 1, 2, 3 xuất hiện trong thesis.tex (Section 1.2 "Gaps in Current Research") VÀ chap2.tex (Section "Research Gaps") — đây là intentional repetition hay lỗi?
- [ ] **Five DQ Problem Categories**: chap2.tex Section "Case Study" liệt kê 5 categories. chap4.tex Section "Dataset-Specific Challenges" cũng liệt kê 3 challenges. Có overlap không?
- [ ] **Hybrid K-Means iForestASD**: chap3.tex (Innovation 4) và chap4.tex (Layer 2b, Cold-Start Model Training) — trùng lặp?
- [ ] **Feature Vector (21D)**: chap3.tex Innovation 4 "Spatio-Temporal Feature Engineering" VÀ chap4.tex Layer 2b Feature Vector table — trùng lặp?
- [ ] **IEC 4 Strategies**: chap3.tex Innovation 3 VÀ chap4.tex Layer 4 — trùng lặp?
- [ ] **Cold-Start Model Training**: chap4.tex Section 4.1.3 dài 14 trang — có trùng với chap3.tex Innovation 4 không?
- [ ] **ADWIN formula**: chap1.tex Section "Drift Detection" VÀ chap3.tex Innovation 3 — justified hay redundant?

**C2. Within-Chapter Duplication**
- [ ] chap4.tex: "System Overview" → Infrastructure Layers (3 bullets) → Processing Pipeline Overview (4 bullets) → Cold-Start Model Training (14 pages) → Four-Layer Pipeline (tiếp). Có quá dài không? Có redundant overview không?
- [ ] chap3.tex Innovation 2: Có 4 sub-sub-sections (Design Properties, Early Exit, Research Contribution) — cấu trúc có nhất quán với Innovation 1, 3, 4 không?

**C3. Quantified Repetition Scope**
- [ ] Ước tính: Có bao nhiêu % nội dung bị trùng lặp?
- [ ] Trùng lặp có justified (VD: ADWIN formula vì 2 contexts khác nhau) hay không justified (VD: Cold-Start trùng 14 trang)?

---

### D. VIOLATIONS & INCONSISTENCIES (Vi phạm & Không nhất quán)

**D1. Numbering Violations**
- [ ] chap3.tex: `\chapter{CORE INNOVATIONS}` nhưng TOC (thesis.toc line 46) = "CHAPTER 4. CORE INNOVATIONS" — chap3.tex nên là Chapter 4, không phải Chapter 3
- [ ] chap4.tex: `\chapter{SYSTEM ARCHITECTURE}` nhưng TOC (thesis.toc line 74) = "CHAPTER 5. SYSTEM ARCHITECTURE" — chap4.tex nên là Chapter 5
- [ ] chap3.tex: Section numbering 3.1, 3.2, 3.3, 3.4, 3.5, 3.6 — nhưng file này là Chapter 4, sections nên là 4.1, 4.2...
- [ ] chap4.tex: Section numbering 4.1, 4.2, 4.3, 4.4, 4.5, 4.6 — nhưng file này là Chapter 5, sections nên là 5.1, 5.2...
- [ ] chap1.tex (Background): Section numbering có bắt đầu từ 2.1 không? Có off-by-one với Chap2 trong thesis structure không?
- [ ] chap2.tex (Problem Definition): Section numbering có bắt đầu từ 3.1 không?
- [ ] Hypothesis labels: thesis.tex defines H1-H6 in Section 3.5, nhưng thesis.toc line 44 = "3.5 Research Hypotheses" (section 3.5, nhưng thesis.tex Section 3.5 là Problem Definition, không phải Introduction). Mismatch này ảnh hưởng gì?
- [ ] chap2.tex: Sections 3.1, 3.2, 3.3, 3.4, 3.5, 3.6 — nhưng file này là Chapter 3, sections nên là 3.1, 3.2...

**D2. Factual Inconsistencies**
- [ ] chap3.tex Chapter Summary: "F1=0.87" nhưng chap5.tex Experiment 3 Results Table: "Weighted Average: F1=0.87" ✓ và Experiment 4 Table: "Proposed: F1=0.91" ✗. Sự khác biệt 0.04 giải thích được không?
- [ ] chap2.tex "Case Study: NYC Yellow Taxi Dataset": "3,076,903 records from July 2024" VÀ "2,964,624 records" (January 2024) — OK, 2 different months
- [ ] chap4.tex Cold-Start: "3,076,903 raw records" → "3.08M" — OK, rounding of 3,076,903
- [ ] chap2.tex: Dataset coverage "January 2024 through February 2026" = 25 months (Jan 2024 to Feb 2026 inclusive = 14 + 12 + 2 = 28? tính lại: Jan 2024 to Dec 2024 = 12 months, Jan 2025 to Dec 2025 = 12 months, Jan-Feb 2026 = 2 months = 26 months ✓). chap5.tex: "26 months" ✓. Nhưng chap2.tex: "temporal validation across 26 months" ✓
- [ ] chap4.tex Cold-Start: "79,485 physical violations (2.58%)" → "79K+" VÀ chap2.tex: "negative fare (1.8%)" + "zero distances (1.5%)" = 3.3%+. Không khớp. Giải thích: chap2 là % trên tổng records, chap4 là % trên filtered set?
- [ ] chap4.tex: "Chapter 4 presents experiments" trong thesis outline → nhưng chap4.tex thực tế là SYSTEM ARCHITECTURE. chap5.tex mới là EXPERIMENTS. Error trong thesis.tex outline.

**D3. Metric Usage Inconsistencies**
- [ ] chap5.tex Section 5.1: Liệt kê BAR (Balanced Accuracy by Requested Labels) nhưng KHÔNG có experiment nào sử dụng BAR. Tại sao định nghĩa nhưng không dùng?
- [ ] chap5.tex Section 5.1: MCC được định nghĩa nhưng không dùng trong experiments. Đúng hay sai?
- [ ] chap5.tex: "Ground Truth: Synthetic anomalies" — có đủ robust để validate production deployment không?

**D4. LaTeX/Formatting Violations**
- [ ] chap4.tex line 49: `\subsection{Cold-Start Model Training (Offline Phase)}` → Section comment là "% 4.1 System Overview" nhưng actual content là Cold-Start. Mismatch comment vs content.
- [ ] chap4.tex line 49-50: System Overview subsection chỉ có 1 dòng, rồi nhảy thẳng vào Infrastructure Layers. Có orphan subsection không?
- [ ] chap4.tex System Overview: "This separation clarifies that the **infrastructure layers**..." → đoạn này dường như bị cắt, không có closing. Kiểm tra line 47-48.
- [ ] thesis.tex: `\input chap1` sau `\chapter{INTRODUCTION}` (inline) — có nghĩa là chap1.tex = Chapter 2 (Background), nhưng thesis.tex Section 1.6 nói "Chapter 1 covers background" → contradiction trong chính thesis.tex

---

### E. CITATION & BIBLIOGRAPHY ANALYSIS

**E1. Citation.bib Errors**
- [ ] `s察ter2018automating` (line 372): Author field = "{Schelter, Sebastian and Grafe, Dustin and Kirchhoff, Kai and Schiller, Johannes and Schenk, Thomas}" nhưng entry type là `@article` — OK, có booktitle trong content nên có thể là `@inproceedings`?
- [ ] `contextdq2022` (line 402): author = "{{Context in Data Quality Management: A Systematic Literature Review}}" — ĐÂY LÀ LỖI NGHIÊM TRỌNG. Author phải là actual author names, không phải title. Entry này cần fix.
- [ ] `bayram2024adaptive` (line 411): Được cite trong chap1.tex như là "Adaptive Data Quality Scoring Framework using Drift-Aware Mechanism" — OK, nhưng có đúng title không?
- [ ] `river_lib` (line 514): author = "{{Online ML}}" — generic organizational author. Acceptable?
- [ ] `grab_streaming_dq` (line 610) và `grab_engineering2024` (line 421): Có 2 entries cho Grab. Có duplicate không?

**E2. Citation Style**
- [ ] Tất cả framework/tool citations (Kafka, Flink, PostgreSQL, etc.) dùng `@misc` — consistent? IEEE/ACM style thường dùng gì?
- [ ] Có citation nào bị orphan (cited trong text nhưng không có trong bib) không?
- [ ] Có bib entry nào không được cited không?

**E3. Citation Completeness**
- [ ] chap1.tex cite `lu2018learning` và `bifet2007learning` cho drift detection — đủ chưa?
- [ ] chap3.tex Innovation 1: Có cite existing context-aware DQ work không? (VD: "prior work on context-aware DQ uses manual threshold specification")
- [ ] chap3.tex Innovation 2: "Great Expectations, AWS Deequ, Soda Core, Stream DaQ all use sequential processing" — có cite đầy đủ không?

---

### F. WRITING QUALITY & STYLE

**F1. Academic Tone**
- [ ] chap3.tex: "IMPORTANT:" và "Note on" boxes — phù hợp với academic thesis tone?
- [ ] chap4.tex Cold-Start: "The term 'ultra-clean' is not marketing language" — defensive tone, không phù hợp với academic writing
- [ ] chap4.tex: "This is CRITICAL:" — emphasis quá mạnh cho academic writing
- [ ] chap4.tex: "No existing framework" claims — có đủ citations để support những claims này không?

**F2. Sentence Quality**
- [ ] chap3.tex: "The root cause is **context collapse**: treating all trips as identical when they are fundamentally different." — Tốt
- [ ] chap3.tex Innovation 2: Equation (4) "= 53.9ms (mean)" — có trailing parenthesis không closed đúng
- [ ] chap4.tex: "CA-DQStream solves this via a **3-Stage Sequential Funnel** applied *offline* to the training dataset (January 2024, 3.08M raw records):" — Có contradiction: January 2024 data = 2,964,624 records (training set), nhưng 3.08M = July 2024 (validation set). Đây là bug.

**F3. Figure & Table Quality**
- [ ] Tất cả figures có source citations chưa?
- [ ] Tất cả tables có captions chưa?
- [ ] Figure labels có consistent (fig: vs fig.) không?

---

### G. EXPERIMENT DESIGN & METHODOLOGY

**G1. Experiment Numbering**
- [ ] RQ1: Exp 1, Exp 2 ✓
- [ ] RQ2: Exp 3, Exp 4 ✓
- [ ] RQ3: chap5.tex nói "Experiment 6" trong RQ3 section (line 272: "\subsection{Experiment 6: ADWIN Drift Detection Effectiveness}") — Vậy Experiment 5 ở đâu? chap5.tex line 496: "\subsection{Experiment 5: Business Rules Effectiveness}" trong "Additional Validation" section. Nhưng thesis.tex và chap2.tex reference "RQ3: [Exp 5, 6]". Có mismatch:
  - RQ3 section in chap5.tex: Experiment 6 (not 5!)
  - Additional Validation: Experiment 5 (Business Rules) — nhưng Business Rules là Layer 2, thuộc RQ1 (Multi-Layer Architecture)
  - Conclusion summary: "RQ3 (IEC Multi-Strategy Adaptation): [Exp 5, 6]" — labels không match actual experiment numbers
- [ ] RQ4: Exp 7, Exp 8 ✓
- [ ] RQ5: Exp 9, Exp 10 ✓
- [ ] Additional: Exp 5 (BR), Exp 11, Exp 12, Exp 13 — these are labeled "additional" nhưng Exp 5 thực ra thuộc RQ1 content

**G2. Baseline Comparison**
- [ ] chap5.tex Experiment 4: 5 baselines được định nghĩa. Nhưng "Baseline 4 — Full iForest" vs "Proposed — Context-Aware iForest" — Sự khác biệt giữa Baseline 4 và Proposed là gì? (cùng 21D, cùng K-Means, cùng 4D thresholds — chỉ khác tên gọi?)
- [ ] chap3.tex Innovation 4: "A common misconception is that the hybrid architecture trains separate Isolation Forest models per cluster... This is **incorrect**." — Phủ định giả thuyết không tồn tại. Có cần thiết không?

**G3. Statistical Rigor**
- [ ] chap5.tex: "p < 0.001 (t-test)" — đủ thông tin về test statistic, degrees of freedom chưa?
- [ ] chap5.tex: Multiple comparisons (5 baselines) — có correction for multiple testing không? (Bonferroni, FDR)
- [ ] chap5.tex: "Run 5 random seeds per variant" — mean hay median? Standard deviation?

---

### H. DEEP CONTENT ANALYSIS (Phân tích sâu)

**H1. Architecture Confusion: Rendezvous vs Four-Layer**
- [ ] thesis.tex Section 1.3 "Expected System": Liệt kê "five layers" (Data Source → Kafka → Processing → Data Storage → Monitoring)
- [ ] thesis.tex Section 1.4 Objectives: "four-stage processing pipeline" (Schema Validation, Business Rules, Isolation Forest, ADWIN)
- [ ] chap4.tex Section 4.2: "four-layer processing pipeline" (Schema → Rendezvous → MetaAgg → IEC)
- [ ] chap3.tex Innovation 2: "four-stage Rendezvous Pipeline" — rendezvous có phải là Layer 2 không?
- [ ] Sự khác biệt giữa "layer" và "stage": được định nghĩa rõ ràng không? Hay dùng interchangeably?
- [ ] "Layer 2a: Canary Branch (Rule-Based)" và "Layer 2b: Complex Branch (ML-Based)" — nếu tách Layer 2 thành 2a và 2b, thì tổng cộng có phải 6 layers/stages không?

**H2. The 13.5% Early Exit Rate**
- [ ] chap3.tex Innovation 2: "13.5% of records flagged by Canary bypass Complex entirely" — đây là experiment finding hay assumption?
- [ ] chap5.tex Experiment 7: So sánh Rendezvous vs Linear. Bảng shows "Early Exit (%)" = 13.5% cho Rendezvous. Vậy 13.5% được MEASURED in Experiment 7?
- [ ] Nhưng Experiment 7 là "RQ4: Rendezvous vs Linear Pipeline" — có đo early exit rate không?
- [ ] chap3.tex (written BEFORE experiments) sử dụng 13.5% như là một measured fact. Có circular reference không?

**H3. Broader Impact Section (chap6.tex)**
- [ ] chap6.tex Section 6.5 "Broader Impact" — đây là standard thesis section hay tự thêm?
- [ ] Content: "Production ML Systems", "Data Quality as Code", "Regulatory Compliance", "Open Source Contribution" — nghe như policy paper, không phải academic contribution
- [ ] So sánh với standard thesis conclusion structure: Summary → Contributions → Limitations → Future Work → [Optionally: Broader Impact]
- [ ] Broader Impact có nên là phần của thesis không? Có phù hợp với thesis scope không?

**H4. Cold-Start Dataset Date Mismatch**
- [ ] chap4.tex Cold-Start: "3-Stage Sequential Funnel applied *offline* to the training dataset (January 2024, 3.08M raw records)"
- [ ] NHƯNG chap2.tex: "Training (Baseline): January 2024, 2,964,624 records" → 2.96M, không phải 3.08M
- [ ] chap2.tex: "Primary Test: July 2024, 3,076,903 records" → 3.08M
- [ ] chap4.tex Cold-Start: Có thể là typo (January = 2.96M, not 3.08M) hoặc có 2 datasets?

---

## 5. OUTPUT FORMAT

### Phân tích phải có:

**PHẦN 1: CRITICAL ISSUES (Vi phạm nghiêm trọng)**
Mỗi issue cần có:
- Issue ID (e.g., CRITICAL-001)
- Category (Structural / Logical / Factual / Repetition / Citation / Writing)
- Severity (CRITICAL / MAJOR / MINOR)
- Description: Mô tả ngắn gọn issue
- Location: File và dòng/line
- Evidence: Quote trực tiếp từ source
- Recommendation: Cách fix cụ thể

**PHẦN 2: STRUCTURAL ANALYSIS (Báo cáo cấu trúc)**
- Chapter-by-chapter flow diagram
- TOC accuracy report
- Section numbering report
- Missing/orphan sections

**PHẦN 3: CONTENT ANALYSIS (Phân tích nội dung)**
- Repetition map (where content is duplicated)
- Logical flow assessment
- Gap→Innovation→Experiment mapping

**PHẦN 4: CITATION AUDIT (Kiểm toán trích dẫn)**
- Citation errors
- Missing citations
- Orphan entries
- Style consistency

**PHẦN 5: EXPERIMENT DESIGN REVIEW (Đánh giá thiết kế thí nghiệm)**
- Experiment numbering consistency
- Baseline comparison validity
- Statistical rigor

**PHẦN 6: WRITING QUALITY (Chất lượng viết)**
- Academic tone
- Sentence quality
- Formatting issues

**PHẦN 7: SUMMARY & RECOMMENDATIONS (Tổng hợp & Khuyến nghị)**
- Top 10 issues by priority
- Suggested fixes (prioritized)
- Overall assessment (Accept / Minor Revisions / Major Revisions / Reject)

---

## 6. AGENT EXECUTION PLAN

### Phase 1: Parallel Reading (3 subagents song song)
1. `reviewer` — Analyze thesis.tex vs chap*.tex structure mismatch
2. `formatter` — Audit citation.bib and LaTeX formatting
3. `peer-review` — Review hypothesis validity and RQ-RQ mapping

### Phase 2: Deep Content Analysis (2 subagents song song)
4. `document-planner` — Analyze logical flow and content gaps
5. `writer` — Check writing quality and style

### Phase 3: Synthesis (1 subagent)
6. `academic-paper-reviewer` — Compile all findings into unified report

---

## 7. ASSUMPTIONS

- Đây là thesis undergraduate hoặc master's level tại Vietnamese university (VNU-UET)
- Thesis được viết theo LaTeX book class
- Citation style: plainnat (numeric with author-year optional)
- Target format: PDF output
- Không cần thay đổi nội dung research — chỉ phân tích và recommend fixes

---

## 8. SUCCESS CRITERIA

Phân tích hoàn chỉnh khi:
- [ ] Tất cả 7 phần được báo cáo đầy đủ
- [ ] Mỗi issue có Evidence trích dẫn trực tiếp từ source files
- [ ] Recommendations cụ thể và actionable
- [ ] Tổng hợp đưa ra overall assessment rõ ràng
- [ ] Sử dụng tối thiểu 3 subagents trong quá trình phân tích
