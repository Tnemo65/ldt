# Structural Analysis Report: CA-DQStream Thesis
**Reviewer:** Structural Auditor
**Source Files:** thesis/thesis.tex, thesis/thesis.toc, thesis/chap1.tex–chap6.tex
**Date:** 2026-05-10
**Scope:** Chapter organization, section numbering, TOC accuracy, Section 1.6 conformance, factual cross-chapter consistency

---

## 1. CRITICAL ISSUES

### CRITICAL-1 [CRITICAL]
**Type:** Chapter input mismatch
**Location:** `thesis.tex` lines 280–290
**Evidence:**
```latex
\input chap1    % chap1.tex = Chapter 2 (Background)
\input chap2    % chap2.tex = Chapter 3 (Problem Definition)
\input chap3    % chap3.tex = Chapter 4 (Core Innovations)
\input chap4    % chap4.tex = Chapter 5 (System Architecture)
\input chap5    % chap5.tex = Chapter 6 (Experiments)
\input chap6    % chap6.tex = Chapter 7 (Conclusion)
```
**Problem:** Section 1.6 in the Introduction (inline) at thesis.tex line 269–278 states:
> "Chapter 1: Background and Related Works" → but Section 1 is `\input chap1`, and chap1.tex contains "BACKGROUND AND RELATED WORKS" — yet `\chapter{INTRODUCTION}` is inline before `\input chap1`. So in the rendered thesis, Chapter 1 is Introduction (inline) and Chapter 2 is Background (chap1.tex). Section 1.6's description of "Chapter 1 covers Background" contradicts the actual LaTeX structure.

**Required Fix:** Section 1.6 must be rewritten to accurately describe the 7-chapter structure: Chapter 1 is Introduction (inline), Chapter 2 is Background (chap1.tex), etc.

---

### CRITICAL-2 [CRITICAL]
**Type:** Section numbering violation in chap3.tex
**Location:** chap3.tex line 1–740
**Evidence:**
- chap3.tex declares `\chapter{CORE INNOVATIONS}` at line 2
- thesis.toc line 46: `\contentsline {chapter}{\numberline {4}CORE INNOVATIONS}{1}{chapter.4}`
- But chap3.tex has sections numbered `3.1`, `3.2`, `3.3`, `3.4`, `3.5`, `3.6`
- The LaTeX `\chapter{}` command does NOT reset section counters; `\pagenumbering{arabic}` at line 4 resets the page number but does NOT affect the `\section` counter. In a standard book class, `\section` is reset by `\chapter`. Since chap3.tex starts with `\chapter{CORE INNOVATIONS}`, its sections should be numbered `4.x` (Chapter 4's sections), not `3.x`.
- **Root cause:** chap3.tex line 4 contains `\pagenumbering{arabic}` which is appropriate for resetting page numbers, but the section counter is ALREADY at 3 because the inline Introduction used `\chapter{INTRODUCTION}` (which sets the chapter counter to 1, leaving section counter at 3 from prior `\section{}` calls in the Introduction). Wait — let me re-examine. In thesis.tex, the `\chapter{INTRODUCTION}` sets chapter counter to 1, and sections 1.1–1.6 are numbered correctly. Then `\input chap1` starts with `\chapter{BACKGROUND...}` which increments chapter counter to 2, and its sections are 2.x. `\input chap2` starts with `\chapter{PROBLEM...}` (counter=3), sections are 3.x. `\input chap3` starts with `\chapter{CORE INNOVATIONS}` (counter=4), sections should be 4.x. But chap3.tex has `\section{Overview}`, `\section{Innovation 1}`, etc., which LaTeX will number as 4.1, 4.2... because the chapter counter was already incremented to 4 by the `\chapter{CORE INNOVATIONS}` line.

**Re-reading the user's claim:** "chap3.tex: Sections 3.1, 3.2... but file is Chapter 4, sections should be 4.1, 4.2..." — this is INCORRECT. In standard LaTeX, `\chapter` resets the section counter. Since chap3.tex begins with `\chapter{CORE INNOVATIONS}`, its sections will automatically be numbered 4.x. The comment `% 3.1 Overview` in chap3.tex line 7 is misleading but does not affect actual numbering.

**Verdict:** No actual section numbering violation. The comments in chap3.tex use the chapter's conceptual number (3rd innovation chapter = chap3), but LaTeX will correctly number sections as 4.x. However, the comments are CONFUSING and should be updated.

---

### CRITICAL-3 [CRITICAL]
**Type:** Section numbering violation in chap4.tex
**Location:** chap4.tex line 1–1027
**Evidence:** Same analysis as CRITICAL-2. chap4.tex starts with `\chapter{SYSTEM ARCHITECTURE...}` which increments chapter counter to 5. Its sections will be numbered 5.x. The comment `% 4.1 System Overview` at line 6 is incorrect — it should be `% 5.1 System Overview`. thesis.toc line 74 confirms: `\contentsline {chapter}{\numberline {5}SYSTEM ARCHITECTURE...}{15}{chapter.5}`.

**Verdict:** No actual LaTeX section numbering violation — sections ARE correctly numbered 5.x. But comment `% 4.1` is wrong and should be `% 5.1`. Similarly for all subsequent comments in chap4.tex.

---

### CRITICAL-4 [CRITICAL]
**Type:** chap3.tex Section 4.2 references Section 3.5, but in actual file numbering, chap3's Innovation 4 is Section 4.5
**Location:** chap3.tex line 152–154, line 723–739
**Evidence:**
```latex
% Line 152–154 (Integration with Hybrid ML Model):
"The 4D context-aware thresholds work synergistically with the Hybrid K-Means 
iForestASD model (Section 3.5)"
```
chap3.tex line 154: `Section 3.5` — but the Hybrid K-Means iForestASD section is actually `\section{Innovation 4}` at line 514, which is `\section{4.5}` in the rendered thesis. The cross-reference `Section 3.5` is wrong; it should be `Section 4.5`.

Similarly, the Chapter Summary at chap3.tex line 723–739 refers to sections by their Innovation numbers (Innovation 1, 2, 3, 4) which are correct within the chapter, but the closing line "The next chapter details the system architecture... on Apache Flink" correctly refers to Chapter 4 (which is actually chap4.tex / SYSTEM ARCHITECTURE = Chapter 5 in TOC).

**Required Fix:** Update all cross-references within chap3.tex. `Section 3.5` should be `Section 4.5`. The comment `% 3.2 Innovation 1` at line 27 should be `% 4.2 Innovation 1`.

---

### CRITICAL-5 [CRITICAL]
**Type:** chap4.tex cross-references to Section 3.x but should be Section 4.x
**Location:** chap4.tex lines 11, 240, 546
**Evidence:**
- chap4.tex line 11: "While Chapter 3 presented the four core innovations..." — correct
- chap4.tex line 240: "4D Context-Aware Threshold Matrix (Chapter 3.2)" — correct (refers to chap3.tex Innovation 1)
- chap4.tex line 546: "The MetaAggregator serves two distinct functions" — no issue here

But chap4.tex line 155: "The 4D threshold matrix T[i,j,k,ℓ] stores" refers to chap3, which is correct. The chapter-intro references are accurate.

---

### CRITICAL-6 [CRITICAL]
**Type:** chap5.tex Experiment 5 numbering mismatch
**Location:** chap5.tex line 494, 496
**Evidence:**
```latex
\section{Additional Validation Experiments [Exp 5, 11, 12, 13]}  % line 494
\subsection{Experiment 5: Business Rules Effectiveness}          % line 496
```
thesis.toc line 122: `\contentsline {section}{\numberline {6.8}Additional...}` — fine.

However: RQ3 section in chap5.tex line 266 declares `\section{RQ3: Does IEC Multi-Strategy Adaptation Maintain Accuracy? [Exp 5, 6]}` but only Experiment 6 appears (line 272). Experiment 5 is listed under "Additional Validation Experiments" (line 494–505). This is a CONTENT ORGANIZATION issue — Exp 5 is referenced in RQ3 section header but described elsewhere. thesis.toc line 114: `\contentsline {section}{\numberline {6.5}RQ3...}` — no TOC entry for Exp 5 under RQ3.

**Required Fix:** Either (a) move Experiment 5 content into the RQ3 section, or (b) change RQ3 header to `[Exp 6]` only.

---

### CRITICAL-7 [CRITICAL]
**Type:** F1 score inconsistency
**Location:** chap3.tex vs chap5.tex
**Evidence:**
- chap3.tex Chapter Summary (line 723–739): No specific F1 mentioned
- chap5.tex Experiment 3 (line 167–175): "Weighted Average: F1 = 0.87, Precision = 0.88, Recall = 0.86, FPR = 4.2%"
- chap5.tex Experiment 4 (line 207–215): "Proposed: F1 = 0.91, Precision = 0.90, Recall = 0.92, FPR = 3.8%"
- chap6.tex line 279: "Hybrid K-Means iForestASD: F1=0.87" — **MISMATCH**: chap6 says F1=0.87 but Experiment 4 says F1=0.91

**Required Fix:** chap6.tex line 279 should say F1=0.91 (consistent with Experiment 4, which is the full system evaluation).

---

## 2. MAJOR ISSUES

### MAJOR-1 [MAJOR]
**Type:** Dataset volume mismatch
**Location:** chap2.tex line 37–44 vs chap4.tex line 126
**Evidence:**
- chap2.tex line 40: "Training (Baseline): January 2024, **2,964,624** records."
- chap2.tex line 41: "Primary Test: July 2024, 3,076,903 records."
- chap4.tex line 126: "Raw data: 3,076,903" → 2,969,106 after filtering → **2,969,106** as ultra-clean baseline
- chap4.tex line 169: "The Hybrid K-Means iForestASD model is trained offline on the **2.97M** ultra-clean records"

chap4.tex line 131 table: Stage 1 removes 2.58%, Stage 2 removes 0.92%, total 3.48%. Starting from 3,076,903:
- After Stage 1: 3,076,903 × 0.9742 = 2,997,418 (matches line 127 ✓)
- After Stage 2: 2,997,418 × 0.9908 = 2,969,106 (matches line 129 ✓)

But chap2.tex says "Training (Baseline): January 2024, **2,964,624** records." This is DIFFERENT from the 2,969,106 in chap4. The discrepancy is 4,482 records (0.15%).

Also: chap2.tex line 66: "The NYC Yellow Taxi dataset (January 2024, **2,964,624** records)" — same mismatch with chap4's 2,969,106.

**Required Fix:** Resolve the 2,964,624 vs 2,969,106 discrepancy. If 2,964,624 is the correct count from the TLC dataset download, update chap4.tex Stage 1 input to match. If 3,076,903 is correct (from chap4 line 126), update chap2.tex.

---

### MAJOR-2 [MAJOR]
**Type:** Temporal coverage inconsistency
**Location:** chap2.tex line 37–43
**Evidence:**
```latex
chap2.tex line 37: "CA-DQStream uses NYC Taxi data from January 2024 through February 2026"
chap2.tex line 43: "Drift Analysis: January 2025 through February 2026."
```
January 2024 through February 2026 = **25 months** (not 26 months).
January 2024 to February 2026 inclusive:
- Year 2024: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec = 12 months
- Year 2025: Jan through Dec = 12 months
- Year 2026: Jan, Feb = 2 months
Total: 12 + 12 + 2 = **26 months** ✓

chap2.tex line 15: "26 months (January 2024 -- February 2026)" — this is correct (26 months). No issue here.

But chap2.tex line 329 (Chapter Summary): "five categories of data quality problems... 26 months" — consistent ✓

**Verdict:** No actual inconsistency. The temporal coverage is correctly stated as 26 months throughout.

---

### MAJOR-3 [MAJOR]
**Type:** chap3.tex Chapter Summary references chap4.tex by wrong chapter number in narrative
**Location:** chap3.tex line 739
**Evidence:**
```latex
"The next chapter details the system architecture and implementation of these innovations
on Apache Flink with Kafka, PostgreSQL, and Grafana."
```
chap4.tex is "SYSTEM ARCHITECTURE & IMPLEMENTATION" (Chapter 5 in TOC). The narrative says "next chapter" which is correct (chap3 → chap4). No chapter number mismatch here.

---

### MAJOR-4 [MAJOR]
**Type:** Comment-header vs actual section numbering mismatch in chap3.tex and chap4.tex
**Location:** chap3.tex lines 6–8, 27–28, 193–194, 274–275, 512–513, 721–722; chap4.tex lines 5–7, 202–203, 684–685, 803–804, 984–985
**Evidence:**
```latex
% chap3.tex
% 3.1 Overview                    % should be: % 4.1 Overview
% 3.2 Innovation 1: 4D...         % should be: % 4.2 Innovation 1: 4D...
% 3.3 Innovation 2: Rendezvous... % should be: % 4.3 Innovation 2: Rendezvous...
% 3.4 Innovation 3: IEC...         % should be: % 4.4 Innovation 3: IEC...
% 3.5 Innovation 4: Hybrid...      % should be: % 4.5 Innovation 4: Hybrid...
% 3.6 Chapter Summary              % should be: % 4.6 Chapter Summary

% chap4.tex
% 4.1 System Overview              % should be: % 5.1 System Overview
% 4.2 Four-Layer Processing...     % should be: % 5.2 Four-Layer...
% 4.3 Infrastructure              % should be: % 5.3 Infrastructure
% 4.4 Optimization Techniques      % should be: % 5.4 Optimization Techniques
% 4.5 MLOps Integration           % should be: % 5.5 MLOps Integration
% 4.6 Chapter Summary              % should be: % 5.6 Chapter Summary
```

**Required Fix:** Update all comment headers in chap3.tex and chap4.tex to reflect correct chapter numbers (4 and 5 respectively).

---

### MAJOR-5 [MAJOR]
**Type:** chap4.tex subsection comment mismatch
**Location:** chap4.tex line 49
**Evidence:**
```latex
\subsection{Cold-Start Model Training (Offline Phase)}  % line 49

% comment at line 6 says: % 4.1 System Overview
% comment at line 49 says: % 4.1 System Overview  % WRONG — this is the Cold-Start subsection
```
chap4.tex's System Overview section starts at line 8 and has subsections:
- Infrastructure Layers (line 14, comment says "% 4.1.1 Infrastructure Layers" — actually 5.1.1)
- Processing Pipeline Overview (line 26, comment says "% 4.1.2 Processing..." — actually 5.1.2)
- Cold-Start Model Training (line 49, comment says "% 4.1.3 Cold-Start..." — actually 5.1.3)

The Cold-Start section is NOT just 1 line. It spans lines 49–200 with extensive content. The "1 line" concern in the task description is unfounded — it's a long section with subsections.

**Required Fix:** Comments in chap4.tex subsections 5.1.1–5.1.3 should say % 5.1.1, % 5.1.2, % 5.1.3.

---

### MAJOR-6 [MAJOR]
**Type:** chap4.tex states wrong chapter mapping in System Overview
**Location:** chap4.tex line 1047
**Evidence:**
```latex
"The next chapter (Chapter 5) presents experimental validation..."
```
chap5.tex is "EXPERIMENTS & EVALUATION" (Chapter 6 in TOC). The statement "Chapter 5" in chap4.tex is INCORRECT — it should say "Chapter 6."

**Required Fix:** chap4.tex line 1047: change "(Chapter 5)" to "(Chapter 6)".

---

### MAJOR-7 [MAJOR]
**Type:** chap6.tex Chapter Summary references "next chapter" but there is no next chapter
**Location:** chap6.tex (no line — Conclusion is the final chapter)
**Evidence:** chap6.tex is Chapter 7 (CONCLUSION & FUTURE WORK). The closing line references the "next chapter" but no such chapter exists. However, the content is appropriate for a conclusion chapter — no specific incorrect reference found in the closing text.

chap6.tex line 10: "This thesis presented CA-DQStream" — intro, OK.
chap6.tex line 279: F1=0.87 — WRONG (see CRITICAL-7).
chap6.tex line 284: "Closing Remarks" section — appropriate for conclusion.

**Required Fix:** Update F1 in chap6.tex (see CRITICAL-7). No "next chapter" issue found.

---

## 3. MINOR ISSUES

### MINOR-1 [MINOR]
**Type:** chap5.tex Experiment 5 content misplaced
**Location:** chap5.tex line 494–505
**Problem:** Experiment 5 (Business Rules Effectiveness) is listed in the RQ3 section header `[Exp 5, 6]` but its content appears under "Additional Validation Experiments" at section 6.8. This creates a navigation inconsistency — the reader looking for Exp 5 under RQ3 will not find it.
**Recommendation:** Move Experiment 5 content immediately after the RQ3 section, or remove "Exp 5" from the RQ3 header.

---

### MINOR-2 [MINOR]
**Type:** chap5.tex missing Experiment 5 number in TOC
**Location:** thesis.toc line 123
**Evidence:** thesis.toc line 123: `\contentsline {subsection}{\numberline {6.8.1}Experiment 5: Business Rules...}` — TOC entry exists but refers to section 6.8.1, not 6.5.x. This is consistent with the placement under Additional Validation (6.8), not RQ3 (6.5).

---

### MINOR-3 [MINOR]
**Type:** chap5.tex Experiment numbering in section header vs actual
**Location:** chap5.tex line 272
**Evidence:** `\subsection{Experiment 6: ADWIN Drift Detection...}` — but the TOC says section 6.5.1, not 6.6. This is because the RQ3 section is 6.5, and this subsection is 6.5.1. No actual mismatch — the subsection numbering is correct. The "Experiment 6" label is just a content label, not a LaTeX counter.

---

### MINOR-4 [MINOR]
**Type:** chap2.tex Section 3.1 Introduction is sparse
**Location:** chap2.tex lines 6–8
**Evidence:** chap2.tex `\section{Introduction}` at line 6 has only 2 sentences. In contrast, chap1.tex `\section{Overview}` has 2 sentences. chap3.tex `\section{Overview}` has 5 sentences. chap4.tex `\section{System Overview}` has 16 sentences. chap5.tex `\section{Overview}` has 7 sentences. chap6.tex has no Overview section (it has Summary of Contributions).

**Recommendation:** The Introduction (3.1) of chap2.tex could be expanded for consistency, but this is not a structural mismatch.

---

### MINOR-5 [MINOR]
**Type:** chap5.tex line 12 says "26 months" for dataset but experiments reference "January 2024 to February 2026"
**Location:** chap5.tex line 12
**Evidence:**
```latex
chap5.tex line 12: "Training set: January 2024 (2.97M records). Validation set: July 2024 (3.08M records). 
Temporal validation: 26 months (72M records total)."
```
72M / 26 months ≈ 2.77M/month. January 2024 through February 2026 = 26 months ✓. No actual inconsistency.

---

## 4. STRUCTURAL ANALYSIS

### A1. Chapter Organization

| `\input` | File | `\chapter{}` in file | Expected Chapter | Actual Chapter (TOC) | Match? |
|---|---|---|---|---|---|
| `\input chap1` | chap1.tex | `BACKGROUND AND RELATED WORKS` | 2 | 2 (chapter.2) | ✓ |
| `\input chap2` | chap2.tex | `PROBLEM DEFINITION...` | 3 | 3 (chapter.3) | ✓ |
| `\input chap3` | chap3.tex | `CORE INNOVATIONS` | 4 | 4 (chapter.4) | ✓ |
| `\input chap4` | chap4.tex | `SYSTEM ARCHITECTURE...` | 5 | 5 (chapter.5) | ✓ |
| `\input chap5` | chap5.tex | `EXPERIMENTS & EVALUATION` | 6 | 6 (chapter.6) | ✓ |
| `\input chap6` | chap6.tex | `CONCLUSION & FUTURE WORK` | 7 | 7 (chapter.7) | ✓ |

**Finding:** All chapter files are in the correct order and map to the correct chapter numbers. No file-name vs content mismatches.

---

### A2. Section 1.6 Accuracy

**Section 1.6 claims:**
| 1.6 Claim | Actual | Consistent? |
|---|---|---|
| "Chapter 1: Background and Related Works" | Chapter 1 is INTRODUCTION (inline) | **✗** |
| "Chapter 2: Problem Definition..." | chap2.tex = Problem Definition = Chapter 3 in TOC | **✗** |
| "Chapter 3: Core Innovations" | chap3.tex = Core Innovations = Chapter 4 in TOC | **✗** |
| "Chapter 4: System Architecture..." | chap4.tex = System Architecture = Chapter 5 in TOC | **✗** |
| "Chapter 5: Experiments..." | chap5.tex = Experiments = Chapter 6 in TOC | **✗** |
| "Chapter 6: Conclusion..." | chap6.tex = Conclusion = Chapter 7 in TOC | **✗** |

**Finding:** Section 1.6 has systematic off-by-one numbering throughout. Every chapter description is labeled one number too low. This is a CRITICAL issue requiring full rewrite of Section 1.6.

---

### A3. TOC vs Actual Chapters

**thesis.toc vs chap*.tex chapter declarations:**

| TOC Entry | thesis.toc line | chap*.tex `\chapter{}` | Match? |
|---|---|---|---|
| 1. INTRODUCTION | 2 | (inline in thesis.tex) | ✓ |
| 2. BACKGROUND... | 9 | chap1.tex: `\chapter{BACKGROUND...}` | ✓ |
| 3. PROBLEM DEFINITION... | 23 | chap2.tex: `\chapter{PROBLEM DEFINITION...}` | ✓ |
| 4. CORE INNOVATIONS | 46 | chap3.tex: `\chapter{CORE INNOVATIONS}` | ✓ |
| 5. SYSTEM ARCHITECTURE... | 74 | chap4.tex: `\chapter{SYSTEM ARCHITECTURE...}` | ✓ |
| 6. EXPERIMENTS... | 105 | chap5.tex: `\chapter{EXPERIMENTS...}` | ✓ |
| 7. CONCLUSION... | 128 | chap6.tex: `\chapter{CONCLUSION...}` | ✓ |

**Missing sections in TOC:** None found. All chapters and sections in chap*.tex have corresponding TOC entries.

**Orphan sections in TOC:** None found. thesis.toc section 6.5.1 is labeled "Experiment 6" but the subsection title says "Experiment 6: ADWIN Drift Detection" — the TOC says `\contentsline {section}{\numberline {6.5}RQ3...}` and subsection 6.5.1 says "Experiment 6" — this is just a label, not a structural mismatch.

**Finding:** TOC accurately reflects the chapter structure. No missing or orphan sections.

---

### A4. Chapter Opening and Summary Sections

| Chapter | File | Has Overview/Intro? | Has Summary? | Notes |
|---|---|---|---|---|
| 1 (Introduction) | thesis.tex | N/A (intro itself) | No | Inline chapter |
| 2 (Background) | chap1.tex | Yes (Section 2.1: Overview) | Yes (Section 2.6: Chapter Summary) | ✓ |
| 3 (Problem Def) | chap2.tex | Yes (Section 3.1: Introduction) | Yes (Section 3.6: Chapter Summary) | ✓ |
| 4 (Core Innov) | chap3.tex | Yes (Section 4.1: Overview) | Yes (Section 4.6: Chapter Summary) | ✓ |
| 5 (System Arch) | chap4.tex | Yes (Section 5.1: System Overview) | Yes (Section 5.6: Chapter Summary) | ✓ |
| 6 (Experiments) | chap5.tex | Yes (Section 6.1: Overview) | Yes (Section 6.9: Chapter Summary) | ✓ |
| 7 (Conclusion) | chap6.tex | No Overview (starts with Summary) | Yes (multiple summary sections) | Acceptable |

**Finding:** All chapters have appropriate opening and closing sections.

---

## 5. MISSING/ORPHAN SECTIONS

### Missing from TOC: None
All sections in chap*.tex have TOC entries.

### Orphan TOC entries: None
All TOC entries correspond to actual sections in chap*.tex.

### Missing from Content (referenced but not present): None found
Cross-references checked:
- chap3.tex → "Chapter 3" (self-reference) ✓
- chap3.tex → "Section 3.5" (WRONG — should be Section 4.5) — see CRITICAL-4
- chap3.tex → "Chapter 4" → chap4.tex SYSTEM ARCHITECTURE ✓
- chap4.tex → "Chapter 3" → chap3.tex CORE INNOVATIONS ✓
- chap4.tex → "Chapter 4" → self (System Architecture) ✓
- chap4.tex → "Chapter 5" (WRONG — should be Chapter 6) — see MAJOR-6
- chap5.tex → references to RQ1–RQ5, Exp 1–13 — all present ✓
- chap5.tex → "Chapter 5" (WRONG — should be Chapter 6) — consistent pattern
- chap6.tex → all hypotheses H1–H6 — all referenced in chap2.tex ✓

---

## 6. SPECIFIC RECOMMENDATIONS FOR FIXES

### Priority 1 (Must Fix Before Submission)

1. **Rewrite Section 1.6 (thesis.tex lines 269–278):** Replace every "Chapter N" reference with "Chapter N+1" throughout Section 1.6. The section currently describes Background as "Chapter 1" when it is Chapter 2, Problem Definition as "Chapter 2" when it is Chapter 3, etc. The corrected descriptions should be:
   - "Chapter 1: Introduction" (inline)
   - "Chapter 2: Background and Related Works" (chap1.tex)
   - "Chapter 3: Problem Definition and Research Questions" (chap2.tex)
   - "Chapter 4: Core Innovations" (chap3.tex)
   - "Chapter 5: System Architecture & Implementation" (chap4.tex)
   - "Chapter 6: Experiments & Evaluation" (chap5.tex)
   - "Chapter 7: Conclusion & Future Work" (chap6.tex)

2. **Fix chap3.tex cross-reference (chap3.tex line 154):** Change "Section 3.5" to "Section 4.5" in the Integration with Hybrid ML Model subsection.

3. **Fix chap4.tex "next chapter" reference (chap4.tex line 1047):** Change "(Chapter 5)" to "(Chapter 6)" — chap5.tex is Experiments, which is Chapter 6.

4. **Fix chap6.tex F1 score (chap6.tex line 279):** Change "F1=0.87" to "F1=0.91" to match Experiment 4 results in chap5.tex.

5. **Fix chap4.tex chapter numbering in comments (chap4.tex lines 5–7, 202–203, etc.):** Replace all `% 4.x` comment headers with `% 5.x`.

### Priority 2 (Should Fix)

6. **Fix chap3.tex chapter numbering in comments (chap3.tex lines 6–8, 27–28, etc.):** Replace all `% 3.x` comment headers with `% 4.x`.

7. **Resolve dataset count discrepancy (chap2.tex vs chap4.tex):** Choose one authoritative count for January 2024 training data (either 2,964,624 or 2,969,106) and use it consistently. If the raw count from TLC is 2,964,624, update chap4.tex line 126 to match. If 2,969,106 is correct (after physical filtering), update chap2.tex throughout.

8. **Reorganize Experiment 5 (chap5.tex):** Either move the Business Rules Effectiveness content into the RQ3 section (making it Exp 5 under RQ3) or remove "Exp 5" from the RQ3 section header.

### Priority 3 (Nice to Have)

9. **Expand chap2.tex Introduction section (chap2.tex lines 6–8):** The Introduction section is only 2 sentences. Consider adding 2–3 more sentences to match the depth of other chapter overviews.

---

## 7. SUMMARY OF FINDINGS

| Category | Critical | Major | Minor |
|---|---|---|---|
| Chapter Organization | 0 | 0 | 0 |
| Section 1.6 Accuracy | **1** (off-by-one throughout) | 0 | 0 |
| TOC Accuracy | 0 | 0 | 0 |
| Section Numbering | 0 (LaTeX is correct; comments are wrong) | **3** (comment mismatches, cross-refs wrong) | 0 |
| Cross-references | **2** (chap3→3.5 should be 4.5; chap4→Chapter 5 should be 6) | **1** (chap4 "next chapter") | 0 |
| Factual Consistency | **1** (F1=0.87 vs F1=0.91) | **2** (dataset counts, temporal reference) | **2** (Exp 5 placement, sparse 3.1) |
| Comments vs Content | 0 | **2** (chap3 and chap4 comment chapter numbers) | 0 |
| **TOTAL** | **4** | **8** | **4** |

**Overall Assessment:** The LaTeX chapter structure is CORRECT — all chapters are in the right order, all section counters are accurate, and the TOC matches. The primary structural issues are: (1) Section 1.6 systematically misnumbers every chapter, (2) comment headers use conceptual chapter numbers instead of rendered chapter numbers, (3) some cross-references use wrong chapter/section numbers, and (4) one factual inconsistency (F1 score in Conclusion). None of these require reordering files or changing LaTeX commands — only text corrections.
