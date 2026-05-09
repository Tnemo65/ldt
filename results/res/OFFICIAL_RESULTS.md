# CA-DQStream Benchmark — Official Results
**Date:** 2026-05-09
**Hardware:** 32 vCPUs, 88 GB RAM, RTX 3090 Ti 24 GB
**Dataset:** NYC Yellow Taxi Jan 2024 (~2.96M raw records)
**Runs:** 5 variants x 5 seeds x 3 difficulty levels = **75 runs**
**Total time:** ~48 minutes (2179s execution + figures)

---

## Pipeline Funnel

| Stage | Records | Retention |
|-------|---------|-----------|
| Raw | 2,964,624 | 100.0% |
| After L1 (Schema) | 2,763,453 | 93.2% |
| After L2 (Rules) | 2,696,728 | 91.0% |
| After L3 (IQR) | 2,389,796 | 80.6% |
| Train (70%) | 1,672,857 | — |
| Test (30%) | 716,939 | — |

---

## Results by Difficulty Level (mean +/- std, 5 seeds)

### EASY Level (F1=0.01, Recall=1.000, FPR=0.0496)
| Rank | Variant | F1 | Recall | Precision | FPR | Time |
|------|---------|-----|--------|-----------|-----|------|
| 1 | **proposed_context_aware** | **0.828 +/- 0.008** | 0.987 +/- 0.016 | 0.712 +/- 0.003 | **0.0299 +/- 0.0001** | 27.0s |
| 2 | baseline_ratio | 0.791 +/- 0.000 | 1.000 +/- 0.000 | 0.654 +/- 0.000 | 0.0396 +/- 0.0001 | 26.9s |
| 3 | opponent_ocsvm | 0.790 +/- 0.002 | 1.000 +/- 0.000 | 0.653 +/- 0.003 | 0.0399 +/- 0.0005 | 11.7s |
| 4 | opponent_lof | 0.761 +/- 0.005 | 1.000 +/- 0.000 | 0.614 +/- 0.007 | 0.0472 +/- 0.0013 | 51.3s |
| 5 | baseline_static | 0.752 +/- 0.000 | 1.000 +/- 0.000 | 0.603 +/- 0.000 | 0.0494 +/- 0.0001 | 30.0s |

### MEDIUM Level (F1=0.01, Recall=0.01, FPR=0.0496)
| Rank | Variant | F1 | Recall | Precision | FPR | Time |
|------|---------|-----|--------|-----------|-----|------|
| 1 | **proposed_context_aware** | **0.751 +/- 0.027** | 0.842 +/- 0.049 | 0.678 +/- 0.013 | **0.0299 +/- 0.0001** | 26.8s |
| 2 | opponent_lof | 0.739 +/- 0.007 | 0.956 +/- 0.005 | 0.603 +/- 0.008 | 0.0472 +/- 0.0013 | 50.6s |
| 3 | baseline_ratio | 0.726 +/- 0.010 | 0.871 +/- 0.020 | 0.622 +/- 0.005 | 0.0396 +/- 0.0001 | 26.6s |
| 4 | opponent_ocsvm | 0.694 +/- 0.006 | 0.815 +/- 0.011 | 0.605 +/- 0.004 | 0.0399 +/- 0.0005 | 11.6s |
| 5 | baseline_static | 0.665 +/- 0.011 | 0.826 +/- 0.020 | 0.556 +/- 0.006 | 0.0494 +/- 0.0001 | 27.0s |

### HARD Level (F1=0.01, Recall=0.01, FPR=0.0496)
| Rank | Variant | F1 | Recall | Precision | FPR | Time |
|------|---------|-----|--------|-----------|-----|------|
| 1 | opponent_lof | 0.609 +/- 0.007 | 0.713 +/- 0.004 | 0.531 +/- 0.008 | 0.0472 +/- 0.0013 | 49.8s |
| 2 | baseline_ratio | 0.578 +/- 0.006 | 0.622 +/- 0.009 | 0.540 +/- 0.004 | 0.0396 +/- 0.0001 | 26.5s |
| 3 | opponent_ocsvm | 0.567 +/- 0.002 | 0.606 +/- 0.004 | 0.533 +/- 0.002 | 0.0399 +/- 0.0005 | 11.5s |
| 4 | **proposed_context_aware** | **0.563 +/- 0.028** | 0.549 +/- 0.038 | **0.579 +/- 0.017** | **0.0299 +/- 0.0001** | 26.6s |
| 5 | baseline_static | 0.497 +/- 0.013 | 0.549 +/- 0.019 | 0.454 +/- 0.009 | 0.0494 +/- 0.0001 | 27.0s |

---

## Hypothesis Validation

| Level | H1 (21D > 15D) | H2 (cluster > global) | H3 (proposed > opponents) |
|-------|-----------------|----------------------|---------------------------|
| **EASY** | **PASS** (+0.039 F1) | **PASS** (FPR 0.030 vs 0.040) | **PASS** (0.828 vs 0.790) |
| **MEDIUM** | **PASS** (+0.061 F1) | **PASS** (FPR 0.030 vs 0.040) | **PASS** (0.751 vs 0.739) |
| **HARD** | **PASS** (+0.081 F1) | **PASS** (FPR 0.030 vs 0.040) | **FAIL** (0.563 vs 0.609) |

### Key Insight on H3 (HARD):
- LOF achieves highest F1 (0.609) at HARD difficulty due to its density-based nature catching subtle anomalies
- proposed_context_aware maintains **lowest FPR** across ALL difficulty levels (0.0299), critical for production
- proposed has **highest precision** at HARD (0.579 vs 0.531 LOF)

---

## Statistical Testing (proposed_context_aware vs baselines)

| Level | Comparison | p-value (t-test) | Cohen's d | Effect | Significance |
|-------|-----------|-------------------|-----------|--------|-------------|
| EASY | proposed vs baseline_static | 0.0000 | 13.85 | large | *** |
| EASY | proposed vs baseline_ratio | 0.0005 | 6.72 | large | *** |
| EASY | proposed vs opponent_lof | 0.0001 | 10.21 | large | *** |
| EASY | proposed vs opponent_ocsvm | 0.0002 | 6.73 | large | *** |
| MEDIUM | proposed vs baseline_static | 0.0005 | 4.13 | large | *** |
| MEDIUM | proposed vs baseline_ratio | 0.0963 | 1.21 | large | ns |
| MEDIUM | proposed vs opponent_lof | 0.4312 | 0.59 | medium | ns |
| MEDIUM | proposed vs opponent_ocsvm | 0.0139 | 2.86 | large | * |
| HARD | proposed vs baseline_static | 0.0014 | 3.05 | large | ** |
| HARD | proposed vs baseline_ratio | 0.2955 | -0.76 | medium | ns |
| HARD | proposed vs opponent_lof | 0.0265 | -2.26 | large | * |
| HARD | proposed vs opponent_ocsvm | 0.7706 | -0.21 | small | ns |

---

## Key Performance Metrics

### Proposed vs Baseline Static Gap (F1)
| Level | Proposed | Static | Gap |
|-------|----------|--------|-----|
| EASY | 0.828 | 0.752 | **+0.076** |
| MEDIUM | 0.751 | 0.665 | **+0.086** |
| HARD | 0.563 | 0.497 | **+0.066** |

### FPR Comparison (Critical Metric)
| Variant | EASY FPR | MEDIUM FPR | HARD FPR |
|---------|----------|------------|----------|
| **proposed_context_aware** | **0.0299** | **0.0299** | **0.0299** |
| baseline_ratio | 0.0396 | 0.0396 | 0.0396 |
| baseline_static | 0.0494 | 0.0494 | 0.0494 |
| opponent_lof | 0.0472 | 0.0472 | 0.0472 |
| opponent_ocsvm | 0.0399 | 0.0399 | 0.0399 |

**Proposed method maintains consistent FPR across all difficulty levels, well below 5% threshold.**

---

## Generated Files

| File | Description |
|------|-------------|
| `benchmark_results.csv` | All 75 raw results (seed-level metrics) |
| `statistical_tests.csv` | p-values, Cohen's d, effect sizes |
| `benchmark_4panel.png/pdf` | Comprehensive 4-panel figure |
| `benchmark_per_scenario.png/pdf` | Per-scenario detection breakdown |
| `benchmark_effect_size.png` | Cohen's d heatmap |
| `benchmark_train_time.png` | Training time comparison |

---

## Conclusions

1. **H1 (21D > 15D):** PASS at ALL difficulty levels. Ratio features consistently outperform raw features (+4-8% F1).
2. **H2 (cluster > global):** PASS at ALL difficulty levels. Per-cluster thresholds reduce FPR by ~25% vs global (0.030 vs 0.040).
3. **H3 (proposed > opponents):** PASS at EASY/MEDIUM, FAIL at HARD. LOF excels at HARD but with 58% higher FPR.
4. **Production viability:** proposed_context_aware has the lowest and most stable FPR (0.0299 across all levels), critical for minimizing false alarms in real deployments.
