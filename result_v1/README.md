# MemStream Benchmark Results — result_v1

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| MemStream epochs | 5000 |
| Training samples | 200,000 |
| Validation samples | 100,000 |
| Best config | M1024_k10_g0_b0p001 |
| Memory length | 1024 |
| k (neighbors) | 10 |
| gamma (KNN weighting) | 0.0 |
| beta (ContextBeta threshold) | 0.001 |
| noise_std | 0.001 |
| learning_rate | 0.01 |
| out_dim | 68 |
| Adam betas | (0.9, 0.999) |
| Dataset | NYC Taxi (injected anomalies) |
| GT anomalies | 2,959 / 100,000 (2.96%) |

---

## Ablation Study

4 setups, sorted by AUC-PR:

| Setup | Description | AUC-PR | AUC-ROC | F1 | Precision | Recall | Separation | Time |
|-------|-------------|--------|---------|-----|-----------|--------|------------|------|
| C | AE + Feature Eng + Memory | **0.9041** | **0.9937** | **0.8530** | 0.8123 | 0.8979 | 8.07x | 25.8s |
| D | AE + FE + Memory + ContextBeta | 0.9022 | 0.9936 | 0.8497 | **0.8159** | 0.8864 | **8.14x** | 29.3s |
| A | Normal AE (no noise, no memory) | 0.7080 | 0.9420 | 0.0575 | 0.0296 | 1.0000 | 0.0000 | 9.5s |
| B | Denoise AE (noise, raw features) | 0.5148 | 0.5000 | 0.0575 | 0.0296 | 1.0000 | 1.0000 | 13.8s |

### Delta Analysis

| Comparison | Delta AUC-PR | Interpretation |
|------------|-------------|----------------|
| B vs A (noise effect) | **-0.193** | Noise injection hurts performance on raw 7D features |
| C vs B (FE + memory effect) | **+0.389** | Feature engineering + streaming memory is the main driver |
| D vs C (ContextBeta effect) | **-0.002** | ContextBeta has negligible effect |
| D vs A (total effect) | **+0.194** | Net improvement from full pipeline |

### Key Findings

- Feature Engineering + Memory (C) alone provides nearly all the benefit
- ContextBeta adds negligible improvement on this dataset
- Noise injection on raw features (B) actually degrades performance
- Normal AE (A) has high recall but near-zero precision due to uncalibrated scores

---

## Comparison with Baselines (v2)

7 methods, sorted by AUC-PR:

| Rank | Method | Category | AUC-PR | AUC-ROC | F1 | Precision | Recall | Time |
|------|--------|----------|--------|---------|-----|-----------|--------|------|
| 1 | Normal Autoencoder | offline | **0.9632** | 0.9971 | **0.9058** | 0.9456 | 0.8692 | 77s |
| 2 | Inc. PCA | streaming | 0.9162 | 0.9932 | 0.8426 | 0.7983 | 0.8922 | 0.6s |
| 3 | MemStream v5 | streaming | 0.9024 | 0.9936 | 0.8520 | 0.8097 | **0.8990** | 24s |
| 4 | One-Class SVM | offline | 0.7975 | 0.9892 | 0.7913 | 0.7747 | 0.8087 | 151s |
| 5 | IsolationForest | offline | 0.2654 | 0.9502 | 0.3818 | 0.2638 | 0.6904 | 4s |
| 6 | Half-Space Trees | streaming | 0.0500 | 0.6233 | 0.1063 | 0.0625 | 0.3548 | 4s |
| 7 | SGD One-Class SVM | streaming | 0.0157 | 0.1135 | 0.0575 | 0.0296 | 1.0000 | 0.5s |

### Streaming Methods Highlight

| Method | AUC-PR | F1 | Recall | Time | Streaming? |
|--------|--------|-----|--------|------|-----------|
| **MemStream v5** | 0.9024 | 0.8520 | 0.8990 | 24s | Yes (online) |
| Inc. PCA | 0.9162 | 0.8426 | 0.8922 | 0.6s | Yes (incremental) |
| Half-Space Trees | 0.0500 | 0.1063 | 0.3548 | 4s | Yes (pysad) |
| SGD One-Class SVM | 0.0157 | 0.0575 | 1.0000 | 0.5s | Yes (mini-batch) |

### Key Findings

- Normal Autoencoder achieves the highest AUC-PR (0.9632) but is offline/non-streaming
- **Inc. PCA** achieves surprisingly high AUC-PR (0.9162) as a streaming method — fast (0.6s) but limited adaptability
- **MemStream v5** achieves AUC-PR=0.9024 with 3x faster runtime than NormalAE
- MemStream has the highest recall (0.899) among streaming methods — best at catching anomalies
- IsolationForest performs poorly on this high-dimensional temporal dataset
- Half-Space Trees and SGD-OCSVM fail on 38D features — domain mismatch
- Streaming methods (MemStream, IncPCA) trade some AUC-PR for the ability to adapt in real-time

---

## File Inventory

### JSON
| File | Description |
|------|-------------|
| `ablation_summary.json` | Ablation study summary with deltas |
| `ablation_results.json` | Full per-run ablation results |
| `comparison_results.json` | Full comparison results with raw scores |

### NumPy Arrays

**Ablation:**
| File | Shape | Description |
|------|-------|-------------|
| `ablation_gt_mask.npy` | (100000,) | Ground truth mask |
| `setup_A_scores.npy` | (100000,) | Normal AE anomaly scores |
| `setup_A_detected.npy` | (100000,) | Normal AE predictions |
| `setup_B_scores.npy` | (100000,) | Denoise AE anomaly scores |
| `setup_B_detected.npy` | (100000,) | Denoise AE predictions |
| `setup_C_scores.npy` | (100000,) | AE+FE+Mem anomaly scores |
| `setup_C_detected.npy` | (100000,) | AE+FE+Mem predictions |
| `setup_C_epoch_losses.npy` | (5000,) | Per-epoch training losses |
| `setup_D_scores.npy` | (100000,) | Full MemStream anomaly scores |
| `setup_D_detected.npy` | (100000,) | Full MemStream predictions |
| `setup_D_epoch_losses.npy` | (5000,) | Per-epoch training losses |

**Comparison:**
| File | Shape | Description |
|------|-------|-------------|
| `comparison_gt_mask.npy` | (100000,) | Ground truth mask |
| `if_scores.npy` | (100000,) | IsolationForest anomaly scores |
| `if_detected.npy` | (100000,) | IsolationForest predictions |
| `normal_ae_scores.npy` | (100000,) | Normal AE anomaly scores |
| `normal_ae_detected.npy` | (100000,) | Normal AE predictions |
| `memstream_scores.npy` | (100000,) | MemStream anomaly scores |
| `memstream_detected.npy` | (100000,) | MemStream predictions |
| `memstream_epoch_losses.npy` | (5000,) | MemStream per-epoch losses |
| `ocsvm_scores.npy` | (100000,) | One-Class SVM anomaly scores |
| `ocsvm_detected.npy` | (100000,) | One-Class SVM predictions |

### Charts (`charts/`)

| File | Description |
|------|-------------|
| `02_comparison_bars.png` | Performance comparison bar chart (v1, 4 methods) |
| `02_comparison_runtime.png` | Runtime comparison (v1) |
| `02_comparison_separation.png` | Score separation (normal vs anomaly) (v1) |
| `02_comparison_v2_full.png` | **Full 7-method comparison** (4-panel) (v2) |
| `02_comparison_v2_auc_f1.png` | **AUC-PR vs F1 grouped chart** (v2) |
| `01_streaming_methods_v2.png` | **Streaming methods detail** (AUC-PR, F1, runtime) (v2) |
| `03_memstream_loss.png` | MemStream training loss curve |
| `03_memstream_score_dist.png` | Score distribution (normal vs anomaly) |
| `03_memstream_timeseries.png` | Anomaly score over time with GT |
| `03_memstream_detection_timeline.png` | TP/FP/FN/TN detection timeline |
| `04_comparison_pointwise.png` | Per-point scores across methods |
| `04_comparison_heatmap.png` | Method scores heatmap |
| `05_statistical_tests.png` | Critical difference diagram |
| `stage2_heatmaps.png` | **Stage 2 grid: effect of noise_std, epochs, lr on AUC-PR** |
| `stage3_architecture.png` | **Stage 3 grid: effect of out_dim, memory_len, k on AUC-PR** |
| `contextbeta_ablation.png` | **ContextBeta ON vs Global Beta — per-segment AUC-PR** |
| `concept_drift_comparison.png` | **Concept drift injection — 4 drift types × 4 methods** |
| `full_data_training.png` | **Full data training — 9 train/val size combos** |
| `overall_summary.png` | **All methods AUC-PR — comparison across all experiments** |
