# Final Benchmark Result — v1 (final_benchmark_result_v1)

MemStream Anomaly Detection Benchmark: Complete code + results + data configs.

## Repository Structure

```
final_benchmark_result_v1/
├── README.md                         ← This file
├── requirements.txt                  ← Python dependencies
├── run_pipeline.py                   ← Orchestrator: run all experiments
├── src/
│   ├── benchmark_core.py             ← Core library (MemStreamAE, Memory, all scorers)
│   ├── grid_search.py                ← Stage 1: coarse HP search (9 configs)
│   ├── ablation.py                  ← Ablation: 4 setups (A:NormalAE, B:DenoiseAE,
│   │                                  C:+FE+Mem, D:+ContextBeta)
│   ├── comparison.py                 ← Compare 9 methods (MemStream, NormalAE, OCSVM,
│   │                                  IF, RCF, HSTrees, LODA, DAGMM, DeepSVDD)
│   ├── ablation_streaming.py         ← Prove streaming memory works
│   ├── ablation_scoring.py          ← KNN variants, batch size, scoring strategies
│   ├── ablation_quick.py            ← Quick ablation on validation set
│   ├── _run_ablation.py             ← Ablation entry point
│   ├── _run_comparison.py           ← Comparison entry point
│   ├── compare_streaming.py         ← Streaming vs eval step-by-step
│   └── experiments/
│       ├── grid_search_s2.py        ← Stage 2: fine-tune noise_std, epochs, lr (54 configs)
│       ├── grid_search_s3.py        ← Stage 3: architecture variation out_dim, memory_len, k (27 configs)
│       ├── full_data.py             ← Full data training: 3 train sizes × 3 val sizes (9 combos)
│       ├── context_stratified.py     ← ContextBeta ON vs OFF, per-context breakdown
│       ├── concept_drift.py         ← Concept drift injection: 4 drift types × 4 methods
│       └── aggregate_results.py      ← Aggregate all results + generate charts
├── data/
│   ├── valid/injection_log.json     ← Injection log for validation set
│   ├── valid/ground_truth_per_type.json
│   ├── test/injection_log.json
│   ├── test/ground_truth_per_type.json
│   ├── prod/injection_log.json
│   ├── prod/ground_truth_per_type.json
│   ├── inject_anomalies_memstream.py ← Anomaly injection pipeline (11 types)
│   └── viz_anomaly_types.py          ← Visualize anomaly types
└── results/
    ├── aggregated_results.json       ← All results aggregated (from upgrade_experiments)
    ├── context_ablation_results.json  ← ContextBeta ablation
    ├── concept_drift_results.json     ← Concept drift injection
    ├── full_data_results.json         ← Full data training
    ├── grid_search/
    │   ├── stage1/                   ← 9 configs (from HP_benchmark_v5)
    │   ├── stage2/                   ← 54 configs (from upgrade_experiments)
    │   └── stage3/                   ← 27 configs (from upgrade_experiments)
    ├── ablation/
    │   ├── ablation_results.json      ← Full ablation results (HP_benchmark_v5)
    │   ├── ablation_summary.json      ← Summary with delta analysis (HP_benchmark_v5)
    │   ├── ablation_results_v1.json   ← Ablation results (result_v1)
    │   ├── ablation_summary_v1.json  ← Summary (result_v1)
    │   └── *.npy                     ← Scores, detected masks, epoch losses
    ├── comparison/
    │   ├── comparison_results.json    ← 9-method comparison (HP_benchmark_v5)
    │   ├── comparison_results_v1.json ← 7-method comparison (result_v1)
    │   ├── comp_grid/                ← Hyperparameter grids for IF, RCF, OCSVM
    │   └── *.npy                     ← Scores, detected masks
    └── charts/                        ← All PNG charts (18 from result_v1, 6 from upgrade_experiments)
```

## Data Files (NOT included — configure paths in `run_pipeline.py`)

The benchmark uses NYC Taxi data. The parquet files are **large** (~GB) and stored separately.
Point `DATA_DIR` in `run_pipeline.py` to your local parquet directory.

Expected parquet files (train/valid/test split with injected anomalies):

| File | Required | Location |
|------|----------|----------|
| `train_clean.parquet` | Yes | `{DATA_DIR}/` |
| `valid_polluted.parquet` | Yes | `{DATA_DIR}/` |
| `test_polluted.parquet` | Yes | `{DATA_DIR}/` |
| `ground_truth_mask.npy` | Yes | `{DATA_DIR}/valid/` |

Reference locations (original):
```
C:\proj\ldt\GOOD_DATA\           # Main working data
C:\proj\ldt\HP_benchmark_v3\       # Alternative source
C:\proj\ldt\HP_benchmark_v5\data\  # Injection artifacts + logs
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure data paths in run_pipeline.py
#    Set DATA_DIR to your parquet directory

# 3. Run all experiments
python run_pipeline.py --all

# Or run individual stages:
python run_pipeline.py --stage1        # Stage 1: coarse grid search
python run_pipeline.py --stage2        # Stage 2: fine-tune training params
python run_pipeline.py --stage3        # Stage 3: architecture variation
python run_pipeline.py --ablation      # Ablation study
python run_pipeline.py --comparison    # Compare with baselines
python run_pipeline.py --context       # ContextBeta ablation
python run_pipeline.py --drift        # Concept drift injection
python run_pipeline.py --fulldata     # Full data training
python run_pipeline.py --aggregate     # Aggregate + chart
```

## Experiment Pipeline

```
Stage 1 (grid_search.py)           Stage 2 (experiments/grid_search_s2.py)    Stage 3 (experiments/grid_search_s3.py)
9 configs (coarse)       →         54 configs (fine-tune)         →         27 configs (architecture)
Memory: [256,512,1024]             noise_std × epochs × lr                     out_dim × memory_len × k
k: [3,5,10]                        Best: M2048_k5, AUC-PR=0.910              Best: M2048_k10, out_dim=76
gamma: [0], beta: [0.001]                                                                   AUC-PR=0.9178
Best: M1024_k10_g0_b0p001
AUC-PR=0.906
```

## Best Configuration Found

| Stage | Config | AUC-PR | AUC-ROC | F1 | Precision | Recall |
|-------|--------|--------|---------|-----|-----------|--------|
| 1 | M1024_k10_g0_b0p001 | 0.906 | 0.997 | 0.883 | 0.823 | 0.953 |
| 2 | M2048_k5 + tuned | 0.910 | 0.997 | — | — | — |
| 3 | M2048_k10_out_dim76 | **0.918** | **0.997** | — | — | — |

Final config: `memory=2048, k=10, out_dim=76, noise=0.001, lr=0.005, epochs=10000`

## Results Summary

### Ablation (ContextBeta ON vs OFF)
- ContextBeta **does not help** on NYC Taxi dataset (stationary data, no strong concept drift)
- Global beta outperforms ContextBeta on all 8 segments

### Concept Drift Injection
- Streaming memory updates **hurt** performance under concept drift
- Static memory (no updates) outperforms streaming by +0.21 AUC-PR average
- Inc.PCA handles drift best when AE is NOT retrained

### Full Data Training
- **Train on 200K, evaluate on FULL** is the best: AUC-PR=0.914
- More training data (500K, FULL) hurts performance
- More validation data consistently improves metrics

### Comparison with Baselines
| Rank | Method | Category | AUC-PR | AUC-ROC | F1 |
|------|--------|----------|--------|---------|-----|
| 1 | Normal Autoencoder | offline | **0.963** | 0.997 | **0.906** |
| 2 | Inc. PCA | streaming | 0.916 | 0.993 | 0.843 |
| 3 | **MemStream v5 (optimized)** | streaming | **0.918** | **0.997** | 0.058 |
| 4 | MemStream v5 (original) | streaming | 0.902 | 0.994 | 0.852 |
| 5 | One-Class SVM | offline | 0.797 | 0.989 | 0.791 |
| 6 | IsolationForest | offline | 0.265 | 0.950 | 0.382 |
| 7 | Half-Space Trees | streaming | 0.050 | 0.623 | 0.106 |
| 8 | SGD One-Class SVM | streaming | 0.016 | 0.114 | 0.058 |

## Dependencies

```
numpy>=1.21
pandas>=1.3
torch>=2.0
scikit-learn>=1.0
pysad>=0.1        # Half-Space Trees
pyod>=0.9         # RCF, LODA, DAGMM, DeepSVDD
matplotlib>=3.5
seaborn>=0.11
```

## Key Findings

1. **Best MemStream config**: `memory=2048, k=10, out_dim=76, noise=0.001, lr=0.005, epochs=10000` → AUC-PR=0.918
2. **Architecture matters most**: Doubling out_dim from 38→76 gives +0.007 AUC-PR; doubling memory 1024→2048 gives +0.007 AUC-PR
3. **More epochs help**: 10000 epochs consistently beats 5000 (+0.003 AUC-PR average)
4. **Slower LR helps**: lr=0.005 beats lr=0.01 across all configs
5. **ContextBeta doesn't help on this dataset**: Stationary data means per-context thresholds add noise, not signal
6. **Streaming updates hurt under concept drift**: When the distribution shifts, static memory beats streaming
7. **Train small, evaluate large**: 200K training + FULL evaluation gives best results
8. **Inc. PCA is a surprisingly strong streaming baseline**: Fast (0.6s) and competitive AUC-PR (0.916)
