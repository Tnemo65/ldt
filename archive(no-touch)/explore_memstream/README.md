# MemStream NYC Taxi Evaluation Suite

Deep evaluation framework for MemStream anomaly detection on NYC taxi periodic data.

## Structure

```
explore_memstream/
├── README.md                       # This file
├── eval_ablation_nyc.py            # Hyperparameter sweep (memory, gamma, k, beta, dim, epochs)
├── eval_drift_learning.py          # No-auto vs drift-triggered vs full-auto learning
├── validate_warmup_coverage.py      # Warmup cycle coverage validator
├── viz_clean_data.py               # 6-panel clean data visualization
├── viz_injected_anomalies.py       # Before/after anomaly injection visualization
├── viz_detection_results.py        # Post-detection results (ROC/PR/F1 heatmap)
├── grafana_dashboard.json          # Grafana monitoring dashboard
├── layer_review_checklist.py       # Layer-by-layer verification checklist
├── config/
│   └── nyc_optimized_config.py    # NYC taxi-optimized hyperparameters
└── tests/
    └── test_warmup_validator.py   # Unit tests for warmup coverage
```

## Quick Start

### 1. Validate Warmup Data

Before running any evaluation, ensure warmup data covers at least 1 complete weekly cycle:

```bash
python validate_warmup_coverage.py --data /path/to/nyc_taxi.csv
```

### 2. Run Ablation Study

```bash
python eval_ablation_nyc.py \
    --data /path/to/nyc_taxi.csv \
    --output results/ablation/ \
    --n-anomalies 2000 \
    --warmup-frac 0.6
```

Run specific configs only:

```bash
python eval_ablation_nyc.py \
    --data /path/to/nyc_taxi.csv \
    --config configs/memory_sweep.json \
    --output results/ablation/
```

### 3. Run Drift Learning Experiment

```bash
python eval_drift_learning.py \
    --data /path/to/nyc_taxi.csv \
    --output results/drift/ \
    --n-anomalies 2000
```

### 4. Generate Visualizations

```bash
# Clean data analysis
python viz_clean_data.py --data /path/to/nyc_taxi.csv --output results/viz/

# Anomaly injection analysis
python viz_injected_anomalies.py --data /path/to/nyc_taxi.csv --output results/viz/

# Detection results (needs ablation results first)
python viz_detection_results.py \
    --results "results/ablation/*.json" \
    --output results/viz/
```

### 5. Layer Review Checklist

```bash
# Review all layers
python layer_review_checklist.py --layer all --output results/layer_review.json

# Review specific layer
python layer_review_checklist.py --layer memstream
python layer_review_checklist.py --layer features
```

### 6. Unit Tests

```bash
python -m pytest tests/ -v
# or
python tests/test_warmup_validator.py
```

## Configuration

NYC-optimized hyperparameters are in `config/nyc_optimized_config.py`:

| Parameter | Value | Rationale |
|-----------|-------|------------|
| `memory_len` | 1024 | ~1 day NYC taxi diversity |
| `hidden_dim` | 68 | 2x input (D >= d) |
| `k` | 10 | NYC diversity needs more neighbors |
| `gamma` | 0.5 | Self-recovery from memory poisoning |
| `default_beta` | 0.5 | From warmup score distribution |
| `warmup_epochs` | 100 | Early stopping typically at 20-100 |

## Grafana Dashboard

Import `grafana_dashboard.json` into Grafana:

1. Go to Dashboards → Import
2. Upload the JSON file
3. Select Prometheus datasource
4. Panels: Records/sec, Schema Violations, Hard Rule Violations, Anomaly Rate, kNN Distance, Memory Fill Rate, Drift Events, BAR Rate, Anomalies by Neighborhood

## Dependencies

```
pandas
numpy
torch
matplotlib
seaborn
scikit-learn
```

## Scientific Claims

1. **ContextBeta** (80 thresholds) reduces false alarms vs. single global threshold
2. **Drift-triggered learning** achieves ≥95% of full auto-learning AUC-PR with ≤5% label budget
3. **Weekly cycle coverage** in warmup is required to avoid false anomalies on weekends
4. **γ=0.5** provides optimal self-recovery when memory is contaminated by anomalies
