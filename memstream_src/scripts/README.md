# CA-DQStream + MemStream Evaluation Scripts

Evaluation and testing suite for the CA-DQStream + MemStream hybrid anomaly detection system.

## Scripts

### Training
- `train_warmup.py` - Time-ordered training pipeline with leakage-free warmup

### Evaluation
- `eval_streaming.py` - Streaming evaluation with latency tracking
- `eval_ablation.py` - Ablation study: 25D vs 40D context-aware
- `eval_bar_score.py` - BAR Score analysis (target: 1-5%)
- `eval_false_alarms.py` - False alarm analysis per context
- `benchmark_hybrid.py` - Hybrid vs baseline comparison

### Utilities
- `inject_anomalies_multi.py` - Multi-strategy anomaly injection

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_memstream_core.py -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html
```

## Key Features

- **C-DE-1**: Time-ordered splits (no random shuffle)
- **C-DE-2**: Leakage-free warmup (stats from first 10%, train middle 80%, memory last 10%)
- **C-ML-1**: max_thres initialized in `__init__`
- **BAR Controller**: 1-5% budget allocation rate
- **4D Context**: Neighborhood + Hour + Day + Trip type embeddings
