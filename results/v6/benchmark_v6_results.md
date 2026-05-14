# Benchmark v6 Results — Scientific Rigour Overhaul

**Generated:** 2026-05-12T17:27:29.211485

**Protocol:** Three-way split (train/val/test), threshold from validation set

**Datasets:** NYC Yellow Taxi Jan-Jun 2024

**Seeds:** [42, 123, 456]

**Difficulties:** ['easy', 'medium', 'hard']

**Anomaly Rate:** 15% (1500 anomalies / 10000 test samples)

**Label Budgets:** [0, 500]


## Summary: Mean AUC-PR by Algorithm

| Algorithm | Mean AUC-PR | Std | N |
|-----------|-------------|-----|---|
| sHST-River | 0.3631 | 0.2050 | 36 |
| CA-DIF-EIA (streaming) | 0.1351 | 0.0057 | 36 |
| sklearn_IF | 0.1321 | 0.0047 | 18 |
| IF-baseline | 0.1321 | 0.0048 | 18 |
| DenoisingAE | 0.1308 | 0.0051 | 18 |
| AE+IF | 0.1293 | 0.0019 | 18 |
| CA-DIF-EIA | 0.1285 | 0.0014 | 18 |
| MemStream | 0.1168 | 0.0163 | 36 |

## AUC-PR by Difficulty

| Algorithm | Easy | Medium | Hard |
|-----------|------|--------|------|
| sHST-River | 0.3627 | 0.3628 | 0.3637 |
| CA-DIF-EIA (streaming) | 0.1345 | 0.1353 | 0.1356 |
| sklearn_IF | 0.1313 | 0.1323 | 0.1328 |
| IF-baseline | 0.1313 | 0.1323 | 0.1328 |
| DenoisingAE | 0.1300 | 0.1314 | 0.1310 |
| AE+IF | 0.1299 | 0.1283 | 0.1298 |
| CA-DIF-EIA | 0.1289 | 0.1275 | 0.1291 |
| MemStream | 0.1167 | 0.1171 | 0.1166 |

## BAR Score by Streaming Algorithm


### Label Budget = 0

- sHST-River: AUC-PR=0.5652, Labels=0, BAR=56.5217
- MemStream: AUC-PR=0.1323, Labels=0, BAR=13.2263
- CA-DIF-EIA (streaming): AUC-PR=0.1350, Labels=0, BAR=13.4991

### Label Budget = 500

- sHST-River: AUC-PR=0.1609, Labels=500, BAR=0.0322
- MemStream: AUC-PR=0.1013, Labels=500, BAR=0.0203
- CA-DIF-EIA (streaming): AUC-PR=0.1353, Labels=500, BAR=0.0271

## Statistical Analysis


### Batch

- Friedman test: stat=1.6000000000000014, p=0.8087921354109986
- Significant: False
  No significant differences (Friedman p=0.8088 >= 0.05).

### Batch_easy

- Friedman test: stat=2.0, p=0.7357588823428847
- Significant: False
  No significant differences (Friedman p=0.7358 >= 0.05).

### Batch_medium

- Friedman test: stat=1.6000000000000014, p=0.8087921354109986
- Significant: False
  No significant differences (Friedman p=0.8088 >= 0.05).

### Batch_hard

- Friedman test: stat=1.2000000000000028, p=0.8780986177504418
- Significant: False
  No significant differences (Friedman p=0.8781 >= 0.05).

### Streaming_500

- Friedman test: stat=4.0, p=0.1353352832366127
- Significant: False
  No significant differences (Friedman p=0.1353 >= 0.05).