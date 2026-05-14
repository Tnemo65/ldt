# Benchmark v8 — MemStream Scientific Correction + Concept Drift

**Generated:** 2026-05-13T07:56:46
**Source:** checkpoint_v8.csv (675 rows)
**Folds:** Leave-one-month-out (5 folds)
**Seeds:** [42, 123, 456]
**Difficulties:** ['easy', 'medium', 'hard']
**Anomaly Rate:** 5% (500 / 10000)

## Summary: Mean AUC-PR by Algorithm

| Algorithm | Mean AUC-PR | Std | N |
|-----------|-------------|-----|---|
| DenoisingAE | 0.9995 | 0.0006 | 45 |
| CA-DIF-EIA-Stream | 0.9992 | 0.0010 | 90 |
| MemStream | 0.9988 | 0.0031 | 90 |
| AE+IF | 0.9984 | 0.0017 | 45 |
| CA-DIF-EIA | 0.9252 | 0.0277 | 45 |
| IF-baseline | 0.8043 | 0.0817 | 45 |
| sklearn_IF | 0.7905 | 0.0940 | 45 |
| sHST-River | 0.2262 | 0.0759 | 90 |
| Random | 0.0468 | 0.0024 | 135 |
| sklearn_OCSVM | 0.0239 | 0.0001 | 45 |

## AUC-PR by Difficulty

| Algorithm | EASY | MEDIUM | HARD |
|-----------|------|--------|------|
| AE+IF | 0.9994 | 0.9978 | 0.9980 |
| CA-DIF-EIA | 0.9275 | 0.9218 | 0.9263 |
| CA-DIF-EIA-Stream | 1.0000 | 0.9990 | 0.9987 |
| DenoisingAE | 1.0000 | 0.9993 | 0.9993 |
| IF-baseline | 0.8042 | 0.7423 | 0.8663 |
| MemStream | 1.0000 | 0.9989 | 0.9974 |
| Random | 0.0468 | 0.0468 | 0.0468 |
| sHST-River | 0.2624 | 0.2025 | 0.2138 |
| sklearn_IF | 0.7927 | 0.7312 | 0.8477 |
| sklearn_OCSVM | 0.0238 | 0.0240 | 0.0240 |

## AUC-PR by Fold

| Algorithm | 1 | 2 | 3 | 4 | 5 |
|-----------|---|---|---|---|---|
| AE+IF | 0.9971 | 0.9999 | 0.9984 | 0.9981 | 0.9985 |
| CA-DIF-EIA | 0.8856 | 0.9012 | 0.9379 | 0.9504 | 0.9510 |
| CA-DIF-EIA-Stream | 0.9993 | 0.9998 | 0.9993 | 0.9995 | 0.9983 |
| DenoisingAE | 0.9994 | 0.9999 | 0.9998 | 0.9996 | 0.9988 |
| IF-baseline | 0.7900 | 0.7620 | 0.7931 | 0.8756 | 0.8008 |
| MemStream | 0.9994 | 0.9961 | 0.9998 | 0.9997 | 0.9988 |
| Random | 0.0475 | 0.0447 | 0.0494 | 0.0475 | 0.0447 |
| sHST-River | 0.2488 | 0.2034 | 0.2019 | 0.2230 | 0.2541 |
| sklearn_IF | 0.7983 | 0.7137 | 0.7686 | 0.8943 | 0.7778 |
| sklearn_OCSVM | 0.0239 | 0.0239 | 0.0239 | 0.0240 | 0.0239 |

## Streaming: AUC-PR by Label Budget

| Algorithm | Budget=0 | Budget=500 |
|-----------|----------|------------|
| CA-DIF-EIA-Stream | 0.9992 | 0.9992 |
| MemStream | 0.9988 | 0.9988 |
| Random | 0.0467 | 0.0468 |
| sHST-River | 0.2379 | 0.2145 |

## Statistical Analysis

### Batch
- Friedman: stat=87.714, p=0.0000 (**SIGNIFICANT**)
- Average ranks:
  - 2: 1.00
  - 0: 2.00
  - 1: 3.40
  - 6: 3.60
  - 4: 5.00
  - 5: 6.13
  - 3: 6.87
  - CA-DIF-EIA vs sklearn_IF: p_raw=0.0000, p_holm=0.0002, sig=**YES**
  - CA-DIF-EIA vs sklearn_OCSVM: p_raw=0.0000, p_holm=0.0001, sig=**YES**
  - CA-DIF-EIA vs IF-baseline: p_raw=0.0000, p_holm=0.0001, sig=**YES**
  - CA-DIF-EIA vs DenoisingAE: p_raw=1.0000, p_holm=1.0000, sig=no
  - CA-DIF-EIA vs AE+IF: p_raw=1.0000, p_holm=1.0000, sig=no

### Streaming_500
- Friedman: stat=41.554, p=0.0000 (**SIGNIFICANT**)
- Average ranks:
  - 0: 1.00
  - 1: 2.00
  - 3: 3.33
  - 2: 3.67

### All_Streaming
- Friedman: stat=41.554, p=0.0000 (**SIGNIFICANT**)
- Average ranks:
  - 0: 1.00
  - 1: 2.00
  - 3: 3.33
  - 2: 3.67