# Benchmark v10 Results

## Overview

- Device: cuda
- Algorithms: ['MemStream', 'CA-MemStream', 'CA-MemStream-EIA', 'Canary-Rules', 'Random']
- Fraud types: ['canary_only', 'type1_only', 'type2_only', 'type3_only', 'mixed', 'hybrid']
- Features: 30D (25D original + 5 RatecodeID one-hot)
- ADWIN: 10 neighborhood-level instances
- Context-beta: 80 thresholds
- JFK flat fare: $70.00

## AUC-PR by Algorithm and Fraud Type

                              AUC_PR     BAR  Precision  Recall      F1  drift_count  update_count
fraud_type  algorithm                                                                             
canary_only CA-MemStream      0.1475  0.1475     0.2160  0.2222  0.2190          0.0           0.0
            CA-MemStream-EIA  0.1354  0.1354     0.1945  0.2007  0.1975         44.1          44.1
            Canary-Rules      1.0000  1.0000     0.4000  0.4000  0.4000          0.0           0.0
            MemStream         0.1467  0.1467     0.2147  0.2208  0.2176          0.0           0.0
            Random            0.0491  0.0491     0.0504  0.0519  0.0511          0.0           0.0
hybrid      CA-MemStream      0.9013  0.9013     0.8533  0.8723  0.8627          0.0           0.0
            CA-MemStream-EIA  0.8879  0.8879     0.8344  0.8531  0.8436         28.3          28.3
            Canary-Rules         NaN     NaN     0.0000  0.0000  0.0000          0.0           0.0
            MemStream         0.9001  0.9001     0.8529  0.8719  0.8623          0.0           0.0
            Random            0.0496  0.0496     0.0515  0.0526  0.0520          0.0           0.0
mixed       CA-MemStream      0.9251  0.9251     0.8656  0.9052  0.8850          0.0           0.0
            CA-MemStream-EIA  0.9207  0.9207     0.8639  0.9034  0.8832         49.7          49.7
            Canary-Rules         NaN     NaN     0.0000  0.0000  0.0000          0.0           0.0
            MemStream         0.9249  0.9249     0.8660  0.9057  0.8854          0.0           0.0
            Random            0.0491  0.0491     0.0544  0.0569  0.0556          0.0           0.0
type1_only  CA-MemStream      0.0688  0.0688     0.0503  0.0503  0.0503          0.0           0.0
            CA-MemStream-EIA  0.0778  0.0778     0.0643  0.0643  0.0643         44.1          44.1
            Canary-Rules         NaN     NaN     0.0000  0.0000  0.0000          0.0           0.0
            MemStream         0.0683  0.0683     0.0536  0.0536  0.0536          0.0           0.0
            Random            0.0510  0.0510     0.0552  0.0552  0.0552          0.0           0.0
type2_only  CA-MemStream      0.0674  0.0674     0.0215  0.0215  0.0215          0.0           0.0
            CA-MemStream-EIA  0.0771  0.0771     0.0304  0.0304  0.0304         44.1          44.1
            Canary-Rules         NaN     NaN     0.0000  0.0000  0.0000          0.0           0.0
            MemStream         0.0674  0.0674     0.0239  0.0239  0.0239          0.0           0.0
            Random            0.0516  0.0516     0.0544  0.0544  0.0544          0.0           0.0
type3_only  CA-MemStream      0.0833  0.0833     0.0555  0.0555  0.0555          0.0           0.0
            CA-MemStream-EIA  0.1152  0.1152     0.0879  0.0879  0.0879         44.1          44.1
            Canary-Rules         NaN     NaN     0.0000  0.0000  0.0000          0.0           0.0
            MemStream         0.0894  0.0894     0.0633  0.0633  0.0633          0.0           0.0
            Random            0.0512  0.0512     0.0533  0.0533  0.0533          0.0           0.0

## Drift/Update counts

                  drift_count  update_count
algorithm                                  
CA-MemStream              0.0           0.0
CA-MemStream-EIA         42.4          42.4
Canary-Rules              0.0           0.0
MemStream                 0.0           0.0
Random                    0.0           0.0