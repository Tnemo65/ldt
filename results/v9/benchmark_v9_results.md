# Benchmark v9 -- CA-MemStream-EIA Ablation Study

**Generated:** by benchmark_v9.py
**Source:** checkpoint_v9.csv (2205 rows)

## Primary Results (medium, budget=500)

                    AUC_PR                 BAR           Precision    Recall        F1
                      mean       std      mean       std      mean      mean      mean
algorithm                                                                             
MemStream         0.196154  0.038629  0.189826  0.037383  0.237778  0.237778  0.237778
Canary-Rules      0.161724  0.020681  0.156507  0.020014  0.161553  0.119556  0.133149
DenoisingAE       0.113812  0.018653  0.110141  0.018051  0.132978  0.132978  0.132978
Random            0.049516  0.001699  0.047919  0.001644  0.051111  0.051111  0.051111
sHST-River        0.048422  0.002690  0.046860  0.002603  0.049966  0.049422  0.049689
CA-MemStream           NaN       NaN       NaN       NaN       NaN       NaN       NaN
CA-MemStream-EIA       NaN       NaN       NaN       NaN       NaN       NaN       NaN

## BAR Score by Budget


### MemStream
  budget=   0: BAR=0.196154
  budget=  50: BAR=0.195502
  budget= 100: BAR=0.194855
  budget= 250: BAR=0.192938
  budget= 500: BAR=0.189826
  budget=1000: BAR=0.183894
  budget=2000: BAR=0.173077

### CA-MemStream
  budget=   0: BAR=nan
  budget=  50: BAR=nan
  budget= 100: BAR=nan
  budget= 250: BAR=nan
  budget= 500: BAR=nan
  budget=1000: BAR=nan
  budget=2000: BAR=nan

### CA-MemStream-EIA
  budget=   0: BAR=nan
  budget=  50: BAR=nan
  budget= 100: BAR=nan
  budget= 250: BAR=nan
  budget= 500: BAR=nan
  budget=1000: BAR=nan
  budget=2000: BAR=nan
