# EDA Findings - January 2024

## Dataset
- Records: 2,964,624
- Memory: 0.71 GB
- Columns: 22

## Null Rates (Top 5)
|                      |   Null Rate (%) |
|:---------------------|----------------:|
| Airport_fee          |         4.72782 |
| congestion_surcharge |         4.72782 |
| passenger_count      |         4.72782 |
| RatecodeID           |         4.72782 |
| store_and_fwd_flag   |         4.72782 |

## Spatial
- Zones: 260/263
- Coverage: 98.9%

## Temporal
- Peak hour: 18:00
- Busiest day: 2024-01-27

## Outliers
|                   |   count |
|:------------------|--------:|
| negative_fare     |   37448 |
| negative_distance |       0 |
| zero_distance     |   60371 |
| extreme_distance  |      59 |
| extreme_fare      |      46 |
| passenger_zero    |   31465 |
| passenger_high    |      60 |

## Business Rules
- Violation rate: 7.93%

## Recommendations
1. Baseline sanitization (Task 0.5) CRITICAL
2. Neighborhood mapping (Task 0.4)
3. Synthetic anomalies (Task 0.6)
