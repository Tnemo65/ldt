import sys, numpy as np, json, os
sys.path.insert(0, r'C:\proj\ldt\HP_benchmark_v5')

print('='*80)
print('ANALYSIS 2: 34D ENGINEERED FEATURES')
print('='*80)

from benchmark_core import extract_features_from_parquet

# Load train data (first 50K)
X_train, hours, dows, rcs, nb = extract_features_from_parquet(
    r'C:\proj\ldt\HP_benchmark_v5\data\train_clean.parquet',
    max_rows=50000
)

print(f'\n=== 34D FEATURE ANALYSIS ===')
print(f'Shape: {X_train.shape}')

feat_names = [
    'dist', 'dur', 'fare', 'pax', 'tot', 'speed',
    'fare_per_mi', 'fare_per_min', 'fare_per_pax', 'hour', 'dow', 'weekend',
    'pux', 'puy', 'dox', 'doy',
    'fpm_c', 'fpm_c2', 'spd_c',
    'dist_per_pax', 'sin_hr', 'cos_hr', 'sin_dw', 'cos_dw',
    'dist_sq', 'rc1', 'rc2', 'rc3', 'rc4', 'rc5',
    'night', 'log_fare', 'log_dist', 'grid_dist'
]

print(f'\n{"F#":>4} {"Name":>14} {"Min":>12} {"Max":>12} {"Mean":>12} {"Std":>12} {"Med":>12} {"P99":>12}')
print('-'*100)
for i in range(34):
    vals = X_train[:, i]
    if np.isnan(vals).any():
        nan_str = f", NAN={np.isnan(vals).sum()}"
    else:
        nan_str = ''
    print(f'{i:4d} {feat_names[i]:>14} {vals.min():12.3f} {vals.max():12.3f} {vals.mean():12.3f} {vals.std():12.3f} {np.median(vals):12.3f} {np.percentile(vals,99):12.3f}{nan_str}')

print(f'\n=== FEATURES WITH EXTREME RANGES ===')
for i in range(34):
    vals = X_train[:, i]
    min_v, max_v = vals.min(), vals.max()
    range_ratio = max_v / max(min_v, 1e-8) if min_v > 0 else (max_v - min_v) / max(abs(min_v), 1e-8)
    nan_count = np.isnan(vals).sum()
    inf_count = np.isinf(vals).sum()
    if range_ratio > 1000 or nan_count > 0 or inf_count > 0:
        print(f'F{i:02d} {feat_names[i]}: range_ratio={range_ratio:.0f}, nan={nan_count}, inf={inf_count}, min={min_v:.3f}, max={max_v:.3f}')

print(f'\n=== FEATURE CORRELATION WITH KEY ANOMALY SIGNALS ===')
# Type 1: fare x 5-15 (short_expensive) -> high fare_per_mile, high log_fare
# Type 2: tip = fare x 10-20 -> tip_amount would be high
# Type 4: dist x 0.05-0.3 + dur x 2-5 -> very short distance, long duration

print('Fare amount range:', X_train[:, 2].min(), '-', X_train[:, 2].max())
print('Trip distance range:', X_train[:, 0].min(), '-', X_train[:, 0].max())
print('Duration range:', X_train[:, 1].min(), '-', X_train[:, 1].max())

# Check feature F6 (fare_per_mile)
fpm = X_train[:, 6]
print(f'\nFare per mile (F6): min={fpm.min():.3f}, max={fpm.max():.3f}, mean={fpm.mean():.3f}')
print(f'  F6 > 10: {(fpm > 10).sum()} samples ({100*(fpm > 10).mean():.2f}%)')
print(f'  F6 > 20: {(fpm > 20).sum()} samples ({100*(fpm > 20).mean():.2f}%)')
print(f'  F6 > 50: {(fpm > 50).sum()} samples ({100*(fpm > 50).mean():.2f}%)')

# Check F31 (log_fare)
log_fare = X_train[:, 31]
print(f'\nLog fare (F31): min={log_fare.min():.3f}, max={log_fare.max():.3f}, mean={log_fare.mean():.3f}')

# Check F0 (dist)
dist = X_train[:, 0]
print(f'\nTrip distance (F0): min={dist.min():.3f}, max={dist.max():.3f}, mean={dist.mean():.3f}')
print(f'  dist < 0.3: {(dist < 0.3).sum()} samples ({100*(dist < 0.3).mean():.2f}%)')
print(f'  dist < 1.0: {(dist < 1.0).sum()} samples ({100*(dist < 1.0).mean():.2f}%)')
