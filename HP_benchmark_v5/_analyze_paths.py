import sys, numpy as np, json, os
sys.path.insert(0, r'C:\proj\ldt\HP_benchmark_v5')

# Check data paths
data_dir = r'C:\proj\ldt\HP_benchmark_v5\data'
valid_dir = os.path.join(data_dir, 'valid')
print('=== DATA PATHS ===')
print('data_dir:', data_dir)
print('  exists:', os.path.exists(data_dir))
print('  valid_dir:', valid_dir)
print('  valid_dir exists:', os.path.exists(valid_dir))

# Check parent dir
parent = r'C:\proj\ldt\HP_benchmark_v4\data'
print('parent data_dir:', parent)
print('  exists:', os.path.exists(parent))

# Try to find parquet files
found = []
for root, dirs, files in os.walk(r'C:\proj\ldt'):
    for f in files:
        if f.endswith('.parquet') or f.endswith('.npy'):
            found.append(os.path.join(root, f))
print('\n=== FOUND FILES ===')
for f in found[:30]:
    print(f)
if len(found) > 30:
    print(f'... and {len(found)-30} more')
if not found:
    print('No parquet or npy files found!')
