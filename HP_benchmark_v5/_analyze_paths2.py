import sys, numpy as np, json, os
sys.path.insert(0, r'C:\proj\ldt\HP_benchmark_v5')

# Check data files in HP_benchmark_v5
data_dir = r'C:\proj\ldt\HP_benchmark_v5\data'
print('=== HP_benchmark_v5/data ===')
if os.path.exists(data_dir):
    for f in os.listdir(data_dir):
        fp = os.path.join(data_dir, f)
        if os.path.isdir(fp):
            print(f'  [DIR] {f}/')
            for sf in os.listdir(fp):
                sfp = os.path.join(fp, sf)
                size = os.path.getsize(sfp) / 1024 / 1024
                print(f'    {sf} ({size:.1f} MB)')
        else:
            size = os.path.getsize(fp) / 1024 / 1024
            print(f'  {f} ({size:.1f} MB)')
else:
    print('  NOT FOUND')

# Check parent dir
parent = r'C:\proj\ldt\HP_benchmark_v4\data'
print('\n=== HP_benchmark_v4/data ===')
if os.path.exists(parent):
    for f in os.listdir(parent):
        fp = os.path.join(parent, f)
        if os.path.isdir(fp):
            print(f'  [DIR] {f}/')
            for sf in os.listdir(fp)[:5]:
                sfp = os.path.join(fp, sf)
                size = os.path.getsize(sfp) / 1024 / 1024
                print(f'    {sf} ({size:.1f} MB)')
        else:
            size = os.path.getsize(fp) / 1024 / 1024
            print(f'  {f} ({size:.1f} MB)')
else:
    print('  NOT FOUND')
