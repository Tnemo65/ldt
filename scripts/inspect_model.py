import pickle
try:
    with open(r'c:\proj\ldt\scripts\memstream_memory.pt', 'rb') as f:
        data = f.read()
    print(f'File size: {len(data):,} bytes')
    # Try unpickling
    import io
    result = pickle.loads(data)
    print(f'Loaded successfully: {type(result)}')
    if isinstance(result, dict):
        print(f'Keys: {list(result.keys())}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
