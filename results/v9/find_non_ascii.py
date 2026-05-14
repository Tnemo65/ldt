import sys
data = open('results/v9/benchmark_v9.py', 'r', encoding='utf-8').read()
bad = {}
for i, c in enumerate(data):
    if ord(c) > 127:
        line = data[:i].count('\n') + 1
        bad.setdefault((c, ord(c), line), []).append(i)

with open('results/v9/non_ascii.txt', 'w', encoding='utf-8') as f:
    for (c, code, line), positions in sorted(bad.items(), key=lambda x: x[0][2]):
        f.write(f'Line {line}: U+{code:04X} {repr(c)}\n')

print(f'Found {len(bad)} unique non-ASCII characters')
