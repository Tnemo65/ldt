import sys, numpy as np, json
sys.path.insert(0, r'C:\proj\ldt\HP_benchmark_v5')

print('='*80)
print('ANALYSIS 6: INJECTION LOG & ACTUAL ANOMALY FEATURES')
print('='*80)

# Load injection log
inj_log = json.load(open(r'C:\proj\ldt\HP_benchmark_v5\data\valid\injection_log.json'))
print(f'Injection log keys: {list(inj_log.keys())}')
print(f'type_counts: {inj_log.get("type_counts")}')
print(f'actual_ratio_pct: {inj_log.get("actual_ratio_pct")}')
print(f'total_target: {inj_log.get("total_target")}')
print(f'total_confirmed: {inj_log.get("total_confirmed")}')

# Load the inject script to understand the rules
print(f'\n=== INJECTION SCRIPT ===')
with open(r'C:\proj\ldt\HP_benchmark_v5\data\inject_anomalies_memstream.py') as f:
    content = f.read()
    
# Find the injection rules
import re
# Find type definitions
type_sections = re.findall(r'Type (\d+|[\w]+)[:\s]+(.*?)(?=\n\n|class |def |Type |\Z)', content, re.DOTALL)
for t, desc in type_sections[:10]:
    print(f'Type {t}: {desc[:200].strip()}...')
    print()

# Also check the injection logic
print(f'\n=== KEY INJECTION FUNCTIONS ===')
funcs = re.findall(r'def inject_(\w+)', content)
print(f'Injection functions: {funcs}')

# Show the actual inject functions
for func in funcs[:5]:
    pattern = rf'def inject_{func}[^:]+:(.*?)(?=\ndef |class |\Z)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        print(f'\n--- inject_{func} ---')
        print(match.group(0)[:500])
