import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\proj\ldt\results\v7\benchmark_results_fast.csv')
print(f'Rows: {len(df)}')
print(f'Errors: {df["error"].notna().sum()}')

if df['error'].notna().any():
    print('\nError rows:')
    print(df[df['error'].notna()][['algorithm','fold','difficulty','seed','error']])

print('\n=== AUC-PR by Algorithm ===')
summary = df.groupby('algorithm')['AUC_PR'].agg(['mean','std','count']).sort_values('mean', ascending=False)
print(summary.to_string())

print('\n=== Batch by Difficulty ===')
batch = df[df['algorithm'].isin(['sklearn_IF','sklearn_OCSVM','DenoisingAE','CA-DIF-EIA','AE+IF','IF-baseline','Random'])]
pivot = batch.pivot_table(index='algorithm', columns='difficulty', values='AUC_PR', aggfunc='mean')
print(pivot.round(4).to_string())

print('\n=== Streaming by Label Budget ===')
stream = df[df['algorithm'].isin(['sHST-River','MemStream','CA-DIF-EIA-Stream','Random'])]
for lb in sorted(stream['label_budget'].unique()):
    sub = stream[stream['label_budget']==lb]
    print(f'\n  Budget={int(lb)}:')
    for algo, row in sub.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).items():
        bar = 100*row/lb if lb>0 else 0
        print(f'    {algo}: AUC_PR={row:.4f}, BAR={bar:.2f}')

# Check hypothesis results
print('\n=== H1 Check: CA-DIF-EIA vs sklearn_IF ===')
ca = df[df['algorithm']=='CA-DIF-EIA']['AUC_PR']
sk = df[df['algorithm']=='sklearn_IF']['AUC_PR']
print(f'  CA-DIF-EIA mean: {ca.mean():.4f} +/- {ca.std():.4f}')
print(f'  sklearn_IF  mean: {sk.mean():.4f} +/- {sk.std():.4f}')
if len(ca) == len(sk):
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(ca.values, sk.values, alternative='greater')
        print(f'  Wilcoxon p: {p:.4f}')
    except:
        pass

print('\n=== H2 Check: AE+IF vs sklearn_IF ===')
ae = df[df['algorithm']=='AE+IF']['AUC_PR']
print(f'  AE+IF  mean: {ae.mean():.4f} +/- {ae.std():.4f}')

print('\n=== H3 Check: CA-DIF-EIA vs AE+IF ===')
if len(ca) == len(ae):
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(ca.values, ae.values, alternative='greater')
        print(f'  Wilcoxon p: {p:.4f}')
    except:
        pass

print('\n=== Random Baseline (calibration) ===')
rnd = df[df['algorithm']=='Random']['AUC_PR']
print(f'  Random AUC-PR mean: {rnd.mean():.4f} (anomaly rate=5%, expected ~0.05)')
