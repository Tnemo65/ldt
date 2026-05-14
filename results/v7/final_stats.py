"""Compute final stats from completed benchmark checkpoint."""
import sys, warnings
sys.path.insert(0, r'C:\proj\ldt\results\v7')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
import benchmark_v7 as bm

CHECKPOINT = r'C:\proj\ldt\results\v7\checkpoint_v7.csv'
results = pd.read_csv(CHECKPOINT)

print(f"Total rows: {len(results)}")
errors = results['error'].notna().sum()
print(f"Errors: {errors}")
if errors > 0:
    print(results[results['error'].notna()][['algorithm','error']].drop_duplicates())

print('\n=== Summary by Algorithm (AUC-PR) ===')
summary = results.groupby('algorithm')['AUC_PR'].agg(['mean','std','min','max','count']).sort_values('mean', ascending=False)
print(summary.round(4).to_string())

print('\n=== AUC-PR by Algorithm and Difficulty ===')
pivot = results.pivot_table(index='algorithm', columns='difficulty', values='AUC_PR', aggfunc='mean')
pivot = pivot[['easy', 'medium', 'hard']].sort_values('hard', ascending=False)
print(pivot.round(4).to_string())

print('\n=== AUC-PR by Algorithm and Label Budget (streaming) ===')
stream = results[results['algorithm'].isin(['sHST-River', 'MemStream', 'CA-DIF-EIA-Stream', 'Random'])]
for lb in sorted(stream['label_budget'].unique()):
    sub = stream[stream['label_budget'] == lb]
    print(f'\n  Budget {int(lb)}:')
    for algo, row in sub.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).items():
        print(f'    {algo}: AUC_PR={row:.4f}')

print('\n=== BAR Score (Streaming, non-Random) ===')
stream_nr = results[results['algorithm'].isin(['sHST-River', 'MemStream', 'CA-DIF-EIA-Stream']) & (results['label_budget'] > 0)]
for lb in sorted(stream_nr['label_budget'].unique()):
    sub = stream_nr[stream_nr['label_budget'] == lb]
    print(f'\n  Budget {int(lb)}:')
    for algo, row in sub.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).items():
        bar = 100 * row / lb if lb > 0 else 0
        print(f'    {algo}: AUC_PR={row:.4f}, BAR={bar:.4f}')

print('\n=== Streaming: AUC-PR over Label Budget ===')
stream2 = results[results['algorithm'].isin(['sHST-River', 'MemStream', 'CA-DIF-EIA-Stream'])]
stream_pivot = stream2.pivot_table(index='label_budget', columns='algorithm', values='AUC_PR', aggfunc='mean')
stream_pivot = stream_pivot[['MemStream', 'CA-DIF-EIA-Stream', 'sHST-River']]
print(stream_pivot.round(4).to_string())

# Friedman test for batch algorithms
print('\n=== Friedman Test: Batch Algorithms ===')
batch = results[~results['algorithm'].isin(['Random', 'sHST-River', 'MemStream', 'CA-DIF-EIA-Stream'])]
algos_batch = ['DenoisingAE', 'AE+IF', 'CA-DIF-EIA', 'IF-baseline', 'sklearn_IF', 'sklearn_OCSVM']
try:
    pivot_batch = batch.pivot_table(index=['fold', 'difficulty'], columns='algorithm', values='AUC_PR')
    pivot_batch = pivot_batch[algos_batch].dropna()
    groups_batch = [pivot_batch[a].dropna().values for a in algos_batch]
    stat_b, p_b = friedmanchisquare(*groups_batch)
    print(f"  Friedman chi-square={stat_b:.4f}, p={p_b:.2e}, significant={p_b < 0.05}")
    for i, b1 in enumerate(algos_batch):
        for b2 in algos_batch[i+1:]:
            try:
                t, p = wilcoxon(pivot_batch[b1].values, pivot_batch[b2].values, zero_method='pratt')
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                print(f"    {b1} vs {b2}: p={p:.4f} {sig}")
            except:
                pass
except Exception as e:
    print(f"  Error: {e}")

# Friedman test for streaming algorithms (at label_budget=500)
print('\n=== Friedman Test: Streaming (budget=500) ===')
stream500 = results[results['label_budget'] == 500]
stream500 = stream500[stream500['algorithm'].isin(['sHST-River', 'MemStream', 'CA-DIF-EIA-Stream'])]
algos_stream = ['MemStream', 'CA-DIF-EIA-Stream', 'sHST-River']
try:
    pivot_s = stream500.pivot_table(index=['fold', 'difficulty'], columns='algorithm', values='AUC_PR')
    pivot_s = pivot_s[algos_stream].dropna()
    groups_s = [pivot_s[a].dropna().values for a in algos_stream]
    stat_s, p_s = friedmanchisquare(*groups_s)
    print(f"  Friedman chi-square={stat_s:.4f}, p={p_s:.2e}, significant={p_s < 0.05}")
    for algo in algos_stream:
        median = pivot_s[algo].median()
        print(f"    {algo}: median={median:.4f}")
    for i, b1 in enumerate(algos_stream):
        for b2 in algos_stream[i+1:]:
            try:
                t, p = wilcoxon(pivot_s[b1].values, pivot_s[b2].values, zero_method='pratt')
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                print(f"    {b1} vs {b2}: p={p:.4f} {sig}")
            except:
                pass
except Exception as e:
    print(f"  Error: {e}")

print('\n=== CA-DIF-EIA Ablation Study ===')
ablation = results[results['algorithm'].isin(['CA-DIF-EIA', 'AE+IF', 'IF-baseline', 'DenoisingAE'])]
ab_pivot = ablation.pivot_table(index='algorithm', columns='difficulty', values='AUC_PR', aggfunc='mean')
ab_pivot = ab_pivot[['easy', 'medium', 'hard']].sort_values('hard', ascending=False)
print(ab_pivot.round(4).to_string())

print('\nDONE!')
