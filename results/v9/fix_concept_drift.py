data = open('results/v9/run_concept_drift_v9.py', 'r', encoding='utf-8').read()
data = data.replace('\u2500', '-').replace('\u2014', '--').replace('\u00d7', '*').replace('\u2192', '->')
open('results/v9/run_concept_drift_v9.py', 'w', encoding='utf-8').write(data)
print('Fixed')
