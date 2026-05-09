import pickle

with open('c:/proj/ldt/models/iforest_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f'Type: {type(model)}')
print(f'Module: {type(model).__module__}')
print(f'Class: {type(model).__name__}')

if hasattr(model, 'n_estimators'):
    print(f'n_estimators: {model.n_estimators}')
if hasattr(model, 'window_size'):
    print(f'window_size: {model.window_size}')
if hasattr(model, 'max_depth'):
    print(f'max_depth: {model.max_depth}')
if hasattr(model, 'n_trees'):
    print(f'n_trees: {model.n_trees}')
if hasattr(model, 'seed'):
    print(f'seed: {model.seed}')

attrs = [a for a in dir(model) if not a.startswith('_')]
print(f'Attrs: {attrs[:30]}')

# Try scoring
import numpy as np
test_data = np.random.randn(10, 21)
try:
    score = model.score(test_data)
    print(f'score(10x21): {score}')
except Exception as e:
    print(f'score error: {e}')

try:
    pred = model.predict_one(test_data[0])
    print(f'predict_one: {pred}')
except Exception as e:
    print(f'predict_one error: {e}')

try:
    score2 = model.score_one(test_data[0])
    print(f'score_one: {score2}')
except Exception as e:
    print(f'score_one error: {e}')
