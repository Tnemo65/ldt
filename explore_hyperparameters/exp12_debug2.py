#!/usr/bin/env python3
import sys, os, json
sys.stderr.write("Starting exp12_debug\n")
sys.stderr.flush()

ROOT = __import__('pathlib').Path('.')
sys.path.insert(0, str(ROOT))
sys.stderr.write(f"ROOT: {ROOT}\n")
sys.stderr.flush()

import warnings
warnings.filterwarnings('ignore')
sys.stderr.write("warnings done\n")
sys.stderr.flush()

import numpy as np
sys.stderr.write("numpy done\n")
sys.stderr.flush()

import torch
sys.stderr.write(f"torch done, cuda={torch.cuda.is_available()}\n")
sys.stderr.flush()

import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from shared import load_data, GPUExperimentModel
sys.stderr.write("imports done\n")
sys.stderr.flush()

OUT = ROOT / 'results'
sys.stderr.write(f"OUT: {OUT.resolve()}\n")
sys.stderr.flush()

try:
    OUT.mkdir(exist_ok=True)
    sys.stderr.write("mkdir done\n")
    sys.stderr.flush()
except Exception as e:
    sys.stderr.write(f"mkdir failed: {e}\n")
    sys.stderr.flush()

data = load_data(n_warmup=10000, n_test=15000)
sys.stderr.write(f"data loaded: warmup={data['n_warmup']}\n")
sys.stderr.flush()

model = GPUExperimentModel(memory_len=256, k=10, gamma=0.0, latent_dim=60, default_beta=0.5, seed=42, device='cuda')
sys.stderr.write("model created\n")
sys.stderr.flush()

model.fit(data['X_warmup'], neighborhood_ids=data['nb_warmup'],
          hour_vals=data['hr_warmup'], dow_vals=data['dw_warmup'],
          ratecode_vals=data['rc_warmup'], epochs=20, batch_size=256)
sys.stderr.write("fit done\n")
sys.stderr.flush()

# Save immediately after fit
result_path = OUT / 'exp12_debug_result.json'
with open(result_path, 'w') as f:
    json.dump({'step': 'fit_done', 'cwd': os.getcwd(), 'out': str(OUT.resolve())}, f)
sys.stderr.write(f"Saved to {result_path}\n")
sys.stderr.flush()
print(f"Saved: {result_path}")
sys.stderr.write("DONE\n")
sys.stderr.flush()
