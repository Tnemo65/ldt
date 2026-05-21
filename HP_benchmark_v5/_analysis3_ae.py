import sys, numpy as np
sys.path.insert(0, r'C:\proj\ldt\HP_benchmark_v5')

print('='*80)
print('ANALYSIS 3: AUTOENCODER ENCODING ANALYSIS')
print('='*80)

import torch
from benchmark_core import MemStreamPipeline, extract_features_from_parquet, DEVICE

# Load training data
X_train, hours, dows, rcs, nb = extract_features_from_parquet(
    r'C:\proj\ldt\HP_benchmark_v5\data\train_clean.parquet',
    max_rows=20000
)

# Train a minimal AE to analyze encoding
pipe = MemStreamPipeline(
    d=34, out_dim=68, memory_len=512, k=3,
    gamma=0.0, beta=0.5, noise_std=0.001,
    lr=0.01, epochs=100, batch_size=1024,
    verbose=False
)
pipe.train(X_train, hours, dows, rcs, nb)

# Analyze input normalization
print(f'\n=== INPUT NORMALIZATION ===')
print(f'Input mean (first 5): {pipe._input_mean[:5]}')
print(f'Input std  (first 5): {pipe._input_std[:5]}')
print(f'Input std range: {pipe._input_std.min():.6f} - {pipe._input_std.max():.6f}')

# Analyze encoder weights
enc_weight = pipe.ae.encoder.weight.detach().cpu().numpy()
enc_bias = pipe.ae.encoder.bias.detach().cpu().numpy()
print(f'\n=== ENCODER WEIGHTS ===')
print(f'Encoder weight shape: {enc_weight.shape}')
print(f'Encoder weight range: {enc_weight.min():.4f} to {enc_weight.max():.4f}')
print(f'Encoder weight mean: {enc_weight.mean():.4f}, std: {enc_weight.std():.4f}')
print(f'Encoder bias range: {enc_bias.min():.4f} to {enc_bias.max():.4f}')

# Analyze encoded representations
X_test = X_train[:1000].astype(np.float32)
Xn = (X_test - pipe._input_mean) / (pipe._input_std + 1e-8)
Xn_t = torch.from_numpy(Xn).to(DEVICE)

with torch.no_grad():
    Z = pipe.ae.encode(Xn_t).cpu().numpy()

print(f'\n=== ENCODED REPRESENTATIONS ===')
print(f'Z shape: {Z.shape}')
print(f'Z min: {Z.min():.6f}, max: {Z.max():.6f}')
print(f'Z mean: {Z.mean():.6f}, std: {Z.std():.6f}')

# Check saturation (Tanh output should be in [-1, 1])
print(f'\n=== TANH SATURATION ANALYSIS ===')
print(f'Z values in [-0.9, 0.9]: {((Z >= -0.9) & (Z <= 0.9)).mean()*100:.1f}%')
print(f'Z values in [-0.5, 0.5]: {((Z >= -0.5) & (Z <= 0.5)).mean()*100:.1f}%')
print(f'Z values at extreme ([-0.99, 0.99]): {((Z >= -0.99) & (Z <= 0.99)).mean()*100:.1f}%')
print(f'Z values saturated (< -0.99 or > 0.99): {((Z < -0.99) | (Z > 0.99)).mean()*100:.1f}%')
print(f'Z values saturated (< -0.999 or > 0.999): {((Z < -0.999) | (Z > 0.999)).mean()*100:.1f}%')

# Check raw encoder output (before Tanh)
with torch.no_grad():
    raw = pipe.ae.encoder(Xn_t).cpu().numpy()

print(f'\n=== BEFORE TANH (raw linear output) ===')
print(f'Raw min: {raw.min():.4f}, max: {raw.max():.4f}')
print(f'Raw mean: {raw.mean():.4f}, std: {raw.std():.4f}')
print(f'Raw abs > 1: {(np.abs(raw) > 1).mean()*100:.1f}%')
print(f'Raw abs > 3: {(np.abs(raw) > 3).mean()*100:.1f}%')
print(f'Raw abs > 5: {(np.abs(raw) > 5).mean()*100:.1f}%')

# Analyze reconstruction loss per dimension
with torch.no_grad():
    recon = pipe.ae(Xn_t).cpu().numpy()

recon_err = np.abs(recon - Xn).mean(axis=0)
print(f'\n=== RECONSTRUCTION ERROR PER DIM (first 10) ===')
for i in range(min(10, 34)):
    print(f'  Dim {i:2d}: recon_err={recon_err[i]:.4f}, input_mean={pipe._input_mean[i]:.4f}, input_std={pipe._input_std[i]:.4f}')
print(f'  ... (remaining dims similar)')

print(f'\n=== kNN SCORE ANALYSIS ===')
# Get memory representations
mem_active = pipe.memory.active()
print(f'Memory active: {mem_active.shape}')
print(f'Memory Z range: {mem_active.min():.4f} to {mem_active.max():.4f}')
print(f'Memory Z mean: {mem_active.mean():.4f}, std: {mem_active.std():.4f}')

# Score some test points
scores = []
for i in range(0, min(500, len(X_test)), 10):
    score = pipe.scorer.score_point(X_test[i], Z[i])
    scores.append(score)
scores = np.array(scores)
print(f'\nScore range: {scores.min():.4f} - {scores.max():.4f}')
print(f'Score mean: {scores.mean():.4f}, std: {scores.std():.4f}')
print(f'Score median: {np.median(scores):.4f}')

# Expected max score for Tanh outputs
print(f'\n=== EXPECTED vs ACTUAL SCORE RANGE ===')
print(f'Tanh output range: [-1, 1]')
print(f'Per-dim max L1 distance: 2')
print(f'For k=3, dim=68: max theoretical score = 3 * 68 * 2 = 408')
print(f'Actual max score: {scores.max():.4f}')
print(f'Score as % of max theoretical: {scores.max()/408*100:.1f}%')
