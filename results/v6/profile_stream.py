"""Profile CADIFEiaStream to find exact bottleneck."""
import sys, time, numpy as np
sys.path.insert(0, r'C:\proj\ldt\results\v6')
from benchmark_v6 import ContextFeatureWeighting, CADIFEiaStream
from sklearn.ensemble import IsolationForest

np.random.seed(42)
X_train = np.random.randn(10000, 25).astype(np.float32)
X_test = np.random.randn(11500, 25).astype(np.float32)

# Init
rng_w = np.random.RandomState(42)
W1 = rng_w.randn(25, 16).astype(np.float32) * 0.1
b1 = rng_w.randn(16).astype(np.float32) * 0.1

Xw = X_train[:2000]
Xp = np.maximum(Xw @ W1 + b1, 0)
cw = ContextFeatureWeighting()
cw.fit(Xw)

IF = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
IF.fit(Xp)

# --- Profile score_one per call ---
x = X_test[0]
t0 = time.time()
for _ in range(11500):
    xf = x.reshape(1, -1).astype(np.float32)
    Xp = np.maximum(xf @ W1 + b1, 0)
    iso = -IF.score_samples(Xp)[0]
    w = cw.get_weights(xf).mean()
    w = max(w, 0.1)
    s = iso * w
total = time.time() - t0
print(f"score_one loop (11500 calls): {total:.1f}s ({total*1000/11500:.2f}ms/call)")

# --- Profile score_batch (vectorized) ---
t0 = time.time()
Xp_batch = np.maximum(X_test @ W1 + b1, 0)
iso_batch = -IF.score_samples(Xp_batch)
w_batch = cw.get_weights(X_test).mean(axis=1)
w_batch = np.maximum(w_batch, 0.1)
scores_batch = iso_batch * w_batch
total = time.time() - t0
print(f"score_batch (11500 at once):   {total:.2f}s")

print(f"\nSpeedup: {total/((total*1000/11500)*11500/1000):.0f}x faster")
