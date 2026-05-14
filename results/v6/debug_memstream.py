"""Debug MemStream."""
import numpy as np
import sys
sys.path.insert(0, r'C:\proj\ldt\results\v6')
from benchmark_v6 import MemStream, evaluate_streaming

np.random.seed(42)
X_train = np.random.randn(1000, 10).astype(np.float64)
np.random.seed(43)
X_test = np.random.randn(100, 10).astype(np.float64)
y_test = np.zeros(100, dtype=np.int8)
y_test[:5] = 1

X_val = X_train[:50]

for seed in [42, 123]:
    m = MemStream(seed=seed)
    m.fit(X_train)
    print("Seed", seed, ": memory len=", len(m.memory), ", buf len=", len(m._buf))
    scores = m.decision_function(X_test)
    print("  Scores: min=%.4f, max=%.4f, mean=%.4f" % (scores.min(), scores.max(), scores.mean()))
    print("  First 5:", scores[:5])
    res = evaluate_streaming(MemStream, X_train, X_val, X_test, y_test[:50], y_test, seed, label_budget=0)
    print("  AUC_PR=%.6f, anomaly_rate=%.4f" % (res["AUC_PR"], res["anomaly_rate"]))
    print()
