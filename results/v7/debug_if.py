"""Check: does IsolationForest separate extreme outliers?"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve, roc_curve, roc_auc_score

np.random.seed(42)
normal = np.random.randn(5000, 1).astype(np.float64)
outliers = np.random.randn(500, 1).astype(np.float64) * 10 + 15  # 10 sigma away

X = np.vstack([normal, outliers])
y = np.concatenate([np.zeros(5000), np.ones(500)])

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

IF = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
IF.fit(X_s)
scores = -IF.score_samples(X_s)

n_scores = scores[y==0]
o_scores = scores[y==1]
print(f"Normal: mean={n_scores.mean():.4f}, std={n_scores.std():.4f}")
print(f"Outlier: mean={o_scores.mean():.4f}, std={o_scores.std():.4f}")
print(f"Separation: {abs(o_scores.mean() - n_scores.mean()) / n_scores.std():.1f} sigma")

pr, rc, _ = precision_recall_curve(y, scores)
print(f"AUC-PR: {auc(rc, pr):.4f}")
print(f"AUC-ROC: {roc_auc_score(y, scores):.4f}")
