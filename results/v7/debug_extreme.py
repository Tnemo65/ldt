"""Deep debug: why is AUC-PR still near-random?"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r'C:\proj\ldt\results\v7')
import benchmark_v7 as bm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, precision_recall_curve, roc_curve

bm.MONTHS = [1]
df = bm.clean(bm.load_month(2024, 1))
X = bm.features(df)

rng = np.random.RandomState(42)
train_idx = rng.choice(len(df), 10000, replace=False)
X_train = X[train_idx]

test_df = df.iloc[rng.choice(len(df), bm.TEST_N, replace=False)].reset_index(drop=True)

# Inject extreme fare
params = {'type': 'extreme_fare', 'fare_range': (150, 500), 'n': 500}
test_df_inj, y_labels = bm.inject_anomalies(test_df, params, 42)
X_test = bm.features(test_df_inj).astype(np.float32)
print(f"Anomalies: {y_labels.sum()}/{len(y_labels)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)

# Check each feature's separation
print("\n=== Feature separation (z-score of anomaly mean from normal mean) ===")
for fi in range(25):
    n_mean = X_train_s[:, fi].mean()
    n_std = X_train_s[:, fi].std()
    a_mean = X_test_s[y_labels==1, fi].mean()
    z = abs(a_mean - n_mean) / (n_std + 1e-9)
    if z > 1.5:
        print(f"  f{fi:02d}: z={z:.1f} sigma | norm_mean={n_mean:.2f}, anom_mean={a_mean:.2f}")

# Train IF and check score separation
IF = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
IF.fit(X_train_s)
raw_scores = -IF.score_samples(X_test_s)

n_scores = raw_scores[y_labels==0]
a_scores = raw_scores[y_labels==1]
print(f"\n=== IF scores ===")
print(f"Normal: mean={n_scores.mean():.4f}, std={n_scores.std():.4f}")
print(f"Anomal: mean={a_scores.mean():.4f}, std={a_scores.std():.4f}")
print(f"Separation: {abs(a_scores.mean() - n_scores.mean()) / (n_scores.std() + 1e-9):.3f} sigma")

pr_curve, rc_curve, _ = precision_recall_curve(y_labels, raw_scores)
auc_pr = auc(rc_curve, pr_curve)
fpr_arr, tpr_arr, _ = roc_curve(y_labels, raw_scores)
auc_roc = auc(fpr_arr, tpr_arr)
print(f"AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}")

# Try with ONLY anomaly-flagged feature
print("\n=== IF trained on feature 2 (fare) only ===")
IF2 = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
IF2.fit(X_train_s[:, [2]])
s2 = -IF2.score_samples(X_test_s[:, [2]])
pr2, rc2, _ = precision_recall_curve(y_labels, s2)
print(f"AUC-PR (f2 only): {auc(rc2, pr2):.4f}")

# Try with top 3 discriminating features
print("\n=== IF on top 3 features ===")
top3 = [2, 6, 7]  # fare, fare/dist, fare/dur
IF3 = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
IF3.fit(X_train_s[:, top3])
s3 = -IF3.score_samples(X_test_s[:, top3])
pr3, rc3, _ = precision_recall_curve(y_labels, s3)
print(f"AUC-PR (f2,6,7): {auc(rc3, pr3):.4f}")

# Check: is the scaler fitted on training only?
print("\n=== Normalize test data with scaler fitted on training ===")
print(f"Train fare mean: {X_train_s[:,2].mean():.4f}")
print(f"Test normal fare mean: {X_test_s[y_labels==0, 2].mean():.4f}")
print(f"Test anom fare mean: {X_test_s[y_labels==1, 2].mean():.4f}")
print(f"Original injected fare range: $150-$500")
