---
name: scikit-learn
description: Machine learning in Python with scikit-learn. Use when working with supervised learning, unsupervised learning, model evaluation, hyperparameter tuning, preprocessing, or building ML pipelines.
license: BSD-3-Clause
metadata:
    skill-author: K-Dense Inc.
---

# Scikit-learn — Classical ML in Python

Industry-standard Python library for classical machine learning.

## Core Workflow Template

```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

# 1. Split (ALWAYS stratify for classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Pipeline (prevents data leakage)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier(random_state=42))
])

# 3. Hyperparameter search
param_grid = {'clf__n_estimators': [100, 200], 'clf__max_depth': [3, 5, 7]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 4. Evaluate
best = grid_search.best_estimator_
y_pred = best.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Algorithm Selection Guide

| Task | Best Choice | Alternative |
|------|-------------|-------------|
| Binary classification | `GradientBoostingClassifier` | `RandomForestClassifier` |
| Regression | `GradientBoostingRegressor` | `Ridge` |
| Anomaly detection | `IsolationForest` | `LocalOutlierFactor` |
| Clustering | `DBSCAN` (density) | `KMeans` (partition) |
| Dimensionality reduction | `PCA` | `UMAP` |

## For Data Quality / Streaming

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Isolation Forest (fast, good for high-dim data)
iso = IsolationForest(contamination=0.05, random_state=42)
labels = iso.fit_predict(X)  # -1 = anomaly, 1 = normal

# Local Outlier Factor (density-based)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
labels = lof.fit_predict(X)
```

## Best Practices

1. **Always use Pipelines** — prevents data leakage in CV
2. **Fit on train only** — `scaler.fit_transform(X_train)`, `scaler.transform(X_test)`
3. **Set random_state** for reproducibility
4. **Use `n_jobs=-1`** for parallelism
5. **Report multiple metrics** — not just accuracy for imbalanced data
