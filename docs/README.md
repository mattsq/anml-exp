# Documentation

## Quick Start

```python
from anomaly_models.isolation_forest import IsolationForestModel
from datasets.registry import load_dataset
from sklearn.metrics import roc_auc_score

# Load synthetic toy data
X_train, _ = load_dataset("toy-blobs", split="train")
X_test, y_test = load_dataset("toy-blobs", split="test")

# Fit and score
model = IsolationForestModel(n_estimators=200).fit(X_train)
scores = model.score_samples(X_test)

print("ROC-AUC:", roc_auc_score(y_test, scores))
```

The above snippet is taken from the project specification and serves as a
minimal working example.
