from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import numpy as np

from data_load import X_train, X_test, y_train, y_test

# ── Scaling ───────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Model definitions: (model, needs_scaling) ─────────────────────────────────
MODEL_CONFIGS = {
    "lr":  (LogisticRegression(C=1.0, max_iter=1000, random_state=42),  True),
    "knn": (KNeighborsClassifier(n_neighbors=5),                         True),
    "rf":  (RandomForestClassifier(n_estimators=100, random_state=42),  False),
    "dt":  (DecisionTreeClassifier(max_depth=3, random_state=42),       False),
}

# ── Training ──────────────────────────────────────────────────────────────────
models = {}
for name, (model, scale) in MODEL_CONFIGS.items():
    models[name] = model.fit(X_train_sc if scale else X_train, y_train)

lr, knn, rf, dt = (models[k] for k in ("lr", "knn", "rf", "dt"))

# ── Learning curves ───────────────────────────────────────────────────────────
LC_PARAMS = dict(cv=5, train_sizes=np.linspace(0.2, 1.0, 5), scoring="accuracy")

learning_curves = {}
for name, (model, scale) in MODEL_CONFIGS.items():
    sizes, train_sc, val_sc = learning_curve(
        model, X_train_sc if scale else X_train, y_train, **LC_PARAMS
    )
    learning_curves[name] = {
        "train_sizes":  sizes,
        "train_scores": train_sc,
        "val_scores":   val_sc,
    }


# ── Backwards-compatible unpacking for visualisations.py ─────────────────────
for _name in MODEL_CONFIGS:
    globals()[f"{_name}_train_sizes"]  = learning_curves[_name]["train_sizes"]
    globals()[f"{_name}_train_scores"] = learning_curves[_name]["train_scores"]
    globals()[f"{_name}_val_scores"]   = learning_curves[_name]["val_scores"]

train_sizes  = learning_curves["knn"]["train_sizes"]
train_scores = learning_curves["knn"]["train_scores"]
val_scores   = learning_curves["knn"]["val_scores"]

# ── Sanity checks ─────────────────────────────────────────────────────────────
assert np.allclose(X_train_sc.mean(axis=0), 0, atol=1e-6), "Scaler mean check failed"
assert np.allclose(X_train_sc.std(axis=0),  1, atol=1e-6), "Scaler std check failed"
assert abs(y_train.mean() - y_test.mean()) < 0.02,          "Stratification drift detected"

print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
print(f"Class balance → train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")