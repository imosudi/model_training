from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_load import data, X_train, y_train
from trainings import (
    X_train_sc, X_test_sc,
    rf, lr, knn, dt, tf_model,
    training_histories,
    learning_curves,
    tf_val_scores
)
from validations import y_pred_rf, y_pred_lr, y_pred_knn, y_pred_dt, y_pred_tf
from data_load import y_test
OUTPUT = "cancer/outputs"

# ── Training vs validation loss & accuracy ───────────────────────────────────
HISTORY_TITLES = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "knn": "k-NN",
    "dt": "Decision Tree",
    "tf_model": "TensorFlow",
}

fig, axes = plt.subplots(len(HISTORY_TITLES), 2, figsize=(12, 4 * len(HISTORY_TITLES)))

for row_idx, (name, title) in enumerate(HISTORY_TITLES.items()):
    history_data = training_histories[name]
    epochs = history_data["epoch"]
    x_label = history_data.get("x_label", "Epoch")

    loss_ax = axes[row_idx, 0]
    loss_ax.plot(epochs, history_data["loss"], marker="o", label="Train Loss")
    loss_ax.plot(epochs, history_data["val_loss"], marker="o", label="Validation Loss")
    loss_ax.set_title(f"{title} Loss")
    loss_ax.set_xlabel(x_label)
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()

    acc_ax = axes[row_idx, 1]
    acc_ax.plot(epochs, history_data["accuracy"], marker="o", label="Train Accuracy")
    acc_ax.plot(epochs, history_data["val_accuracy"], marker="o", label="Validation Accuracy")
    acc_ax.set_title(f"{title} Accuracy")
    acc_ax.set_xlabel(x_label)
    acc_ax.set_ylabel("Accuracy")
    acc_ax.legend()

plt.tight_layout()
path = f"{OUTPUT}/training_validation_curves.png"
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved {path}")

# ── Learning curves ───────────────────────────────────────────────────────────
LC_TITLES = {
    "rf":  "Random Forest",
    "lr":  "Logistic Regression",
    "knn": "k-NN",
    "dt":  "Decision Tree",
    "tf_model": "TensorFlow",
}
LC_FILENAMES = {
    "rf":  "learning_curve_randomforest",
    "lr":  "learning_curve_logistic",
    "knn": "learning_curve_knn",
    "dt":  "learning_curve_decisiontree",
    "tf_model": "learning_curve_tensorflow",
}

for name, title in LC_TITLES.items():
    lc   = learning_curves[name]
    sizes, tr, val = lc["train_sizes"], lc["train_scores"], lc["val_scores"]

    plt.plot(sizes, tr.mean(axis=1),  label="Train accuracy")
    plt.plot(sizes, val.mean(axis=1), label="Validation accuracy")
    plt.fill_between(sizes,
        val.mean(axis=1) - val.std(axis=1),
        val.mean(axis=1) + val.std(axis=1),
        alpha=0.15)
    plt.xlabel("Training samples"); plt.ylabel("Accuracy")
    plt.title(f"Learning Curve - {title}", fontsize=13)
    plt.legend(); plt.tight_layout()
    path = f"{OUTPUT}/{LC_FILENAMES[name]}.png"
    plt.savefig(path, dpi=300); plt.close()
    print(f"Saved {path}")

# ── Confusion matrices ────────────────────────────────────────────────────────
CM_CONFIGS = [
    ("lr",  y_pred_lr,  "Logistic Regression",  "confusion_matrix_logistic"),
    ("rf",  y_pred_rf,  "Random Forest",         "confusion_matrix_randomforest"),
    ("knn", y_pred_knn, "k-NN",                  "confusion_matrix_knn"),
    ("dt",  y_pred_dt,  "Decision Tree",         "confusion_matrix_decisiontree"),
    ("tf_model",  y_pred_tf,  "TensorFlow",            "confusion_matrix_tensorflow"),
]

for _, y_pred, title, fname in CM_CONFIGS:
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Malignant", "Benign"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {title}", fontsize=13)
    plt.tight_layout()
    path = f"{OUTPUT}/{fname}.png"
    plt.savefig(path, dpi=300); plt.close()
    print(f"Saved {path}")

# ── Feature importances ───────────────────────────────────────────────────────
def _perm(model, X):
    result = permutation_importance(model, X, y_test, n_repeats=10, random_state=42)
    return result.importances_mean


def _tf_accuracy(X, y):
    y_pred = np.argmax(tf_model.predict(X, verbose=0), axis=1)
    return np.mean(y_pred == np.asarray(y))


def _tf_perm(X, y, n_repeats=10):
    rng = np.random.default_rng(42)
    baseline = _tf_accuracy(X, y)
    importances = np.zeros(X.shape[1], dtype=float)

    for col_idx in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, col_idx] = rng.permutation(X_permuted[:, col_idx])
            drops.append(baseline - _tf_accuracy(X_permuted, y))
        importances[col_idx] = np.mean(drops)

    return importances

IMPORTANCE_CONFIGS = [
    ("rf",  rf,  lambda: rf.feature_importances_,           "Random Forest",       "feature_importances_random_forest"),
    ("lr",  lr,  lambda: lr.coef_[0],                       "Logistic Regression", "feature_importances_logistic_regression"),
    ("knn", knn, lambda: _perm(knn, X_test_sc),             "k-NN (Permutation)",  "feature_importances_knn"),
    ("dt",  dt,  lambda: dt.feature_importances_,           "Decision Tree",       "feature_importances_decision_tree"),
    ("tf_model",  tf_model, lambda: _tf_perm(X_test_sc, y_test),  "TensorFlow (Permutation)", "feature_importances_tensorflow"),
]

for name, model, get_imp, title, fname in IMPORTANCE_CONFIGS:
    values = get_imp()
    # LR uses absolute coefficients
    if name == "lr":
        values = np.abs(values)
    importances = pd.Series(values, index=data.feature_names).sort_values(ascending=False)
    #print(f"\nTop 10 features — {title}:")
    #print(importances.head(10))

    importances.head(10).plot(kind="barh")
    plt.xlabel("Importance score")
    plt.title(f"Top 10 features - {title}")
    plt.tight_layout()
    path = f"{OUTPUT}/{fname}.png"
    plt.savefig(path, dpi=300); plt.close()
    print(f"Saved {path}")
