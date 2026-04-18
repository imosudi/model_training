from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_load import data, X_train, y_train
from trainings import (
    X_train_sc, X_test_sc,
    rf, lr, knn, dt,
    learning_curves,
)
from validations import y_pred_rf, y_pred_lr, y_pred_knn, y_pred_dt
from data_load import y_test
OUTPUT = "cancer/outputs"

# ── Learning curves ───────────────────────────────────────────────────────────
LC_TITLES = {
    "rf":  "Random Forest",
    "lr":  "Logistic Regression",
    "knn": "k-NN",
    "dt":  "Decision Tree",
}
LC_FILENAMES = {
    "rf":  "learning_curve_randomforest",
    "lr":  "learning_curve_logistic",
    "knn": "learning_curve_knn",
    "dt":  "learning_curve_decisiontree",
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

IMPORTANCE_CONFIGS = [
    ("rf",  rf,  lambda: rf.feature_importances_,           "Random Forest",       "feature_importances_random_forest"),
    ("lr",  lr,  lambda: lr.coef_[0],                       "Logistic Regression", "feature_importances_logistic_regression"),
    ("knn", knn, lambda: _perm(knn, X_test_sc),             "k-NN (Permutation)",  "feature_importances_knn"),
    ("dt",  dt,  lambda: dt.feature_importances_,           "Decision Tree",       "feature_importances_decision_tree"),
]

for name, model, get_imp, title, fname in IMPORTANCE_CONFIGS:
    values = get_imp()
    # LR uses absolute coefficients
    if name == "lr":
        values = np.abs(values)
    importances = pd.Series(values, index=data.feature_names).sort_values(ascending=False)
    print(f"\nTop 10 features — {title}:")
    print(importances.head(10))

    importances.head(10).plot(kind="barh")
    plt.xlabel("Importance score")
    plt.title(f"Top 10 features - {title}")
    plt.tight_layout()
    path = f"{OUTPUT}/{fname}.png"
    plt.savefig(path, dpi=300); plt.close()
    print(f"Saved {path}")