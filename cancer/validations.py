import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from trainings import (
    MODEL_CONFIGS,
    TF_LOOP_FIT_PARAMS,
    X_test,
    X_test_sc,
    X_train,
    X_train_sc,
    build_tensorflow_model,
    tf_model,
    y_train,
)


def _tensorflow_cross_val_accuracy(X, y, n_splits=5):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    y_array = np.asarray(y)

    for train_idx, val_idx in splitter.split(X, y_array):
        fold_model = build_tensorflow_model()
        fold_model.fit(X[train_idx], y_array[train_idx], **TF_LOOP_FIT_PARAMS)
        y_pred = np.argmax(fold_model.predict(X[val_idx], verbose=0), axis=1)
        scores.append(accuracy_score(y_array[val_idx], y_pred))

    return float(np.mean(scores))


# ── Test-set predictions ──────────────────────────────────────────────────────
predictions = {}
for name, (model, scale) in MODEL_CONFIGS.items():
    if name == "tf_model":
        tf_probs = tf_model.predict(X_test_sc, verbose=0)
        predictions[name] = np.argmax(tf_probs, axis=1)
        continue

    predictions[name] = model.predict(X_test_sc if scale else X_test)

y_pred_lr, y_pred_knn, y_pred_rf, y_pred_dt, y_pred_tf = (
    predictions[k] for k in ("lr", "knn", "rf", "dt", "tf_model")
)
