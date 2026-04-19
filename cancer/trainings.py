from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, learning_curve

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
import seaborn as sns



import numpy as np

from data_load import NUM_CLASSES, X_train, X_test, X_train_feat, y_train, y_test

tf.random.set_seed(42)




# ── Scaling ───────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


TF_MAIN_FIT_PARAMS = {
    "epochs": 50,
    "batch_size": 32,
    "verbose": 1,
}

TF_LOOP_FIT_PARAMS = {
    "epochs": 50,
    "batch_size": 32,
    "verbose": 0,
}


def build_tensorflow_model():
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def build_sklearn_history(model, X_fit, y_fit, X_val, y_val):
    return {
        "epoch": [1],
        "accuracy": [accuracy_score(y_fit, model.predict(X_fit))],
        "val_accuracy": [accuracy_score(y_val, model.predict(X_val))],
    }


def _tensorflow_learning_curve(X, y, train_sizes, n_splits=5):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_array = np.asarray(y)
    train_scores = []
    val_scores = []
    absolute_sizes = []

    for size_idx, frac in enumerate(train_sizes):
        fold_train_scores = []
        fold_val_scores = []
        fold_sizes = []

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y_array)):
            fold_train_idx = train_idx[: max(1, int(len(train_idx) * frac))]
            fold_sizes.append(len(fold_train_idx))

            tf.keras.utils.set_random_seed(42 + size_idx * 10 + fold_idx)
            fold_model = build_tensorflow_model()
            fold_model.fit(X[fold_train_idx], y_array[fold_train_idx], **TF_LOOP_FIT_PARAMS)

            train_acc = fold_model.evaluate(
                X[fold_train_idx], y_array[fold_train_idx], verbose=0
            )[1]
            val_acc = fold_model.evaluate(X[val_idx], y_array[val_idx], verbose=0)[1]
            fold_train_scores.append(train_acc)
            fold_val_scores.append(val_acc)

        absolute_sizes.append(int(np.mean(fold_sizes)))
        train_scores.append(fold_train_scores)
        val_scores.append(fold_val_scores)

    return (
        np.array(absolute_sizes),
        np.array(train_scores),
        np.array(val_scores),
    )

# ── Model definitions: (model, needs_scaling) ─────────────────────────────────
MODEL_CONFIGS = {
    "lr":  (LogisticRegression(solver="saga", C=1.0, max_iter=1000, random_state=42,  verbose=1),  True),
    "knn": (KNeighborsClassifier(n_neighbors=5),                         True),
    "rf":  (RandomForestClassifier(n_estimators=100, random_state=42, verbose=1),  False),
    "dt":  (DecisionTreeClassifier(max_depth=3, random_state=42),       False),
    "tf_model": (build_tensorflow_model(), True),
}

# ── Training ──────────────────────────────────────────────────────────────────
training_models = {}
training_histories = {}
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)
for name, (model, scale) in MODEL_CONFIGS.items():
    print(f"\nTraining {name} model...")
    if name == "tf_model":
        print("\nTensorFlow model summary:")
        model.summary()
        training_histories[name] = model.fit(
            X_train_sc if scale else X_train,
            y_train,
            validation_data=(X_test_sc, y_test),
            callbacks=[early_stop],
            **TF_MAIN_FIT_PARAMS,
        )
        training_models[name] = model
        continue

    X_fit = X_train_sc if scale else X_train
    X_val = X_test_sc if scale else X_test
    training_models[name] = model.fit(X_fit, y_train)
    training_histories[name] = build_sklearn_history(
        training_models[name], X_fit, y_train, X_val, y_test
    )

lr, knn, rf, dt, tf_model = (training_models[k] for k in ("lr", "knn", "rf", "dt", "tf_model"))
history = training_histories["tf_model"]

# ── Learning curves ───────────────────────────────────────────────────────────
LC_PARAMS = dict(cv=5, train_sizes=np.linspace(0.2, 1.0, 5), scoring="accuracy")

learning_curves = {}
for name, (model, scale) in MODEL_CONFIGS.items():
    if name == "tf_model":
        sizes, train_sc, val_sc = _tensorflow_learning_curve(
            X_train_sc, y_train, LC_PARAMS["train_sizes"], n_splits=LC_PARAMS["cv"]
        )
    else:
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

tf_val_scores = learning_curves["tf_model"]["val_scores"]


# ── Sanity checks ─────────────────────────────────────────────────────────────
assert np.allclose(X_train_sc.mean(axis=0), 0, atol=1e-6), "Scaler mean check failed"
assert np.allclose(X_train_sc.std(axis=0),  1, atol=1e-6), "Scaler std check failed"
assert abs(y_train.mean() - y_test.mean()) < 0.02,          "Stratification drift detected"


print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
print(f"Class balance → train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")
