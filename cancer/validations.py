from sklearn.model_selection import cross_val_score

from trainings import MODEL_CONFIGS, X_train_sc, X_train, X_test_sc, X_test, y_train

# ── Cross-validation ──────────────────────────────────────────────────────────
cv_scores = {}
for name, (model, scale) in MODEL_CONFIGS.items():
    X = X_train_sc if scale else X_train
    cv_scores[name] = cross_val_score(model, X, y_train, cv=5).mean()
    print(f"{name} CV accuracy: {cv_scores[name]:.3f}")

# ── Test-set predictions ──────────────────────────────────────────────────────
predictions = {}
for name, (model, scale) in MODEL_CONFIGS.items():
    predictions[name] = model.predict(X_test_sc if scale else X_test)

y_pred_lr, y_pred_knn, y_pred_rf, y_pred_dt = (predictions[k] for k in ("lr", "knn", "rf", "dt"))