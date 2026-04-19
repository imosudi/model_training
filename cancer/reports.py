# Full classification report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from validations import (_tensorflow_cross_val_accuracy, 
                         y_pred_rf, y_pred_lr, 
                         y_pred_knn, y_pred_dt, 
                         y_pred_tf
                         )
from data_load import X_test, y_test, X_train, y_train

from trainings import (
    MODEL_CONFIGS,
    X_test_sc,
    X_train_sc,
    tf_model,
    rf, lr, knn, dt, tf_model
)
print("Classification Reports:\n")
# 
print("Random Forest:\n", classification_report(y_test, y_pred_rf,
      target_names=["Malignant", "Benign"]))
# Logistic Regression
print("Logistic Regression:\n", classification_report(y_test, y_pred_lr,
      target_names=["Malignant", "Benign"]))
# k-NN  
print("k-NN:\n", classification_report(y_test, y_pred_knn,
      target_names=["Malignant", "Benign"]))
# Decision Tree
print("Decision Tree:\n", classification_report(y_test, y_pred_dt,
      target_names=["Malignant", "Benign"]))
# TensorFlow
print("TensorFlow:\n", classification_report(y_test, y_pred_tf,
      target_names=["Malignant", "Benign"]))

#time.sleep(200)  # just to space out the prints a bit


# AUC-ROC (1.0 = perfect, 0.5 = random)
print("AUC-ROC:")
y_prob_rf = rf.predict_proba(X_test)[:, 1]
print(f"Random Forest: {roc_auc_score(y_test, y_prob_rf):.3f}") # ~0.994

y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]
print(f"Logistic Regression: {roc_auc_score(y_test, y_prob_lr):.3f}") # ~0.992  

y_prob_knn = knn.predict_proba(X_test_sc)[:, 1]
print(f"k-NN: {roc_auc_score(y_test, y_prob_knn):.3f}") # ~0.987

y_prob_dt = dt.predict_proba(X_test)[:, 1]
print(f"Decision Tree: {roc_auc_score(y_test, y_prob_dt):.3f}") # ~0.980

y_prob_tf = tf_model.predict(X_test_sc, verbose=0)[:, 1]
print(f"TensorFlow: {roc_auc_score(y_test, y_prob_tf):.3f}")





# ── Cross-validation ──────────────────────────────────────────────────────────
cv_scores = {}
for name, (model, scale) in MODEL_CONFIGS.items():
    if name == "tf_model":
        cv_scores[name] = _tensorflow_cross_val_accuracy(X_train_sc, y_train)
        print(f"{name} CV accuracy: {cv_scores[name]:.3f}")
        continue

    X = X_train_sc if scale else X_train
    cv_scores[name] = cross_val_score(model, X, y_train, cv=5).mean()
    print(f"{name} CV accuracy: {cv_scores[name]:.3f}")
