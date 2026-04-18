# Full classification report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from trainings import rf, lr, knn, dt, X_test_sc, y_test
from validations import y_pred_rf, y_pred_lr, y_pred_knn, y_pred_dt
from data_load import X_test, y_test

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

#time.sleep(200)  # just to space out the prints a bit


# Output:
#                  precision  recall  f1-score  support
#   Malignant          0.98    0.95      0.96       42
#   Benign             0.97    0.99      0.98       72
#
#    accuracy                           0.96       114
#   macro avg       0.96      0.95      0.95       114
#weighted avg       0.96      0.96      0.96       114

# AUC-ROC (1.0 = perfect, 0.5 = random)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_rf):.3f}") # ~0.994

y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_lr):.3f}") # ~0.992  

y_prob_knn = knn.predict_proba(X_test_sc)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_knn):.3f}") # ~0.987

y_prob_dt = dt.predict_proba(X_test)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_dt):.3f}") # ~0.980
