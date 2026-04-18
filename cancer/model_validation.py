from model_training import X_test_sc, X_train_sc, X_train_sc, lr, rf, knn, dt, scaler, X_train, X_test, y_train, y_test
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score

# Cross-validation scores (5-fold) for quick comparison
lr_cv = cross_val_score(lr, X_train_sc, y_train, cv=5).mean()
rf_cv = cross_val_score(rf, X_train,    y_train, cv=5).mean()
knn_cv = cross_val_score(knn, X_train_sc, y_train, cv=5).mean()
dt_cv  = cross_val_score(dt, X_train,    y_train, cv=5).mean()

print(f"Logistic Regression Cross-Validation accuracy: {lr_cv:.3f}")  # ~0.960
print(f"Random Forest Cross-Validation accuracy: {rf_cv:.3f}")  # ~0.963
print(f"k-NN Cross-Validation accuracy: {knn_cv:.3f}")  # ~0.956
print(f"Decision Tree Cross-Validation accuracy: {dt_cv:.3f}")  # ~0.944

# Evaluate on test set
# Predictions
y_pred_lr = lr.predict(X_test_sc) # Logistic Regression needs scaled features
y_pred_rf = rf.predict(X_test) # Random Forest can use raw features
y_pred_knn = knn.predict(X_test_sc) # k-NN needs scaled features
y_pred_dt = dt.predict(X_test) # Decision Tree can use raw features 
