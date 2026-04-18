# Install once: pip install scikit-learn pandas numpy matplotlib
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

"""from sklearn.metrics import (classification_report,
                               confusion_matrix, roc_auc_score)"""
from sklearn.metrics import (classification_report, confusion_matrix,
                               ConfusionMatrixDisplay, roc_auc_score)
import matplotlib.pyplot as plt

np.random.seed(42)  # reproducibility


# Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

"""print("=" * 60)
print("BREAST CANCER WISCONSIN (DIAGNOSTIC) DATASET")
print("=" * 60)
print(f"Number of samples: \n {X}") 
print(f"y: {y}")

time.sleep(200) """

# Quick overview
print(X.shape)          # (569, 30)
print(y.value_counts()) # 1 (benign): 357, 0 (malignant): 212
print(X.isnull().sum().sum())  # 0 - no missing values

# Peek at key features
print(X[["mean radius", "mean texture", "mean area"]].describe())
#time.sleep(200)  # just to space out the prints a bit

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% → test set
    random_state=42,
    stratify=y          # keep class proportions equal
)

print(f"Train: {len(X_train)} samples")  # 455
print(f"Test:  {len(X_test)} samples")   # 114

# Verify stratification
print(y_train.mean().round(3))  # 0.628 ≈ same ratio
print(y_test.mean().round(3))   # 0.623


# Feature scaling
scaler = StandardScaler()

# Fit ONLY on training data, then transform both
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)   # no re-fitting!

# Verify: training features now have μ≈0, σ≈1
print(X_train_sc.mean(axis=0).round(3))   # [0. 0. 0. ...]
print(X_train_sc.std(axis=0).round(3))    # [1. 1. 1. ...]

# Not needed for tree-based models (Random Forest)
# but required for Logistic Regression

# Model Training
# Model 1: Logistic Regression (linear, fast, interpretable)
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)

# Model 2: Random Forest (ensemble, handles non-linearity)
rf = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    max_depth=None,     # grow fully unless limited
    random_state=42
)
rf.fit(X_train, y_train)  # RF doesn't need scaling

# Model 3: k-Nearest Neighbors (non-parametric, distance-based)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sc, y_train)  # k-NN needs scaled features

# Model 4: Decision Tree (single tree, easy to visualize)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)  # Decision Tree doesn't need scaling



# Learning curves - train vs validation score across dataset sizes - Random Forest
train_sizes, train_scores, val_scores = learning_curve(
    rf, X_train, y_train,
    cv=5,                                    # 5-fold CV
    train_sizes=np.linspace(0.2, 1.0, 5),  # 20%→100%
    scoring="accuracy"
)

# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label="Train accuracy")
plt.plot(train_sizes, val_scores.mean(axis=1),   label="Validation accuracy")
plt.fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1),
    alpha=0.15)
plt.xlabel("Training samples"); plt.ylabel("Accuracy")
plt.legend();
plt.title("Learning Curve - Random Forest", fontsize=13)
plt.tight_layout(); plt.savefig("cancer/learning_curve_randomforest.png", dpi=300)
print("\nSaved cancer/learning_curve_randomforest.png")
plt.close()
#plt.show() 

# Learning curves - train vs validation score across dataset sizes - Logistic Regression
train_sizes, train_scores, val_scores = learning_curve(
    lr, X_train_sc, y_train,
    cv=5,                                    # 5-fold CV
    train_sizes=np.linspace(0.2, 1.0, 5),  # 20%→100%
    scoring="accuracy"
)   
# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label="Train accuracy")
plt.plot(train_sizes, val_scores.mean(axis=1),   label="Validation accuracy")
plt.fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1),
    alpha=0.15)
plt.xlabel("Training samples"); plt.ylabel("Accuracy")
plt.legend();
plt.title("Learning Curve - Logistic Regression", fontsize=13)
plt.tight_layout(); plt.savefig("cancer/learning_curve_logistic.png", dpi=300)
print("\nSaved cancer/learning_curve_logistic.png")
plt.close()
#plt.show()

# Learning curves - train vs validation score across dataset sizes - k-NN
train_sizes, train_scores, val_scores = learning_curve(
    knn, X_train_sc, y_train,
    cv=5,                                    # 5-fold CV
    train_sizes=np.linspace(0.2, 1.0, 5),  # 20%→100%
    scoring="accuracy"
)
# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label="Train accuracy")
plt.plot(train_sizes, val_scores.mean(axis=1),   label="Validation accuracy")
plt.fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1),
    alpha=0.15)
plt.xlabel("Training samples"); plt.ylabel("Accuracy")
plt.legend();
plt.title("Learning Curve - k-NN", fontsize=13)
plt.tight_layout(); plt.savefig("cancer/learning_curve_knn.png", dpi=300)
print("\nSaved cancer/learning_curve_knn.png")
plt.close()
#plt.show()     

# Learning curves - train vs validation score across dataset sizes - Decision Tree
train_sizes, train_scores, val_scores = learning_curve(
    dt, X_train, y_train,
    cv=5,                                    # 5-fold CV
    train_sizes=np.linspace(0.2, 1.0, 5),  # 20%→100%
    scoring="accuracy"
)
# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label="Train accuracy")
plt.plot(train_sizes, val_scores.mean(axis=1),   label="Validation accuracy")
plt.fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1),
    alpha=0.15)
plt.xlabel("Training samples"); plt.ylabel("Accuracy")
plt.legend();
plt.title("Learning Curve - Decision Tree", fontsize=13)
plt.tight_layout(); plt.savefig("cancer/learning_curve_decisiontree.png", dpi=300)
print("\nSaved cancer/learning_curve_decisiontree.png")
plt.close()
#plt.show()


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

# Confusion matrix - Logistic Regression
cm = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(cm, display_labels=["Malignant","Benign"])
disp.plot(cmap="Blues"); 
plt.title("Confusion Matrix - Logistic Regression", fontsize=13)
plt.tight_layout()
plt.savefig("cancer/confusion_matrix_logistic.png", dpi=300)  # save figure
print("\nSaved cancer/confusion_matrix_logistic.png")
#plt.show()
plt.close()


# Confusion matrix - Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(cm, display_labels=["Malignant","Benign"])
disp.plot(cmap="Blues"); 
plt.title("Confusion Matrix - Random Forest", fontsize=13)
plt.tight_layout()
plt.savefig("cancer/confusion_matrix_randomforest.png", dpi=300)  # save figure
print("\nSaved cancer/confusion_matrix_randomforest.png")  
plt.close()
#plt.show()

# Confusion matrix - k-NN
cm = confusion_matrix(y_test, y_pred_knn)
disp = ConfusionMatrixDisplay(cm, display_labels=["Malignant","Benign"])
disp.plot(cmap="Blues");
plt.title("Confusion Matrix - k-NN", fontsize=13)
plt.tight_layout()
plt.savefig("cancer/confusion_matrix_knn.png", dpi=300)  # save figure
print("\nSaved cancer/confusion_matrix_knn.png")
plt.close()
#plt.show()     

# Confusion matrix - Decision Tree
cm = confusion_matrix(y_test, y_pred_dt)
disp = ConfusionMatrixDisplay(cm, display_labels=["Malignant","Benign"])
disp.plot(cmap="Blues");
plt.title("Confusion Matrix - Decision Tree", fontsize=13)
plt.tight_layout()
plt.savefig("cancer/confusion_matrix_decisiontree.png", dpi=300)  # save figure
print("\nSaved cancer/confusion_matrix_decisiontree.png")
plt.close()
#plt.show() 


# Full classification report
# Random Forest
print("Classification Report:\n", classification_report(y_test, y_pred_rf,
      target_names=["Malignant", "Benign"]))
# Logistic Regression
print("Classification Report:\n", classification_report(y_test, y_pred_lr,
      target_names=["Malignant", "Benign"]))
# k-NN  
print("Classification Report:\n", classification_report(y_test, y_pred_knn,
      target_names=["Malignant", "Benign"]))
# Decision Tree
print("Classification Report:\n", classification_report(y_test, y_pred_dt,
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


# Extract feature importances from Random Forest
importances_rf = pd.Series(
    rf.feature_importances_,
    index=data.feature_names
).sort_values(ascending=False)

# Top 10 most influential features
print(importances_rf.head(10))

# Visualise
importances_rf.head(10).plot(kind="barh")
plt.xlabel("Importance score")
plt.title("Top 10 features - Random Forest")
plt.tight_layout()
plt.savefig("cancer/random_forest_feature_importances.png", dpi=300)
print("\nSaved cancer/random_forest_feature_importances.png")
plt.close()
#plt.show()

# Logistic Regression: inspect coefficients
coefs = pd.Series(lr.coef_[0], index=data.feature_names)
print(coefs.abs().sort_values(ascending=False).head(5))

# Extract feature importances from Logistic Regression using absolute coefficients
importances_lr = coefs.abs().sort_values(ascending=False)

# Top 10 most influential features
print(importances_lr.head(10))
importances_lr.head(10).plot(kind="barh")
plt.xlabel("Importance score")
plt.title("Top 10 features - Logistic Regression")
plt.tight_layout()
plt.savefig("cancer/logistic_regression_feature_importances.png", dpi=300)
print("\nSaved cancer/logistic_regression_feature_importances.png")
plt.close()
#plt.show()



# Save model + scaler as a bundle (Random Forest doesn't need scaler, but we include it for completeness)
bundle_rf = {"model": rf, "scaler": scaler}
joblib.dump(bundle_rf, "cancer/randomforest_model.pkl")
print("Random Forest model saved. cancer/randomforest_model.pkl")

# Save Logistic Regression model + scaler as a bundle
bundle_lr = {"model": lr, "scaler": scaler}
joblib.dump(bundle_lr, "cancer/logisticregression_model.pkl")
print("Logistic Regression model saved. cancer/logisticregression_model.pkl")

