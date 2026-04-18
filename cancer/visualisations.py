from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import learning_curve

from trainings import X_train_sc, rf, lr, knn, dt, X_test_sc, y_test
from data_load import data,  X_train, y_train, y_test
from validations import y_pred_rf, y_pred_lr, y_pred_knn, y_pred_dt
from trainings import train_sizes, train_scores, val_scores


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
plt.tight_layout(); plt.savefig("cancer/outputs/learning_curve_randomforest.png", dpi=300)
print("\nSaved cancer/outputs/learning_curve_randomforest.png")
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
plt.tight_layout(); plt.savefig("cancer/outputs/learning_curve_logistic.png", dpi=300)
print("\nSaved cancer/outputs/learning_curve_logistic.png")
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
plt.tight_layout(); plt.savefig("cancer/outputs/learning_curve_knn.png", dpi=300)
print("\nSaved cancer/outputs/learning_curve_knn.png")
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
plt.tight_layout(); plt.savefig("cancer/outputs/learning_curve_decisiontree.png", dpi=300)
print("\nSaved cancer/outputs/learning_curve_decisiontree.png")
plt.close()
#plt.show()



# Confusion matrix - Logistic Regression
cm = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(cm, display_labels=["Malignant","Benign"])
disp.plot(cmap="Blues"); 
plt.title("Confusion Matrix - Logistic Regression", fontsize=13)
plt.tight_layout()
plt.savefig("cancer/outputs/confusion_matrix_logistic.png", dpi=300)  # save figure
print("\nSaved cancer/outputs/confusion_matrix_logistic.png")
#plt.show()
plt.close()


# Confusion matrix - Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(cm, display_labels=["Malignant","Benign"])
disp.plot(cmap="Blues"); 
plt.title("Confusion Matrix - Random Forest", fontsize=13)
plt.tight_layout()
plt.savefig("cancer/outputs/confusion_matrix_randomforest.png", dpi=300)  # save figure
print("\nSaved cancer/outputs/confusion_matrix_randomforest.png")  
plt.close()
#plt.show()

# Confusion matrix - k-NN
cm = confusion_matrix(y_test, y_pred_knn)
disp = ConfusionMatrixDisplay(cm, display_labels=["Malignant","Benign"])
disp.plot(cmap="Blues");
plt.title("Confusion Matrix - k-NN", fontsize=13)
plt.tight_layout()
plt.savefig("cancer/outputs/confusion_matrix_knn.png", dpi=300)  # save figure
print("\nSaved cancer/outputs/confusion_matrix_knn.png")
plt.close()
#plt.show()     

# Confusion matrix - Decision Tree
cm = confusion_matrix(y_test, y_pred_dt)
disp = ConfusionMatrixDisplay(cm, display_labels=["Malignant","Benign"])
disp.plot(cmap="Blues");
plt.title("Confusion Matrix - Decision Tree", fontsize=13)
plt.tight_layout()
plt.savefig("cancer/outputs/confusion_matrix_decisiontree.png", dpi=300)  # save figure
print("\nSaved cancer/outputs/confusion_matrix_decisiontree.png")
plt.close()
#plt.show() 



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
plt.savefig("cancer/outputs/feature_importances_random_forest.png", dpi=300)
print("\nSaved cancer/outputs/feature_importances_random_forest.png")
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
plt.savefig("cancer/outputs/feature_importances_logistic_regression.png", dpi=300)
print("\nSaved cancer/outputs/feature_importances_logistic_regression.png")
plt.close()
#plt.show()

# Extract feature importances from k-NN using permutation importance
perm_importance_knn = permutation_importance(
    knn, X_test_sc, y_test, n_repeats=10, random_state=42
)
importances_knn = pd.Series(
    perm_importance_knn.importances_mean,
    index=data.feature_names
).sort_values(ascending=False)      

# Top 10 most influential features
print(importances_knn.head(10))
importances_knn.head(10).plot(kind="barh")
plt.xlabel("Importance score")
plt.title("Top 10 features - k-NN (Permutation Importance)")
plt.tight_layout()
plt.savefig("cancer/outputs/feature_importances_knn.png", dpi=300)
print("\nSaved cancer/outputs/feature_importances_knn.png")
plt.close()
#plt.show()     

# Extract feature importances from Decision Tree
importances_dt = pd.Series(
    dt.feature_importances_,
    index=data.feature_names
).sort_values(ascending=False)  

# Top 10 most influential features
print(importances_dt.head(10))
importances_dt.head(10).plot(kind="barh")
plt.xlabel("Importance score")
plt.title("Top 10 features - Decision Tree")
plt.tight_layout()
plt.savefig("cancer/outputs/feature_importances_decision_tree.png", dpi=300)
print("\nSaved cancer/outputs/feature_importances_decision_tree.png")
plt.close()
#plt.show()     
