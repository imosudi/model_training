
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import numpy as np

from data_load import X_train, X_test, y_train, y_test


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



print(f"Train: {len(X_train)} samples")  # 455
print(f"Test:  {len(X_test)} samples")   # 114

# Verify stratification
print(y_train.mean().round(3))  # 0.628 ≈ same ratio
print(y_test.mean().round(3))   # 0.623






