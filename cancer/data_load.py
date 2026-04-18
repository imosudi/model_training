from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split
import pandas as pd


# Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("=" * 60)
print("BREAST CANCER WISCONSIN (DIAGNOSTIC) DATASET")
print("=" * 60)
print(f"Number of samples: \n {X}") 
print(f"y: {y}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% → test set
    random_state=42,
    stratify=y          # keep class proportions equal
)

# Quick overview
print(X.shape)          # (569, 30)
print(y.value_counts()) # 1 (benign): 357, 0 (malignant): 212
print(X.isnull().sum().sum())  # 0 - no missing values

# Peek at key features
print(X[["mean radius", "mean texture", "mean area"]].describe())
#time.sleep(200)  # just to space out the prints a bit