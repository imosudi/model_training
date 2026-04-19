import time

import numpy as np
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

def extract_features_window(window):
    feats = []
    for axis in range(6):
        feats.append(np.mean(window[:, axis]))
        feats.append(np.std(window[:, axis]))
    for axis in range(6):
        feats.append(np.min(window[:, axis]))
        feats.append(np.max(window[:, axis]))

    feats.append(np.sum(np.abs(window[:, :3])) / window.shape[0])
    feats.append(np.sum(np.abs(window[:, 3:])) / window.shape[0])

    acc_mag = np.linalg.norm(window[:, :3], axis=1)
    feats.append(np.mean(acc_mag))
    feats.append(np.std(acc_mag))

    return np.array(feats)
NUM_CLASSES = len(np.unique(y_train))
print(f"NUM_CLASSES: {NUM_CLASSES}");#time.sleep(100)
print("Extracting features from raw data...")
print(f"Original shape - train: {X_train.shape}, test: {X_test.shape}")
print("X_train: ", X_train)
X_train_feat = np.array([extract_features_window(w) for w in X_train.values.reshape(-1, 5, 6)])
X_test_feat  = np.array([extract_features_window(w) for w in X_test.values.reshape(-1, 5, 6)])

print(f"Feature shape - train: {X_train_feat.shape}, test: {X_test_feat.shape}")