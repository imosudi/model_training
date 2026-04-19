"""
iris_classification.py
======================
A beginner-friendly walkthrough of data classification using the Iris dataset.

Covers:
  1. Downloading the data (two methods)
  2. Exploring the data
  3. Splitting into train / test sets
  4. Scaling features
  5. Training a k-NN classifier
  6. Evaluating: accuracy, classification report, confusion matrix
  7. Comparing with a Decision Tree
  8. Visualising the decision boundary

Run with:  python3 iris_classification.py
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── 1. LOAD THE DATA ──────────────────────────────────────────────────────────
#
# Method A — download the raw CSV from UCI (shows you how real-world loading works)
# ---------------------------------------------------------------------------------
# import urllib.request
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# urllib.request.urlretrieve(url, "iris.csv")
# col_names = ["sepal_length","sepal_width","petal_length","petal_width","species"]
# df = pd.read_csv("iris.csv", header=None, names=col_names)
#
# Method B — load directly from scikit-learn (used here for convenience)
# ---------------------------------------------------------------------------------
iris = load_iris(as_frame=True)       # returns a Bunch object (dict-like)
df   = iris.frame                     # pandas DataFrame, species stored as integer codes
df["species_name"] = df["target"].map(dict(enumerate(iris.target_names)))

print("=" * 60)
print("IRIS DATASET  — first 5 rows")
print("=" * 60)
print(df.head())

# ── 2. EXPLORE THE DATA ───────────────────────────────────────────────────────
print("\n── Shape ──────────────────────────────────")
print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")

#print(df.shape)  # 150 samples, 5 columns (4 features + target)
#time.sleep(200)  # just to space out the prints a bit

print("\n── Class distribution ──────────────────────")
print(df["species_name"].value_counts())   # perfectly balanced: 50 per class
print("Check 1: ", df["species_name"])  
print("\n── Summary statistics ───────────────────────")
print("Check 2: ", df.describe().round(2))
print("Check 3: ", df.describe())
#time.sleep(200)  # just to space out the prints a bit
# Pairplot — see how well the 4 features separate the 3 species visually
print("\nSaving pairplot → iris_pairplot.png")
pairplot_fig = sns.pairplot(df, hue="species_name", vars=iris.feature_names,
                             plot_kws={"alpha": 0.7, "s": 40})
pairplot_fig.figure.suptitle("Iris — feature pairplot", y=1.02)
pairplot_fig.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 3. PREPARE FEATURES AND LABELS ────────────────────────────────────────────
#
# X  = the input features the model learns from  (4 columns)
# y  = the label (target) the model tries to predict  (0 / 1 / 2)
#
X = df[iris.feature_names].values   # shape (150, 4)
y = df["target"].values             # shape (150,)

print("\n── Feature matrix shape:", X.shape)
print("── Label vector shape:  ", y.shape)
print("── Class labels:        ", np.unique(y), "→", iris.target_names)

# ── 4. SPLIT INTO TRAIN / TEST ────────────────────────────────────────────────
#
# We hold out 20 % of the data as a test set the model never sees during training.
# random_state fixes the shuffle so results are reproducible.
# stratify=y keeps the class proportions the same in both splits.
#
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\n── Train size: {X_train.shape[0]} samples")
print(f"── Test size:  {X_test.shape[0]} samples")
"""print(f"\n── Train size: {X_train.shape} samples")
print(f"── Test size:  {X_test.shape} samples")
print(f"\n── Train size: {X_train} samples")
print(f"── Test size:  {X_test} samples")

print("\n── Train class distribution:")
print(pd.Series(y_train).value_counts())
print("\n── Test class distribution:")
print(pd.Series(y_test).value_counts())     

print("\n── Train class proportions:")
print(X_train.shape[0], y_train.sum(), y_train.mean().round(3))  # 120 samples, 40 positive, 0.333
print("\n── Test class proportions:")
print(X_test.shape[0], y_test.sum(), y_test.mean().round(3))      # 30 samples, 10 positive, 0.333 — same ratio as train, thanks to stratify=y      
"""
#time.sleep(200)  # just to space out the prints a bit

# ── 5. SCALE THE FEATURES ─────────────────────────────────────────────────────
#
# k-NN measures distance between samples.  If one feature ranges 0–10 and another
# 0–0.1, the large one dominates the distance calculation unfairly.
# StandardScaler rescales each feature to mean=0, std=1.
#
# IMPORTANT: fit only on the training data, then transform both sets.
# Fitting on test data would be "data leakage" — peeking at the future.
#
scaler   = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # learn mean/std from train, then scale
X_test_sc  = scaler.transform(X_test)        # use the SAME mean/std on test

# ── 6. TRAIN — k-Nearest Neighbors ────────────────────────────────────────────
#
# k-NN stores all training samples.  When asked to classify a new point, it finds
# the k closest training points (by Euclidean distance) and takes a majority vote.
#
# k=5 means: look at the 5 nearest neighbors.
#
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sc, y_train)

# ── 7. EVALUATE — k-NN ────────────────────────────────────────────────────────
y_pred_knn = knn.predict(X_test_sc)

print("\n" + "=" * 60)
print("k-NN (k=5) EVALUATION on test set")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.2%}")
print("\nClassification report:")
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))

# ── 8. COMPARE — Decision Tree ────────────────────────────────────────────────
#
# A Decision Tree learns a sequence of yes/no questions ("is petal_length < 2.5?")
# that partition the data into pure groups.  It needs no scaling (distance-free).
#
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)                   # raw features — no scaling needed
y_pred_dt = dt.predict(X_test)

print("=" * 60)
print("Decision Tree (max_depth=3) EVALUATION on test set")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.2%}")
print("\nClassification report:")
print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))

# ── 9. CONFUSION MATRICES ─────────────────────────────────────────────────────
#
# Rows = actual class, Columns = predicted class.
# Perfect classifier → all numbers sit on the diagonal.
# Off-diagonal cells reveal which classes get confused with which.
#
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (title, y_pred) in zip(axes, [
    ("k-NN (k=5)", y_pred_knn),
    ("Decision Tree (depth 3)", y_pred_dt)
]):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)

plt.tight_layout()
plt.savefig("iris_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved iris_confusion_matrices.png")

# ── 10. VISUALISE THE DECISION TREE ──────────────────────────────────────────
#
# One of the best things about Decision Trees: you can literally read the rules
# the model learned.
#
fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(
    dt,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)
ax.set_title("Iris Decision Tree — learned rules (max_depth=3)", fontsize=13)
plt.tight_layout()
plt.savefig("iris_decision_tree.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved iris_decision_tree.png")

# ── 11. HOW DOES k AFFECT ACCURACY? (k-NN sensitivity check) ─────────────────
k_values   = range(1, 21)
accuracies = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_sc, y_train)
    accuracies.append(accuracy_score(y_test, model.predict(X_test_sc)))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_values, accuracies, marker="o", linewidth=1.5, markersize=5, color="#185FA5")
ax.axvline(x=5, color="#D85A30", linestyle="--", linewidth=1, label="k=5 (default)")
ax.set_xlabel("k (number of neighbors)")
ax.set_ylabel("Test accuracy")
ax.set_title("k-NN accuracy vs. k on Iris test set")
ax.set_xticks(list(k_values))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("iris_knn_k_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved iris_knn_k_sweep.png")

# ── 12. PREDICT A NEW, UNSEEN SAMPLE ─────────────────────────────────────────
#
# This is how you'd use the model in production.
# Supply the 4 measurements; the model returns the predicted species.
#
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # classic setosa dimensions
new_scaled = scaler.transform(new_sample)
pred_class  = knn.predict(new_scaled)[0]
pred_proba  = knn.predict_proba(new_scaled)[0]

print("\n" + "=" * 60)
print("PREDICTING A NEW SAMPLE")
print("=" * 60)
print(f"  Input:      sepal_len=5.1, sepal_wid=3.5, petal_len=1.4, petal_wid=0.2")
print(f"  Prediction: {iris.target_names[pred_class]}")
print(f"  Confidence: {dict(zip(iris.target_names, pred_proba.round(2)))}")

print("\nDone.  Check the four .png files for visualisations.")