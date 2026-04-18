# AI Training Repository

Educational machine learning project covering classification algorithms and model evaluation techniques.

## Overview

This repository contains hands-on examples of machine learning classification models using popular datasets and scikit-learn. Explore fundamental concepts like data preprocessing, model training, hyperparameter tuning, and comprehensive evaluation metrics.

[![data-science](https://img.shields.io/badge/-data--science-informational?style=flat)](#) [![machine-learning](https://img.shields.io/badge/-machine--learning-blue?style=flat)](#) [![random-forest](https://img.shields.io/badge/-random--forest-brightgreen?style=flat)](#) [![linear-regression](https://img.shields.io/badge/-linear--regression-orange?style=flat)](#) [![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](#) [![python3](https://img.shields.io/badge/-python3-3776AB?style=flat&logo=python&logoColor=white)](#) [![classification](https://img.shields.io/badge/-classification-red?style=flat)](#) [![educational](https://img.shields.io/badge/-educational-purple?style=flat)](#) [![decision-tree](https://img.shields.io/badge/-decision--tree-yellow?style=flat)](#) [![model-evaluation](https://img.shields.io/badge/-model--evaluation-teal?style=flat)](#) [![feature-importance](https://img.shields.io/badge/-feature--importance-blueviolet?style=flat)](#) [![k-nearest-neighbors](https://img.shields.io/badge/-k--nearest--neighbors-green?style=flat)](#)

## Projects

### 1. Breast Cancer Diagnosis (`cancer/`)
**Dataset:** Breast Cancer Wisconsin (569 samples, 30 features)

**Files:**
- `cancer_models.py` - Main model training and evaluation
- `data_load.py` - Data loading and preprocessing utilities
- `trainings.py` - Training functions and pipelines
- `validations.py` - Model validation and cross-validation
- `visualisations.py` - Plotting and visualization functions
- `reports.py` - Report generation and metrics calculation
- `outputs/` - Directory for generated plots and model files

**Models:**
- Logistic Regression (linear classification)
- Random Forest (ensemble learning)
- k-Nearest Neighbors (k-NN)
- Decision Tree (max_depth=3)

**Features:**
- Full dataset exploration and statistical summary
- Train/test splitting with stratification
- Feature scaling for Logistic Regression
- Learning curves (train vs validation accuracy across dataset sizes)
- Cross-validation (5-fold CV) for model comparison
- Comprehensive evaluation metrics:
  - Accuracy scores
  - Confusion matrices with visualization
  - Classification reports (precision, recall, F1-score)
  - AUC-ROC scoring
- Feature importance analysis:
  - Random Forest: built-in `feature_importances_`
  - Logistic Regression: absolute coefficients
- Model serialization (joblib pickle files)

**Outputs:** 5 PNG visualizations (learning curves, confusion matrices, feature importance plots)

### 2. Single Model Training (`one/train_one.py`)
Training pipeline for individual machine learning models.

### 3. Multi-Model Comparison (`three/`)
Advanced model comparison and evaluation framework.

---

## Project Structure

```
model_training/
├── cancer/                    # Breast cancer classification project
│   ├── cancer_models.py      # Main training script
│   ├── data_load.py          # Data loading utilities
│   ├── trainings.py          # Training functions
│   ├── validations.py        # Validation methods
│   ├── visualisations.py     # Plotting functions
│   ├── reports.py            # Report generation
│   └── outputs/              # Generated files and plots
├── one/                      # Single model training
│   └── train_one.py
├── three/                    # Multi-model comparison
├── README.md                 # This file
└── LICENSE                   # Project license
```

---

## Core Concepts Covered

- **Data Exploration:** Shape, class distribution, summary statistics, pairplot visualization
- **Train/Test Splitting:** Stratified splits to preserve class proportions
- **Feature Scaling:** StandardScaler for distance-based models (k-NN, Logistic Regression)
- **Cross-Validation:** k-fold CV for robust model evaluation
- **Model Comparison:** Side-by-side evaluation of multiple algorithms
- **Evaluation Metrics:**
  - Accuracy
  - Confusion matrices
  - Classification reports (precision, recall, F1-score, support)
  - AUC-ROC score
- **Feature Importance:** Understanding which features drive predictions
- **Visualization:** Decision trees, confusion matrices, learning curves, feature importance plots
- **Production Ready:** Model serialization for deployment

---

## Requirements

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

## Usage

Run the Iris classification example:
```bash
python one/train_one.py
```

Run the Breast Cancer diagnosis example:
```bash
python cancer/cancer_train.py
```

---

## Educational Value

These scripts are designed as learning resources for:
- Understanding how different classifiers work
- Learning proper ML workflow (explore → split → scale → train → evaluate)
- Interpreting model outputs and evaluation metrics
- Comparing algorithm performance
- Extracting actionable insights from feature importance

---

## Repository Structure

```
ai_training/
├── README.md              # This file
├── one/
│   ├── train_one.py      # Iris classification
│   ├── iris_pairplot.png
│   ├── iris_confusion_matrices.png
│   ├── iris_decision_tree.png
│   └── iris_knn_k_sweep.png
├── cancer/
│   ├── cancer_train.py   # Breast cancer diagnosis
│   ├── confusion_matrix_logistic.png
│   ├── confusion_matrix_randomforest.png
│   ├── random_forest_feature_importances.png
│   ├── logistic_regression_feature_importances.png
│   ├── randomforest_model.pkl
│   └── logisticregression_model.pkl
└── three/                 # Future project
```

---

## Notes

- All random states are fixed (42) for reproducibility
- Stratified splitting ensures balanced train/test distributions
- Feature scaling is crucial for distance-based models
- Cross-validation provides robust performance estimates
- Confusion matrices reveal which classes are confused with each other
- Feature importance helps understand model decisions

