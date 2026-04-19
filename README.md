# Basic AI/ML Model Training

Educational machine learning project covering classical ML and TensorFlow classification workflows, evaluation, visualisation, and model serialisation.

## Overview

This repository contains hands-on classification examples built with scikit-learn and TensorFlow. It covers data preprocessing, model training, cross-validation, reporting, visualisation, and model export.

Breast Cancer Diagnosis now compares Logistic Regression, Random Forest, k-NN, Decision Tree, and a TensorFlow neural network on the Breast Cancer Wisconsin dataset. The workflow includes data exploration, train/test splitting, feature scaling, cross-validation, classification reports, ROC-AUC, confusion matrices, learning curves, feature importance analysis, and training-vs-validation plots.

[![data-science](https://img.shields.io/badge/-data--science-informational?style=flat)](#) [![machine-learning](https://img.shields.io/badge/-machine--learning-blue?style=flat)](#) [![tensorflow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](#) [![keras](https://img.shields.io/badge/-Keras-D00000?style=flat&logo=keras&logoColor=white)](#) [![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](#) [![python3](https://img.shields.io/badge/-python3-3776AB?style=flat&logo=python&logoColor=white)](#) [![pandas](https://img.shields.io/badge/-pandas-150458?style=flat&logo=pandas&logoColor=white)](#) [![numpy](https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white)](#) [![matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat)](#) [![seaborn](https://img.shields.io/badge/-Seaborn-4C72B0?style=flat)](#) [![classification](https://img.shields.io/badge/-classification-red?style=flat)](#) [![model-evaluation](https://img.shields.io/badge/-model--evaluation-teal?style=flat)](#) [![cross-validation](https://img.shields.io/badge/-cross--validation-0A9396?style=flat)](#) [![roc-auc](https://img.shields.io/badge/-ROC--AUC-7B2CBF?style=flat)](#) [![feature-importance](https://img.shields.io/badge/-feature--importance-blueviolet?style=flat)](#) [![random-forest](https://img.shields.io/badge/-random--forest-brightgreen?style=flat)](#) [![linear-regression](https://img.shields.io/badge/-linear--regression-orange?style=flat)](#) [![decision-tree](https://img.shields.io/badge/-decision--tree-yellow?style=flat)](#) [![k-nearest-neighbors](https://img.shields.io/badge/-k--nearest--neighbors-green?style=flat)](#) [![educational](https://img.shields.io/badge/-educational-purple?style=flat)](#)

## Projects

### 1. Breast Cancer Diagnosis (`cancer/`)
**Dataset:** Breast Cancer Wisconsin (569 samples, 30 features)

**Files:**
- `serialise_models.py` - Main model serialisation script
- `data_load.py` - Data loading and preprocessing utilities
- `trainings.py` - Training functions and pipelines
- `validations.py` - Model validation and cross-validation
- `visualisations.py` - Plotting and visualisation functions
- `reports.py` - Report generation and metrics calculation
- `outputs/` - Directory for generated plots and model files

**Models:**
- Logistic Regression
- Random Forest
- k-Nearest Neighbors (k-NN)
- Decision Tree
- TensorFlow dense neural network

**Features:**
- Full dataset exploration and statistical summary
- Train/test splitting with stratification
- Feature scaling for Logistic Regression, k-NN, and TensorFlow
- TensorFlow training with model summary, epoch logs, validation tracking, and early stopping
- Cross-validation for all models, including manual TensorFlow CV
- Learning curves for all models
- Comprehensive evaluation metrics:
  - Accuracy
  - Classification reports
  - Confusion matrices
  - ROC-AUC
- Feature importance analysis:
  - Random Forest and Decision Tree: built-in importances
  - Logistic Regression: absolute coefficients
  - k-NN and TensorFlow: permutation importance
- Unified training-history plots for train vs validation loss and accuracy
- Model serialisation:
  - scikit-learn models saved as `.pkl`
  - TensorFlow model saved as `.keras`
  - TensorFlow scaler saved separately as `.pkl`

**Generated outputs include:**
- `training_validation_curves.png`
- Per-model learning curves
- Per-model confusion matrices
- Per-model feature importance plots
- Serialised model artifacts in `cancer/outputs/models/`

### 2. Single Model Training (`one/train_one.py`)
Training pipeline for individual machine learning models.

### 3. Multi-Model Comparison (`three/`)
Advanced model comparison and evaluation framework.

---

## Project Structure

```
model_training/
├── cancer/                    # Breast cancer classification project
│   ├── serialise_models.py   # Model serialisation script
│   ├── data_load.py          # Data loading utilities
│   ├── trainings.py          # Training functions
│   ├── validations.py        # Validation methods
│   ├── visualisations.py     # Plotting functions
│   ├── reports.py            # Report generation
│   └── outputs/              # Generated files and plots
├── one/                      # Single model training
│   └── train_one.py
├── three/                    # Multi-model comparison
├── requirements.txt          # dependency
├── README.md                 # This file
└── LICENSE                   # Project license
```

---

## Core Concepts Covered

- **Data Exploration:** Shape, class distribution, summary statistics, pairplot visualisation
- **Train/Test Splitting:** Stratified splits to preserve class proportions
- **Feature Scaling:** StandardScaler for distance-based and neural-network models
- **Cross-Validation:** k-fold CV for robust model evaluation
- **Model Comparison:** Side-by-side evaluation of multiple algorithms
- **Deep Learning Basics:** Dense neural networks with TensorFlow/Keras
- **Evaluation Metrics:**
  - Accuracy
  - Confusion matrices
  - Classification reports (precision, recall, F1-score, support)
  - AUC-ROC score
- **Feature Importance:** Understanding which features drive predictions
- **Visualisation:** Training-validation curves, confusion matrices, learning curves, feature importance plots
- **Serialisation:** Exporting sklearn and TensorFlow models for reuse

---

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Run the Iris classification example:
```bash
python one/train_one.py
```

Run the Breast Cancer diagnosis example:
```bash
python cancer/serialise_models.py
```

This command trains the models, generates reports and visualisations, and writes serialised artifacts to `cancer/outputs/models/`.

---

## Educational Value

These scripts are designed as learning resources for:
- Understanding how different classifiers work
- Learning proper ML workflow (explore → split → scale → train → evaluate)
- Interpreting model outputs and evaluation metrics
- Comparing algorithm performance
- Extracting actionable insights from feature importance

---


## Notes

- All random states are fixed (42) for reproducibility
- Stratified splitting ensures balanced train/test distributions
- Feature scaling is crucial for distance-based models and the TensorFlow model
- Cross-validation provides robust performance estimates
- Confusion matrices reveal which classes are confused with each other
- Feature importance helps understand model decisions
- TensorFlow uses CPU if CUDA drivers are not available
