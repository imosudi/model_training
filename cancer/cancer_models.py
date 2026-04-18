# Install once: pip install scikit-learn pandas numpy matplotlib
import time
import os
import joblib
import numpy as np

from data_load import *
from trainings import *
from validations import *
from visualisations import *
from reports import *


np.random.seed(42)  # reproducibility

# Create output directory if it doesn't exist
os.makedirs("cancer/outputs/models", exist_ok=True)


# Save model + scaler as a bundle (Random Forest doesn't need scaler, but we include it for completeness)
bundle_rf = {"model": rf, "scaler": scaler}
joblib.dump(bundle_rf, "cancer/outputs/models/randomforest_model.pkl")
print("Random Forest model saved. cancer/outputs/models/randomforest_model.pkl")

# Save Logistic Regression model + scaler as a bundle
bundle_lr = {"model": lr, "scaler": scaler}
joblib.dump(bundle_lr, "cancer/outputs/models/logisticregression_model.pkl")
print("Logistic Regression model saved. cancer/outputs/models/logisticregression_model.pkl")

# Save k-NN model + scaler as a bundle
bundle_knn = {"model": knn, "scaler": scaler}
joblib.dump(bundle_knn, "cancer/outputs/models/knn_model.pkl")
print("k-NN model saved. cancer/outputs/models/knn_model.pkl")

# Save Decision Tree model as a bundle (no scaler needed)
bundle_dt = {"model": dt}
joblib.dump(bundle_dt, "cancer/outputs/models/decisiontree_model.pkl")
print("Decision Tree model saved. cancer/outputs/models/decisiontree_model.pkl")    
