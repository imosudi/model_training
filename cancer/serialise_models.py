import os
import joblib
import numpy as np


OUTPUT_DIR = "cancer/outputs/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from trainings import scaler
from visualisations import *
from reports import *

np.random.seed(42)

# Models that benefit from the scaler are bundled with it;
# tree-based models that don't need it are saved without.
model_bundles = {
    "randomforest":       {"model": rf,  "scaler": scaler},
    "logisticregression": {"model": lr,  "scaler": scaler},
    "knn":                {"model": knn, "scaler": scaler},
    "decisiontree":       {"model": dt},
}

for name, bundle in model_bundles.items():
    path = os.path.join(OUTPUT_DIR, f"{name}_model.pkl")
    joblib.dump(bundle, path)
    print(f"{name} model saved → {path}")