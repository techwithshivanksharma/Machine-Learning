# question11.py
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing

# Load model
model = joblib.load('linear_regression_model.joblib')

# Example: create one sample with same features (use mean values)
data = fetch_california_housing(as_frame=True)
sample = data.frame.drop(columns=['MedHouseVal']).mean().to_frame().T
print("Sample features:\n", sample)

pred = model.predict(sample)
print("Prediction for sample:", pred)
