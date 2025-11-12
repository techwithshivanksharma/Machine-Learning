# question13.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
# Use two features for demonstration
X = data.frame[['AveRooms', 'AveOccup']].copy()

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print("Polynomial feature names:", poly.get_feature_names_out(['AveRooms','AveOccup']))
print("Transformed shape:", X_poly.shape)
