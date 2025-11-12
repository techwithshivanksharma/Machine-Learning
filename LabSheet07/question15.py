# question15.py
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import joblib
import numpy as np

X = load_iris().data
km = KMeans(n_clusters=3, random_state=42).fit(X)

joblib.dump(km, 'kmeans_iris.joblib')
print("Saved kmeans_iris.joblib")

# load and predict on new sample (use means)
loaded = joblib.load('kmeans_iris.joblib')
sample = X.mean(axis=0).reshape(1,-1)
print("Predicted cluster for mean sample:", loaded.predict(sample))
