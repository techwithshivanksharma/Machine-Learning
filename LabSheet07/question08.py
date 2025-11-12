# question08.py
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
import numpy as np

X = load_iris().data
km_raw = KMeans(n_clusters=3, random_state=42).fit(X)
score_raw = silhouette_score(X, km_raw.labels_)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
km_scaled = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
score_scaled = silhouette_score(X_scaled, km_scaled.labels_)

print(f"Silhouette raw: {score_raw:.4f}, scaled: {score_scaled:.4f}")
