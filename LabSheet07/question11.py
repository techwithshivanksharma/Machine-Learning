# question11.py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np

X = load_iris().data
km = KMeans(n_clusters=3, random_state=42).fit(X)
labels = km.labels_
centroids = km.cluster_centers_

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=200, c='red', label='centroids')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend()
plt.title('Clusters and Centroids')
plt.show()
