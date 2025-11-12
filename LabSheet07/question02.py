# question02.py
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
X = iris.data

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

print("Inertia:", kmeans.inertia_)
print("Cluster centers:\n", kmeans.cluster_centers_)
