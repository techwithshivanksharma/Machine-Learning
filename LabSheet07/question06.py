# question06.py
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

X = load_iris().data
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
labels = kmeans.labels_
print("Silhouette Score (k=3):", silhouette_score(X, labels))
