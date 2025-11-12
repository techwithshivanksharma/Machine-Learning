# question14.py
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
km = KMeans(n_clusters=4, random_state=42).fit(X)

plt.scatter(X[:,0], X[:,1], c=km.labels_, cmap='tab10', alpha=0.7)
plt.title('K-Means on make_blobs synthetic data')
plt.show()
