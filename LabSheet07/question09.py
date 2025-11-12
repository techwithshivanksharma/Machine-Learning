# question09.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# create synthetic data with correlated features
rng = np.random.RandomState(42)
x = rng.normal(loc=0, scale=1, size=200)
# create y highly correlated with x
y = 2.5 * x + rng.normal(scale=0.1, size=200)
X = np.vstack([x,y]).T

# clustering
km = KMeans(n_clusters=2, random_state=42).fit(X)
plt.scatter(X[:,0], X[:,1], c=km.labels_, cmap='coolwarm')
plt.xlabel('x (feature1)')
plt.ylabel('y (feature2)')
plt.title('Clusters on Highly Correlated Features')
plt.show()

print("Centroids:\n", km.cluster_centers_)
