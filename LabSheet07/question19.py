# question19.py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

wine = load_wine(as_frame=True)
X = wine.data

# Reduce to 2D for visualization
pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(X)

km = KMeans(n_clusters=3, random_state=42).fit(X2)
plt.scatter(X2[:,0], X2[:,1], c=km.labels_, cmap='tab10', alpha=0.7)
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Wine clusters (PCA 2D)')
plt.show()
