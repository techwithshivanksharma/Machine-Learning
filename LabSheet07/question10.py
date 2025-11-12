# question10.py
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

X = load_iris().data

km_default = KMeans(n_clusters=3, random_state=42)  # since sklearn 1.4 default n_init='auto' but setting to explicit default behavior can vary
km_custom = KMeans(n_clusters=3, n_init=10, random_state=42)

labels_def = km_default.fit_predict(X)
labels_custom = km_custom.fit_predict(X)

print("Silhouette default:", silhouette_score(X, labels_def))
print("Silhouette n_init=10:", silhouette_score(X, labels_custom))

