# question13.py
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd

wine = load_wine(as_frame=True)
X = wine.data
y_true = wine.target

km = KMeans(n_clusters=3, random_state=42).fit(X)
print("Inertia:", km.inertia_)
print("Adjusted Rand Index vs true labels:", adjusted_rand_score(y_true, km.labels_))
