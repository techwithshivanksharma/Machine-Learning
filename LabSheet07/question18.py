# question18.py
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score
import numpy as np

X = load_iris().data
methods = ['ward','complete','average','single']

for m in methods:
    Z = linkage(X, method=m)
    labels = fcluster(Z, t=3, criterion='maxclust')
    # adjusted rand needs labels starting at 0
    from sklearn.metrics import adjusted_rand_score
    print(f"Method: {m}, ARI (vs true):", adjusted_rand_score(load_iris().target, labels-1))
