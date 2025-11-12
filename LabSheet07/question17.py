# question17.py
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X = iris.data[:60]  # smaller subset for clarity

Z = linkage(X, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='lastp', p=12, show_leaf_counts=True)
plt.title('Dendrogram (ward linkage)')
plt.xlabel('Cluster size')
plt.ylabel('Distance')
plt.show()
