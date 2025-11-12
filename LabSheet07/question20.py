# question20.py
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

iris = load_iris()
X = iris.data
y_true = iris.target

Z = linkage(X, method='ward')
labels = fcluster(Z, t=3, criterion='maxclust') - 1  # convert to 0-based
print("Adjusted Rand Index:", adjusted_rand_score(y_true, labels))
print("NMI:", normalized_mutual_info_score(y_true, labels))
