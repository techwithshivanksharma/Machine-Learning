# question16.py
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score

iris = load_iris()
X = iris.data
y_true = iris.target

agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)
print("ARI vs true labels:", adjusted_rand_score(y_true, labels))
