# question07.py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris

X = load_iris().data
ks = range(2,11)
scores = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    scores.append(silhouette_score(X, km.labels_))

plt.plot(ks, scores, '-o')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for k=2..10')
plt.show()
print("Best k:", ks[int(scores.index(max(scores)))])
