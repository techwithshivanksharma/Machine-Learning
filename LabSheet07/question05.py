# question05.py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

data = load_iris().data
inertias = []
ks = range(1,11)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(data)
    inertias.append(km.inertia_)

plt.plot(ks, inertias, '-o')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.xticks(ks)
plt.show()
