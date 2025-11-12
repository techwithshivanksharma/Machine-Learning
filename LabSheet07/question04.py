# question04.py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame.copy()

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[iris.feature_names])

plt.figure(figsize=(8,6))
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris Clusters (2D)')
plt.show()
