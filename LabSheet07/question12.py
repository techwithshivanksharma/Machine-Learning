# question12.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame.copy()

km = KMeans(n_clusters=3, random_state=42)
df['cluster'] = km.fit_predict(df[iris.feature_names])

def cluster_stats(df, feature_cols, label_col='cluster'):
    stats = df.groupby(label_col)[feature_cols].agg(['count','mean'])
    return stats

print(cluster_stats(df, iris.feature_names))
