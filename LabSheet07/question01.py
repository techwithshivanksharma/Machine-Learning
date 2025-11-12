# question01.py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame
df['target'] = iris.target

# Pairplot
sns.pairplot(df, hue='target', vars=df.columns[:4])
plt.suptitle("Iris Pairplot", y=1.02)
plt.show()
