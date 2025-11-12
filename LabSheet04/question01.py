import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Sample data
np.random.seed(42)
data = pd.DataFrame({'income': np.random.normal(50000, 15000, 100)})
data.loc[::10, 'income'] = data['income'] * 3  # add outliers

# Histogram before and after outlier removal
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(data['income'], kde=True)
plt.title("Before Outlier Removal")

z = np.abs(stats.zscore(data['income']))
filtered_data = data[z < 3]

plt.subplot(1,2,2)
sns.histplot(filtered_data['income'], kde=True)
plt.title("After Outlier Removal")
plt.show()
