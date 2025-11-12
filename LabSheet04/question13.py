import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

np.random.seed(42)
data = pd.DataFrame({'income': np.random.normal(50000,15000,100)})
pt = PowerTransformer()
data['power'] = pt.fit_transform(data[['income']])
sns.histplot(data['power'], kde=True)
plt.title("Power Transformed Income")
plt.show()
