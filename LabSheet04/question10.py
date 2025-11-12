import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

np.random.seed(42)
data = pd.DataFrame({'income': np.random.normal(50000,15000,100)})
scaler = RobustScaler()
data['robust'] = scaler.fit_transform(data[['income']])
sns.histplot(data['robust'], kde=True)
plt.title("RobustScaler Result")
plt.show()
