import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

np.random.seed(42)
data = pd.DataFrame({'income': np.random.normal(50000, 15000, 100)})
data['minmax'] = MinMaxScaler().fit_transform(data[['income']])
data['standard'] = StandardScaler().fit_transform(data[['income']])

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(data['minmax'], kde=True)
plt.title('MinMaxScaler')

plt.subplot(1,2,2)
sns.histplot(data['standard'], kde=True)
plt.title('StandardScaler')
plt.show()
