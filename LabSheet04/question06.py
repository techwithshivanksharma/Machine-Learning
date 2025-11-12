import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
data = pd.DataFrame({'income': np.random.normal(50000, 15000, 100)})
scaler = MinMaxScaler()
data['normalized'] = scaler.fit_transform(data[['income']])
print(data.head())
