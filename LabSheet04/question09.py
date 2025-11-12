import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
data = pd.DataFrame({
    'income': np.random.normal(50000, 15000, 100),
    'expenses': np.random.normal(20000, 5000, 100),
    'category': np.random.choice(['A','B','C'], 100)
})
scaler = MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(data[['income','expenses']]), columns=['income','expenses'])
final = pd.concat([scaled, data[['category']]], axis=1)
print(final.head())
