import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
data = pd.DataFrame({'income': np.random.normal(50000, 15000, 100)})
scaler = StandardScaler()
data['standardized'] = scaler.fit_transform(data[['income']])
print('Mean:', data['standardized'].mean())
print('Std:', data['standardized'].std())
