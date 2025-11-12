import pandas as pd
import numpy as np
from scipy import stats

np.random.seed(42)
data = pd.DataFrame({'income': np.random.normal(50000, 15000, 100)})
z = np.abs(stats.zscore(data['income']))
filtered = data[z < 3]

print("Before:\n", data['income'].describe())
print("\nAfter:\n", filtered['income'].describe())
