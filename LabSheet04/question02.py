import pandas as pd
import numpy as np
from scipy import stats

np.random.seed(42)
data = pd.DataFrame({'income': np.random.normal(50000, 15000, 100)})
data['zscore'] = stats.zscore(data['income'])
outliers = data[data['zscore'].abs() > 3]
print(outliers)
