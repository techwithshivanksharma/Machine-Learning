import pandas as pd
import numpy as np

data = pd.DataFrame({
    'income': np.random.normal(50000,15000,100),
    'expenses': np.random.normal(20000,5000,100)
})
data['ratio'] = data['income'] / data['expenses']
data['total'] = data['income'] + data['expenses']
print(data.head())
