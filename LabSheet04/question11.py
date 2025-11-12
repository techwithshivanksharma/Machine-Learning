import pandas as pd
import numpy as np

np.random.seed(42)
data = pd.DataFrame({'income': np.random.normal(50000, 15000, 100)})

def manual_normalize(s):
    return (s - s.min()) / (s.max() - s.min())

data['manual_norm'] = manual_normalize(data['income'])
print(data.head())
