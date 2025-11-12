import pandas as pd
import numpy as np

np.random.seed(42)
data = pd.DataFrame({'income': np.random.normal(50000, 15000, 100)})

def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    return df[(df[col] < lower) | (df[col] > upper)]

print(detect_outliers_iqr(data, 'income'))
