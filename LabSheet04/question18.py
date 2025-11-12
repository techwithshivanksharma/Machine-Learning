import pandas as pd
import numpy as np

data = pd.DataFrame({'age': np.random.randint(18,70,100)})
bins = [0,25,40,60,100]
labels = ['Young','Adult','Middle Age','Senior']
data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels)
print(data.head())
