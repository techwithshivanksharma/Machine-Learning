import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

X = pd.DataFrame({
    'a': np.random.randn(100),
    'b': np.random.randn(100),
    'c': np.random.randn(100),
    'd': np.random.randn(100),
    'e': np.random.randn(100),
    'f': np.random.randn(100)
})
y = np.random.choice([0,1], size=100)
selector = SelectKBest(score_func=f_classif, k=5)
fit = selector.fit(X, y)
print("Top 5 features:", X.columns[selector.get_support()])
