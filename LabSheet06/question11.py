import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
models = {'Linear': LinearRegression(), 'Ridge': Ridge(), 'Lasso': Lasso()}
results = {name: cross_val_score(m, X, y, cv=5, scoring='r2').mean() for name, m in models.items()}
print(pd.DataFrame(list(results.items()), columns=['Model', 'Avg R2']))
