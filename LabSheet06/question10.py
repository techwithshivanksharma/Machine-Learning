from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
