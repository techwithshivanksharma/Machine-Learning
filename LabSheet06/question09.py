from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import load_diabetes
from sklearn.metrics import make_scorer, mean_squared_error

X, y = load_diabetes(return_X_y=True)
kf = KFold(n_splits=5)
model = LinearRegression()
r2 = cross_val_score(model, X, y, cv=kf, scoring='r2')
mse = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

print("Average R2:", r2.mean())
print("Average MSE:", mse.mean())
