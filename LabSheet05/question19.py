# question19.py
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
import numpy as np

data = fetch_california_housing(as_frame=True)
X = data.frame[['AveRooms']].values
y = data.frame['MedHouseVal'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

deg = 3
poly = PolynomialFeatures(deg, include_bias=False)
scaler = StandardScaler()
Xtr = scaler.fit_transform(poly.fit_transform(X_train))
Xte = scaler.transform(poly.transform(X_test))

alpha = 1.0
ridge = Ridge(alpha=alpha).fit(Xtr, y_train)
lasso = Lasso(alpha=alpha, max_iter=10000).fit(Xtr, y_train)

print("Ridge R2 (test):", ridge.score(Xte, y_test))
print("Lasso R2 (test):", lasso.score(Xte, y_test))
print("\nRidge coefs (first 10):", ridge.coef_[:10])
print("Lasso coefs (first 10):", lasso.coef_[:10])
