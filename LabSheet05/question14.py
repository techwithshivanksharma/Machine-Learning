# question14.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = fetch_california_housing(as_frame=True)
# Use single feature for easy curve plotting
X = data.frame[['AveRooms']].values
y = data.frame['MedHouseVal'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear
lin = LinearRegression().fit(X_train, y_train)
y_pred_lin = lin.predict(X_test)
r2_lin = r2_score(y_test, y_pred_lin)

# Polynomial degree 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_p = poly.fit_transform(X_train)
X_test_p = poly.transform(X_test)
poly_model = LinearRegression().fit(X_train_p, y_train)
y_pred_poly = poly_model.predict(X_test_p)
r2_poly = r2_score(y_test, y_pred_poly)

print("Linear R2:", r2_lin)
print("Poly (deg2) R2:", r2_poly)

# Plot
xx = np.linspace(X.min(), X.max(), 200).reshape(-1,1)
plt.scatter(X_test, y_test, label='Actual', alpha=0.5)
plt.scatter(X_test, y_pred_lin, label='Linear Pred', alpha=0.5)
plt.scatter(X_test, y_pred_poly, label='Poly Pred', alpha=0.5)
plt.plot(xx, lin.predict(xx), color='red', label='Linear line')
plt.plot(xx, poly_model.predict(poly.transform(xx)), color='green', label='Poly curve')
plt.legend()
plt.show()
