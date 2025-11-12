import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

X = np.random.rand(100, 1) * 10
y = 2*X + X**2 + np.random.randn(100, 1) * 5

linear = LinearRegression().fit(X, y)
poly = Pipeline([('poly', PolynomialFeatures(degree=2)),
                 ('lin', LinearRegression())]).fit(X, y)

plt.scatter(X, y - linear.predict(X), label="Linear Residuals")
plt.scatter(X, y - poly.predict(X), label="Poly Residuals")
plt.axhline(0, color='r')
plt.legend(); plt.show()
