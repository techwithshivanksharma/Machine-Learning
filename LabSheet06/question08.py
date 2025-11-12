import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.random.rand(100, 1) * 10
y = 3 * X + np.random.randn(100, 1) * 2
model = LinearRegression().fit(X, y)
residuals = y - model.predict(X)

plt.scatter(X, residuals)
plt.axhline(0, color='r')
plt.title("Residual Plot")
plt.show()

