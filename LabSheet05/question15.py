# question15.py
# Uses code similar to question14 but focuses on smoother curve plot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = fetch_california_housing(as_frame=True)
X = data.frame[['AveRooms']].values
y = data.frame['MedHouseVal'].values

# Fit polynomial degree 3
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)

# smooth curve
xx = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
yy = model.predict(poly.transform(xx))

plt.scatter(X, y, alpha=0.3, label='Data')
plt.plot(xx, yy, color='red', linewidth=2, label='Poly deg3')
plt.xlabel('AveRooms')
plt.ylabel('MedHouseVal')
plt.legend()
plt.show()
