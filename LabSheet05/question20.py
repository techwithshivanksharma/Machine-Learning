# question20.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

data = fetch_california_housing(as_frame=True)
X = data.frame[['AveRooms']].values
y = data.frame['MedHouseVal'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

deg = 3
poly = PolynomialFeatures(deg, include_bias=False)
scaler = StandardScaler()
Xtr = scaler.fit_transform(poly.fit_transform(X_train))

alphas = np.logspace(-3, 3, 30)
ridge_coefs = []
lasso_coefs = []

for a in alphas:
    ridge = Ridge(alpha=a).fit(Xtr, y_train)
    lasso = Lasso(alpha=a, max_iter=10000).fit(Xtr, y_train)
    ridge_coefs.append(ridge.coef_)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for i in range(ridge_coefs.shape[1]):
    plt.plot(alphas, ridge_coefs[:,i], label=f'coef_{i}')
plt.xscale('log')
plt.xlabel('alpha')
plt.title('Ridge coefficients vs alpha')

plt.subplot(1,2,2)
for i in range(lasso_coefs.shape[1]):
    plt.plot(alphas, lasso_coefs[:,i], label=f'coef_{i}')
plt.xscale('log')
plt.xlabel('alpha')
plt.title('Lasso coefficients vs alpha')

plt.tight_layout()
plt.show()
