# question18.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = fetch_california_housing(as_frame=True)
X = data.frame[['AveRooms']].values
y = data.frame['MedHouseVal'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

deg = 3
poly = PolynomialFeatures(deg, include_bias=False)
scaler = StandardScaler()
Xtr = scaler.fit_transform(poly.fit_transform(X_train))
Xte = scaler.transform(poly.transform(X_test))

alphas = [0.01, 0.1, 1, 10, 100]
for a in alphas:
    model = Ridge(alpha=a).fit(Xtr, y_train)
    print(f"Ridge alpha={a}: Train R2={model.score(Xtr,y_train):.4f}, Test R2={model.score(Xte,y_test):.4f}")
