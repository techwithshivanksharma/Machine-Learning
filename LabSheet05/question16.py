# question16.py
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = fetch_california_housing(as_frame=True)
X = data.frame[['AveRooms']].values
y = data.frame['MedHouseVal'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for deg in [1,2,3,4]:
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    Xtr = poly.fit_transform(X_train)
    Xte = poly.transform(X_test)
    model = LinearRegression().fit(Xtr, y_train)
    print(f"Degree {deg}: Train R2 = {model.score(Xtr, y_train):.4f}, Test R2 = {model.score(Xte, y_test):.4f}")
