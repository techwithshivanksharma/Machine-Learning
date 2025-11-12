# question17.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

data = fetch_california_housing(as_frame=True)
X = data.frame[['AveRooms']].values
y = data.frame['MedHouseVal'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

degrees = list(range(1,9))
train_scores = []
test_scores = []

for deg in degrees:
    poly = PolynomialFeatures(deg, include_bias=False)
    Xtr = poly.fit_transform(X_train)
    Xte = poly.transform(X_test)
    model = LinearRegression().fit(Xtr, y_train)
    train_scores.append(model.score(Xtr, y_train))
    test_scores.append(model.score(Xte, y_test))

plt.plot(degrees, train_scores, label='Train R2')
plt.plot(degrees, test_scores, label='Test R2')
plt.xlabel('Polynomial Degree')
plt.ylabel('R2 Score')
plt.legend()
plt.title('Train vs Test R2 by Degree')
plt.show()
