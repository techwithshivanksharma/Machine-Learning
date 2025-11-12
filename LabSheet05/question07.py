# question07.py
# Similar to question03 but more explicit plotting setup
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = fetch_california_housing(as_frame=True)
X = data.frame[['AveRooms']]
y = data.frame['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, label='Actual', alpha=0.6)
plt.scatter(X_test, y_pred, label='Predicted', alpha=0.6)
# regression line
xx = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1,1)
plt.plot(xx, lr.predict(xx), color='red', linewidth=2, label='Regression line')
plt.xlabel('AveRooms')
plt.ylabel('MedHouseVal')
plt.legend()
plt.title('Simple Linear Regression: Actual vs Predicted')
plt.show()
