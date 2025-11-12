# question03.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = fetch_california_housing(as_frame=True)
X = data.frame[['AveRooms']]  # single feature
y = data.frame['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Score (R^2) on test:", lr.score(X_test, y_test))

# Plot regression line
plt.scatter(X_test, y_test, alpha=0.5, label="Actual")
x_vals = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1,1)
y_pred_line = lr.predict(x_vals)
plt.plot(x_vals, y_pred_line, color='red', label="Regression line")
plt.xlabel('AveRooms')
plt.ylabel('MedHouseVal')
plt.legend()
plt.show()
