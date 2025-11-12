# question10.py
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = fetch_california_housing(as_frame=True)
X = data.frame.drop(columns=['MedHouseVal'])
y = data.frame['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)

# Save
joblib.dump(model, 'linear_regression_model.joblib')
print("Model saved to linear_regression_model.joblib")
