# question12.py
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Ensure model is present; if not, train & save (quick fallback)
try:
    model = joblib.load('linear_regression_model.joblib')
except Exception:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    data = fetch_california_housing(as_frame=True)
    X = data.frame.drop(columns=['MedHouseVal'])
    y = data.frame['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    joblib.dump(model, 'linear_regression_model.joblib')

data = fetch_california_housing(as_frame=True)
feature_names = list(data.frame.drop(columns=['MedHouseVal']).columns)

def predict_from_dict(feature_dict):
    # feature_dict should map feature_name -> value
    df = pd.DataFrame([feature_dict], columns=feature_names)
    return model.predict(df)[0]

# Example usage
mean_sample = data.frame.drop(columns=['MedHouseVal']).mean().to_dict()
print("Prediction for mean sample:", predict_from_dict(mean_sample))
