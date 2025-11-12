from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"MSE: {mse:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR2: {r2:.3f}")

evaluate_model([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
