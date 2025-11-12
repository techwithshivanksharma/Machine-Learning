from sklearn.metrics import mean_squared_error
import numpy as np

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mse = mean_squared_error(y_true, y_pred)
print("RMSE:", np.sqrt(mse))
