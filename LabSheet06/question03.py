from sklearn.metrics import mean_squared_error
import numpy as np

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print("MSE:", mean_squared_error(y_true, y_pred))
