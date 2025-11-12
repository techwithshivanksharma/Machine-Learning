from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import numpy as np

X, y = load_diabetes(return_X_y=True)
train_sizes, train_scores, test_scores = learning_curve(LinearRegression(), X, y, cv=5)
plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Test")
plt.legend(); plt.show()
