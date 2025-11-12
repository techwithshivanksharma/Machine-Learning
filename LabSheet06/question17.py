from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "Linear": LinearRegression(),
    "Polynomial": Pipeline([('poly', PolynomialFeatures(2)), ('lin', LinearRegression())]),
    "DecisionTree": DecisionTreeRegressor()
}

for name, m in models.items():
    m.fit(X_train, y_train)
    print(name, "R2:", m.score(X_test, y_test))
