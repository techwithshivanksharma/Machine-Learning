import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
X = df[['rm', 'lstat']]
y = df['medv']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model1 = LinearRegression().fit(X_train, y_train)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model2 = LinearRegression().fit(X_scaled, y)
print("Without Scaling:", model1.score(X_test, y_test))
print("With Scaling:", model2.score(X_scaled, y))
