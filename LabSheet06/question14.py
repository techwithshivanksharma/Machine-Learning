import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
X, y = df[['rm']], df['medv']

model = LinearRegression().fit(X, y)
print("Before:", model.score(X, y))

filtered = df[df['medv'] < 50]
model2 = LinearRegression().fit(filtered[['rm']], filtered['medv'])
print("After:", model2.score(filtered[['rm']], filtered['medv']))

