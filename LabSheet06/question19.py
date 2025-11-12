import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({'Hours': [1,2,3,4,5,6,7,8,9,10],
                     'Scores': [10,20,30,35,45,50,60,65,80,85]})
X, y = data[['Hours']], data['Scores']
model = LinearRegression().fit(X, y)
print("Predicted score for 7.5 hours:", model.predict([[7.5]])[0])
