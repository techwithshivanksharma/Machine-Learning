# question09.py
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

X_sm = sm.add_constant(X)  # adds intercept term
model = sm.OLS(y, X_sm).fit()
print(model.summary())
