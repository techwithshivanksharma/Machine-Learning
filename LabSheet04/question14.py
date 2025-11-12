import pandas as pd

data = pd.DataFrame({'date': pd.date_range('2023-01-01', periods=5, freq='M')})
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
print(data)
