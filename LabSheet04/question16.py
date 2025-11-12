import pandas as pd

data = pd.DataFrame({'category': ['Low','Medium','High','Low','High']})
encoded = pd.get_dummies(data, columns=['category'], drop_first=True)
print(encoded)
