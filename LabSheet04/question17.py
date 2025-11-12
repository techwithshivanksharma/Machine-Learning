import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.DataFrame({'category': ['Low','Medium','High','Medium','Low']})
encoder = LabelEncoder()
data['encoded'] = encoder.fit_transform(data['category'])
print(data)
