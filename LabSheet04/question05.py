import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
data = pd.DataFrame({'income': np.random.normal(50000, 15000, 100)})

data['log_income'] = np.log1p(data['income'])
sns.histplot(data['log_income'], kde=True)
plt.title("Log-transformed Income")
plt.show()
