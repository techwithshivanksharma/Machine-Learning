import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
data = pd.DataFrame({'expenses': np.random.normal(20000, 5000, 100)})

sns.histplot(data['expenses'], kde=True)
plt.title("Before Log Transform")
plt.show()

data['log_expenses'] = np.log1p(data['expenses'])
sns.histplot(data['log_expenses'], kde=True)
plt.title("After Log Transform")
plt.show()
