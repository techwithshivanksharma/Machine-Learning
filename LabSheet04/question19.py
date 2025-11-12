import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

data = pd.DataFrame({'income': np.random.normal(50000,15000,5),
                     'expenses': np.random.normal(20000,5000,5)})
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_data = poly.fit_transform(data)
print(pd.DataFrame(poly_data, columns=poly.get_feature_names_out(['income','expenses'])))
