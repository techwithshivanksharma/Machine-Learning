import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["SepalLength","SepalWidth","PetalLength","PetalWidth","Class"]
iris = pd.read_csv(url, names=columns)
print(iris.head())
