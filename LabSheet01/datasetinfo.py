import pandas as pd

df = pd.read_csv("your_file.csv")  # replace with your CSV path
print(df.head())

print("Shape:", df.shape)
print("Columns:", df.columns)
