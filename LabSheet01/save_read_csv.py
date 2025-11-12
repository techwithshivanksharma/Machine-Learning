import pandas as pd

df = pd.read_csv("your_file.csv")  # replace with your CSV path
print(df.head())

df.to_csv("saved_data.csv", index=False)
df_new = pd.read_csv("saved_data.csv")
print(df_new.head())
