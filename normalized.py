import pandas as pd

df = pd.read_csv('normalized.csv')

columns = df.columns.to_list()


print(df.head())