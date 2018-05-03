import pandas as pd

df = pd.read_csv("csv_file")
df['column_name']
df[['column_name_1', "column_name_2"]]
df['column_name'].max() # max value of that column
df.describe() # statistics of values in
df[df["column_name"] > some_value] #table of booleans
