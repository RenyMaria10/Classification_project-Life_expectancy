import pandas as pd
df = pd.read_csv('Life Expectancy Data.csv')
print("Status values:", df['Status'].unique())
print("Status value counts:")
print(df['Status'].value_counts())
