import joblib
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Life Expectancy Data.csv')

# Look at the numeric columns
numeric_df = df.select_dtypes(include=[np.number])
print("Numeric columns:", list(numeric_df.columns))
print("Number of numeric columns:", len(numeric_df.columns))

# Let's try to pass the first row to the model, filling NaNs
try:
    model = joblib.load('knn_model.pkl')
    # Try using all 17 features maybe it's just the numeric ones dropping some
    if len(numeric_df.columns) == 17:
        print("Model probably uses all numeric features.")
    
except Exception as e:
    print(e)
