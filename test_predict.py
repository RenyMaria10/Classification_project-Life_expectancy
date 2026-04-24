import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = joblib.load('knn_model.pkl')
df = pd.read_csv('Life Expectancy Data.csv')

FEATURES = [
    'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 
    'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 
    'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 
    ' thinness  1-19 years', ' thinness 5-9 years', 
    'Income composition of resources', 'Schooling'
]

X_all = df[FEATURES].fillna(df[FEATURES].median())
scaler = StandardScaler()
scaler.fit(X_all)

X_test = scaler.transform(X_all.head(5))
preds = model.predict(X_test)
print("True labels:", df['Status'].head(5).values)
print("Predicted labels:", preds)

X_test_dev = scaler.transform(X_all[df['Status'] == 'Developed'].head(5))
preds_dev = model.predict(X_test_dev)
print("True labels Developed:", ['Developed']*5)
print("Predicted labels Developed:", preds_dev)
