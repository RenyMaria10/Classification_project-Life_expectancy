import joblib
import pandas as pd

try:
    model = joblib.load('knn_model.pkl')
    print("Model:", type(model))
    if hasattr(model, 'feature_names_in_'):
        print("Expected Features:", list(model.feature_names_in_))
    elif hasattr(model, 'n_features_in_'):
        print("Number of Features:", model.n_features_in_)
except Exception as e:
    print("Error getting features with joblib:", e)
