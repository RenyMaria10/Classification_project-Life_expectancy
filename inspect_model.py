import pickle
import pandas as pd

# Load the model
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model:", type(model))
try:
    if hasattr(model, 'feature_names_in_'):
        print("Expected Features:", model.feature_names_in_)
    elif hasattr(model, 'n_features_in_'):
        print("Number of Features:", model.n_features_in_)
except Exception as e:
    print("Error getting features:", e)
