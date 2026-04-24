import joblib
try:
    model = joblib.load('knn_model.pkl')
    print("Classes:", model.classes_)
except Exception as e:
    print("Error:", e)
