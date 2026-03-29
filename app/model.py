import joblib

model = joblib.load('../models/property_price_predictor.pkl')
feature_cols = joblib.load('../models/feature_columns.pkl')