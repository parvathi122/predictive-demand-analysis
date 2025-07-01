import pandas as pd
import joblib

# Load trained model and feature columns
model = joblib.load('trained_model.pkl')
X_columns = joblib.load('X_columns.pkl')  # make sure you saved this during training

# Example new data
new_data = pd.DataFrame([{
    'Ship Mode': 'Second Class',
    'Customer ID': 'AB-12345',
    'Segment': 'Consumer',
    'Country': 'United States',
    'City': 'New York',
    'State': 'New York',
    'Postal Code': 10001,
    'Region': 'East',
    'Product ID': 'FUR-CH-10001',
    'Category': 'Furniture',
    'Sub-Category': 'Chairs',
    'Product Name': 'Office Chair',
    'Quantity': 2,
    'Discount': 0.0,
    'Profit': 50
}])

# Preprocess new data
for col in new_data.select_dtypes(include='object').columns:
    new_data[col] = new_data[col].astype('category').cat.codes

# Align columns exactly with training features
new_data = new_data.reindex(columns=X_columns, fill_value=0)

# Predict
prediction = model.predict(new_data)
print(f"Predicted Sales: {prediction[0]:.2f}")
