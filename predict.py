import pandas as pd
import joblib
import os
import numpy as np

# Load trained models
sales_model = joblib.load('trained_model.pkl')
demand_model = joblib.load('demand_model.pkl')

# Load feature columns
X_columns = joblib.load('X_columns.pkl')
X_demand_columns = joblib.load('X_demand_columns.pkl')

# Load MAE values
sales_mae = joblib.load('sales_mae.pkl')
demand_mae = joblib.load('demand_mae.pkl')

# Load encoders
encoders = {}
encoders_dir = 'encoders'
for filename in os.listdir(encoders_dir):
    if filename.startswith('le_') and filename.endswith('.pkl'):
        col_name = filename.replace('le_', '').replace('.pkl', '')
        encoders[col_name] = joblib.load(os.path.join(encoders_dir, filename))

# Fields to ignore (time-based + target + shipping delay)
ignore_fields = {'Sales', 'Quantity', 'Order Month', 'Month_sin', 'Month_cos', 'Shipping Delay'}
input_fields = sorted(set(X_columns + X_demand_columns) - ignore_fields)

print("\n📥 Please enter the following input values:")

user_input = {}
for field in input_fields:
    if field in encoders:
        print(f"\n💡 Options for {field}: {list(encoders[field].classes_)}")

    value = input(f"{field}: ").strip()

    # Try converting to numeric
    if value.replace('.', '', 1).isdigit():
        value = float(value) if '.' in value else int(value)
    elif field in encoders:
        lower_classes = [cls.lower() for cls in encoders[field].classes_]
        if value.lower() in lower_classes:
            matched = encoders[field].classes_[lower_classes.index(value.lower())]
            value = encoders[field].transform([matched])[0]
        else:
            print(f"⚠️ Unseen label for '{field}' – assigning -1")
            value = -1
    else:
        print(f"⚠️ Unexpected string input for {field}, setting to -1")
        value = -1

    user_input[field] = value

# Create input DataFrame
input_df = pd.DataFrame([user_input])

# Ensure numeric columns are clean
for col in input_df.columns:
    if col not in encoders and input_df[col].dtype == 'object':
        try:
            input_df[col] = pd.to_numeric(input_df[col])
        except:
            input_df[col] = -1

# Align with model features
X_sales_input = input_df.reindex(columns=X_columns, fill_value=0)
X_demand_input = input_df.reindex(columns=X_demand_columns, fill_value=0)

# Make predictions
predicted_log_sales = sales_model.predict(X_sales_input)[0]
predicted_sales = np.expm1(predicted_log_sales)

predicted_quantity = round(demand_model.predict(X_demand_input)[0])

# Clamp values for safety
predicted_sales = max(0, predicted_sales)
predicted_quantity = max(0, predicted_quantity)

# Print result
print("\n🔮 ----- Prediction Results -----")
print(f"📈 Predicted Sales: ₹{predicted_sales:.2f}")
print(f"📏 Sales Range: ₹{predicted_sales - sales_mae:.2f} to ₹{predicted_sales + sales_mae:.2f}")
print(f"\n📦 Predicted Demand (Quantity): {predicted_quantity}")
print(f"📏 Quantity Range: {predicted_quantity - demand_mae:.2f} to {predicted_quantity + demand_mae:.2f}")
