import os
import zipfile
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# 📁 Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
zip_path = r"C:\Users\parva\Downloads\archive.zip"  # <-- Update if needed
extract_dir = os.path.join(base_dir, 'dataa', 'extracted')

# 📦 Extract Excel file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    print("📂 Files extracted:")
    for name in zip_ref.namelist():
        print(" -", name)
    excel_files = [f for f in zip_ref.namelist() if f.endswith('.xlsx')]
    if not excel_files:
        raise FileNotFoundError("❌ No .xlsx file found in the ZIP.")
    excel_filename = excel_files[0]

excel_path = os.path.join(extract_dir, excel_filename.replace('/', os.sep))

# 📊 Load Data
df = pd.read_excel(excel_path)
print("\n✅ Data Loaded.")

# 🗓️ Feature Engineering on Dates
if 'Order Date' in df.columns and 'Ship Date' in df.columns:
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Year'] = df['Order Date'].dt.year
    df['Shipping Delay'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['Month_sin'] = np.sin(2 * np.pi * df['Order Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Order Month'] / 12)
    df = df.drop(columns=['Order Date', 'Ship Date'])

# ❌ Drop Irrelevant Columns
df = df.drop(columns=['Row ID', 'Order ID', 'Customer Name'], errors='ignore')

# 🧹 Clean Data
df = df[(df['Sales'] > 0) & (df['Quantity'] > 0)]

# 🔠 Label Encoding
os.makedirs('encoders', exist_ok=True)
categorical_cols = df.select_dtypes(include='object').columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    joblib.dump(le, f'encoders/le_{col}.pkl')
print(f"✅ Encoded columns: {list(encoders.keys())}")

# 🎯 Demand Model Training
X_demand = df.drop(columns=['Quantity', 'Sales'])
y_demand = df['Quantity']
params = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1]
}
grid_d = GridSearchCV(XGBRegressor(random_state=42), param_grid=params, scoring='neg_mean_absolute_error', cv=3)
grid_d.fit(X_demand, y_demand)
demand_model = grid_d.best_estimator_
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_demand, y_demand, test_size=0.2, random_state=42)
demand_model.fit(X_train_d, y_train_d)
y_pred_d = demand_model.predict(X_test_d)
mae_d = mean_absolute_error(y_test_d, y_pred_d)

joblib.dump(demand_model, 'demand_model.pkl')
joblib.dump(X_demand.columns.tolist(), 'X_demand_columns.pkl')
joblib.dump(float(mae_d), 'demand_mae.pkl')

print(f"📦 Demand MAE: {mae_d:.2f}")

# 📈 Sales Model Training
X_sales = df.drop(columns=['Sales'])
y_sales = np.log1p(df['Sales'])  # log1p for stability
grid_s = GridSearchCV(XGBRegressor(random_state=42), param_grid=params, scoring='neg_mean_absolute_error', cv=3)
grid_s.fit(X_sales, y_sales)
sales_model = grid_s.best_estimator_
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sales, y_sales, test_size=0.2, random_state=42)
sales_model.fit(X_train_s, y_train_s)
y_pred_s = sales_model.predict(X_test_s)
mae_s = mean_absolute_error(y_test_s, y_pred_s)

joblib.dump(sales_model, 'trained_model.pkl')
joblib.dump(X_sales.columns.tolist(), 'X_columns.pkl')
joblib.dump(float(mae_s), 'sales_mae.pkl')

print(f"📈 Sales MAE: {mae_s:.2f}")

# 🔁 Summary
print("\n✅ Training Complete. Models & encoders saved.")
