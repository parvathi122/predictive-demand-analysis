import os
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# STEP 1: Load CSV from ZIP
base_dir = os.path.dirname(os.path.abspath(__file__))
zip_path = os.path.join(base_dir, 'dataa', 'archive.zip')

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    csv_filename = [f for f in zip_ref.namelist() if f.endswith('.csv')][0]
    with zip_ref.open(csv_filename) as file:
        df = pd.read_csv(file)

print("✅ Data Loaded:")
print(df.head())

# STEP 2: Drop ID columns (adjust as per your data)
df = df.drop(columns=['Row ID', 'Order ID'], errors='ignore')

# STEP 3: Handle categorical columns (label encoding)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category').cat.codes

# STEP 4: Features and target
X = df.drop(columns=['Sales'])
y = df['Sales']

# STEP 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 6: Train the model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# STEP 7: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"📊 Mean Squared Error: {mse:.2f}")
import joblib
joblib.dump(model, "trained_model.pkl")
# Save columns used during training
X_columns = X.columns.tolist()
joblib.dump(X_columns, 'X_columns.pkl')
