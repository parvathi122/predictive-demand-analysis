import streamlit as st
import pandas as pd
import joblib

# Load trained model and column list
model = joblib.load('trained_model.pkl')
X_columns = joblib.load('X_columns.pkl')

st.title("📦 Predictive Demand Analysis")

st.subheader("Enter Order Details")

# Collect user input
quantity = st.number_input("Quantity", min_value=1, value=2)
discount = st.number_input("Discount", min_value=0.0, max_value=1.0, value=0.1)
profit = st.number_input("Profit", value=50.0)
category = st.selectbox("Category", ['Furniture', 'Office Supplies', 'Technology'])

# Create input DataFrame
input_df = pd.DataFrame([{
    'Quantity': quantity,
    'Discount': discount,
    'Profit': profit,
    'Category': category
}])

# Encode categorical columns
for col in input_df.select_dtypes(include='object').columns:
    input_df[col] = input_df[col].astype('category').cat.codes

# Match columns with training features
input_df = input_df.reindex(columns=X_columns, fill_value=0)

# Predict and display result
if st.button("Predict Sales"):
    prediction = model.predict(input_df)
    st.success(f"📊 Predicted Sales: ₹{prediction[0]:.2f}")
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and column names
model = joblib.load('trained_model.pkl')
X_columns = joblib.load('X_columns.pkl')

# Title
st.title("📦 Predictive Demand Analysis")
st.subheader("Enter Order Details")

# --- Input Fields (with unique keys) ---
quantity = st.number_input("Quantity", min_value=1, value=2, key="quantity_input")
discount = st.number_input("Discount", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="discount_input")
profit = st.number_input("Profit", value=50.0, key="profit_input")
category = st.selectbox("Category", ['Furniture', 'Office Supplies', 'Technology'], key="category_select")

# --- Input DataFrame ---
input_df = pd.DataFrame([{
    'Quantity': quantity,
    'Discount': discount,
    'Profit': profit,
    'Category': category
}])

# --- Encode Categorical Columns ---
for col in input_df.select_dtypes(include='object').columns:
    input_df[col] = input_df[col].astype('category').cat.codes

# --- Align input with training columns ---
input_df = input_df.reindex(columns=X_columns, fill_value=0)

# --- Prediction ---
if st.button("Predict Sales", key="predict_button"):
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Sales: ₹{prediction[0]:,.2f}")

    # --- Chart 1: Dummy Sales Trend ---
    st.subheader("📈 Example Sales Trend (Dummy Data)")
    sales_trend = pd.DataFrame({
        'Day': list(range(1, 11)),
        'Sales': np.random.normal(loc=prediction[0], scale=20, size=10)
    })
    st.line_chart(sales_trend.set_index('Day'))

    # --- Chart 2: Feature Importance ---
    if hasattr(model, "feature_importances_"):
        st.subheader("🌟 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X_columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))
