import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import smtplib
import bcrypt
from email.message import EmailMessage
from datetime import datetime
from fpdf import FPDF
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import plotly.express as px
import random
import requests

# 📁 Paths
DATA_FOLDER = "user_data"
USER_DATA_FILE = os.path.join(DATA_FOLDER, "users.json")
PREDICTION_HISTORY_FILE = os.path.join(DATA_FOLDER, "prediction_history.json")
OTP_STORAGE = {}

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER, exist_ok=True)

if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

@st.cache_resource
def load_models():
    return (
        joblib.load("trained_model.pkl"),
        joblib.load("demand_model.pkl"),
        joblib.load("X_columns.pkl"),
        joblib.load("X_demand_columns.pkl"),
        joblib.load("sales_mae.pkl"),
        joblib.load("demand_mae.pkl")
    )

@st.cache_resource
def load_encoders():
    encoders = {}
    if os.path.exists("encoders"):
        for fname in os.listdir("encoders"):
            if fname.endswith(".pkl"):
                col = fname.replace("le_", "").replace(".pkl", "")
                encoders[col] = joblib.load(os.path.join("encoders", fname))
    return encoders

def send_email_otp(email, otp):
    try:
        msg = EmailMessage()
        msg.set_content(f"Your OTP is: {otp}")
        msg['Subject'] = "Your OTP Verification"
        msg['From'] = "your_email@gmail.com"
        msg['To'] = email

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login("your_email@gmail.com", "your_app_password")
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(e)
        return False

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

# 🔐 Auth Utilities
def save_user(username, password, phone=None, email=None):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[username] = {"password": hashed, "phone": phone, "email": email, "address": ""}
    write_json(USER_DATA_FILE, users)

def authenticate(username, password):
    if username in users:
        return bcrypt.checkpw(password.encode(), users[username]['password'].encode())
    return False

def generate_otp():
    return str(random.randint(100000, 999999))

def read_json(path, default):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return default

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

# 🔁 Load data
users = read_json(USER_DATA_FILE, {})
prediction_history = read_json(PREDICTION_HISTORY_FILE, {})
sales_model, demand_model, X_cols, X_demand_cols, sales_mae, demand_mae = load_models()
encoders = load_encoders()

st.set_page_config(layout="centered", page_title="Sales & Demand App")

# Sidebar
with st.sidebar:
    page = option_menu("📜 Menu", ["Login", "Predict", "History", "Settings"],
                       icons=["person", "bar-chart", "clock-history", "gear"])

if "user" not in st.session_state:
    st.session_state.user = None

# ---------------- LOGIN PAGE ----------------
# ---------------- LOGIN PAGE ----------------
if page == "Login":
    st.markdown("## 👋 Welcome to the Sales & Demand Portal")
    st.write("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # ✅ Show Lottie Animation
        lottie_login = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_49rdyysj.json")
        if lottie_login:
            st_lottie(lottie_login, height=200)

        if not st.session_state.show_signup:
            st.subheader("🔐 Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Log In"):
                if authenticate(username, password):
                    st.session_state.user = username
                    st.success("✅ Login successful!")
                    st.query_params.update(logged_in="true")
                    st.stop()
                else:
                    st.error("❌ Invalid username or password")
            if st.button("Switch to Sign Up", type="secondary"):
                st.session_state.show_signup = True
                st.rerun()
        else:
            st.subheader("📝 Sign Up")
            new_username = st.text_input("Create Username")
            new_password = st.text_input("Create Password", type="password")
            if st.button("Create Account"):
                if new_username in users:
                    st.warning("⚠ This username is already taken.")
                else:
                    save_user(new_username, new_password)
                    st.success("🎉 Account created! Please log in.")
                    st.session_state.show_signup = False
                    st.rerun()
            if st.button("Back to Login"):
                st.session_state.show_signup = False
                st.rerun()


# ---------------- PREDICT PAGE ----------------
elif page == "Predict":
    if not st.session_state.user:
        st.warning("Please login to access predictions.")
    else:
        st.title("📈 Sales & Demand Prediction")
        st.write("---")
        with st.form("predict_form"):
            subcat = st.selectbox("Sub-Category", options=encoders['Sub-Category'].classes_.tolist())
            prod_name = st.selectbox("Product Name", options=encoders['Product Name'].classes_.tolist())
            quantity = st.number_input("Quantity", min_value=1, max_value=100, value=1)
            order_date = st.date_input("Order Date")
            submitted = st.form_submit_button("Predict")

        if submitted:
            df_input = pd.DataFrame({
                'Sub-Category': [subcat],
                'Product Name': [prod_name],
                'Quantity': [quantity],
                'Order Date': [order_date.strftime('%Y-%m-%d')]
            })

            for col in df_input.columns:
                if col in encoders:
                    df_input[col] = encoders[col].transform(df_input[col])

            X_sales = df_input.reindex(columns=X_cols, fill_value=0)
            X_demand = df_input.reindex(columns=X_demand_cols, fill_value=0)

            sales_pred = np.expm1(sales_model.predict(X_sales)[0])
            demand_pred = np.expm1(demand_model.predict(X_demand)[0])

            st.success(f"✅ Predicted Sales: ₹{sales_pred:.2f} ± ₹{sales_mae:.2f}")
            st.success(f"✅ Predicted Demand: {demand_pred:.0f} units ± {demand_mae:.0f}")

            pred_entry = {
                "Username": st.session_state.user,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Sub-Category": subcat,
                "Product Name": prod_name,
                "Quantity": quantity,
                "Order Date": order_date.strftime("%Y-%m-%d"),
                "Predicted Sales": round(sales_pred, 2),
                "Predicted Demand": round(demand_pred, 0)
            }
            prediction_history.setdefault(st.session_state.user, []).append(pred_entry)
            write_json(PREDICTION_HISTORY_FILE, prediction_history)

            fig = px.bar(
                x=["Sales", "Demand"],
                y=[sales_pred, demand_pred],
                color=["Sales", "Demand"],
                text=[f"₹{sales_pred:.2f}", f"{demand_pred:.0f} units"],
                labels={"x": "Prediction Type", "y": "Value"},
                title="📊 Predicted Sales & Demand Chart"
            )
            fig.update_traces(textposition="outside", marker_line_color='black', marker_line_width=1.5)
            fig.update_layout(
                yaxis=dict(title="Value", gridcolor="lightgrey"),
                xaxis=dict(title=""),
                title_font=dict(size=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14)
            )
            st.plotly_chart(fig, use_container_width=True)


        # 🔄 Bulk CSV Prediction
        st.markdown("### 📤 Bulk Prediction from CSV")
        csv_file = st.file_uploader("Upload CSV", type=["csv"])
        if csv_file:
            try:
                bulk_df = pd.read_csv(csv_file)
                for col in encoders:
                    if col in bulk_df:
                        bulk_df[col] = encoders[col].transform(bulk_df[col].astype(str))
                bulk_df["Order Date"] = pd.to_datetime(bulk_df["Order Date"]).dt.strftime('%Y-%m-%d')

                X_sales_bulk = bulk_df.reindex(columns=X_cols, fill_value=0)
                X_demand_bulk = bulk_df.reindex(columns=X_demand_cols, fill_value=0)

                sales_preds = np.expm1(sales_model.predict(X_sales_bulk))
                demand_preds = np.expm1(demand_model.predict(X_demand_bulk))

                bulk_df["Predicted Sales"] = np.round(sales_preds, 2)
                bulk_df["Predicted Demand"] = np.round(demand_preds, 0)

                st.success("✅ Bulk predictions completed!")
                st.dataframe(bulk_df)

                csv = bulk_df.to_csv(index=False).encode()
                st.download_button("📥 Download Results as CSV", csv, file_name="bulk_predictions.csv")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ---------------- HISTORY PAGE ----------------
elif page == "History":
    st.title("🕓 Prediction History")
    if not st.session_state.user:
        st.warning("Please login to view history.")
    else:
        user_history = prediction_history.get(st.session_state.user, [])
        if user_history:
            df = pd.DataFrame(user_history)[["Timestamp", "Sub-Category", "Product Name", "Quantity", "Predicted Sales", "Predicted Demand"]]
            st.write("### Your History")
            st.dataframe(df, use_container_width=True, height=400)

            # 🔍 Trend Chart
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df_sorted = df.sort_values("Timestamp")
            fig = px.line(df_sorted, x="Timestamp", y=["Predicted Sales", "Predicted Demand"], title="📈 Sales & Demand Trend")
            st.plotly_chart(fig, use_container_width=True)

            selected_indices = st.multiselect("Select entries", options=df.index.tolist(),
                                              format_func=lambda i: f"{df.iloc[i]['Timestamp']} | {df.iloc[i]['Product Name']}")
            if selected_indices and st.button("Delete Selected"):
                prediction_history[st.session_state.user] = [entry for idx, entry in enumerate(user_history) if idx not in selected_indices]
                write_json(PREDICTION_HISTORY_FILE, prediction_history)
                st.success("Deleted selected entries.")
                st.rerun()

            st.download_button("Download as CSV", df.to_csv(index=False), file_name="prediction_history.csv")
            if st.button("Delete All History"):
                prediction_history[st.session_state.user] = []
                write_json(PREDICTION_HISTORY_FILE, prediction_history)
                st.success("All history cleared.")
                st.rerun()
        else:
            st.info("No predictions found yet.")

# ---------------- SETTINGS PAGE ----------------
elif page == "Settings":
    st.title("⚙ Settings")
    if st.session_state.user:
        st.markdown(f"*Logged in as:* {st.session_state.user}")
        user_data = users.get(st.session_state.user, {})

        st.subheader("📄 My Profile")
        email = st.text_input("Email", value=user_data.get("email", ""), placeholder="Enter email")
        phone = st.text_input("Phone", value=user_data.get("phone", ""), placeholder="Enter phone")
        address = st.text_area("Address", value=user_data.get("address", ""), placeholder="Enter address")

        if st.button("Save Changes"):
            users[st.session_state.user]["email"] = email
            users[st.session_state.user]["phone"] = phone
            users[st.session_state.user]["address"] = address
            write_json(USER_DATA_FILE, users)
            st.success("✅ Profile updated.")

        st.subheader("🔐 Change Password via OTP")
        new_pass = st.text_input("New Password", type="password")
        otp_input = st.text_input("Enter OTP", max_chars=6)

        if st.button("Send OTP to Email"):
            if email:
                otp_code = generate_otp()
                OTP_STORAGE[st.session_state.user] = otp_code
                if send_email_otp(email, otp_code):
                    st.success("✅ OTP sent to email.")
                else:
                    st.error("❌ Failed to send OTP.")
            else:
                st.warning("Enter email to send OTP.")

        if st.button("Update Password"):
            if OTP_STORAGE.get(st.session_state.user) == otp_input:
                hashed = bcrypt.hashpw(new_pass.encode(), bcrypt.gensalt()).decode()
                users[st.session_state.user]["password"] = hashed
                write_json(USER_DATA_FILE, users)
                OTP_STORAGE.pop(st.session_state.user, None)
                st.success("✅ Password changed.")
            else:
                st.error("❌ Invalid OTP.")

        if st.button("🗑 Delete My Account"):
            confirm = st.radio("Are you sure?", ["No", "Yes, delete my account"])
            if confirm == "Yes, delete my account":
                users.pop(st.session_state.user, None)
                prediction_history.pop(st.session_state.user, None)
                write_json(USER_DATA_FILE, users)
                write_json(PREDICTION_HISTORY_FILE, prediction_history)
                st.success("✅ Account deleted.")
                st.session_state.user = None
                st.rerun()

        if st.button("🚪 Logout"):
            st.session_state.user = None
            st.success("Logged out.")
            st.rerun()
    else:
        st.info("Please log in to access settings.")
