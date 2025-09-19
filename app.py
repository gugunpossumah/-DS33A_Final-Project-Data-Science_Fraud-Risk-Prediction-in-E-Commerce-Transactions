import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="⚠️",
    layout="wide"
)

BASE_DIR = os.path.dirname(__file__)

# ------------------------
# Load model & preprocessing
# ------------------------
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "fraud_detection_model.pkl"))
    preprocessing = joblib.load(os.path.join(BASE_DIR, "preprocessing_objects.joblib"))
    return model, preprocessing

try:
    model, preprocessing = load_model()
    scaler = preprocessing['scaler']
    label_encoders = preprocessing['label_encoders']
    selected_features = preprocessing['selected_features']  # 20 fitur final

    st.success("✅ Model & preprocessing berhasil dimuat!")
except Exception as e:
    st.error("❌ Model/preprocessing tidak ditemukan.")
    st.stop()

# ------------------------
# Sidebar: user input
# ------------------------
st.sidebar.header("Input Data Transaksi")

def user_input_features():
    transaction_amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
    quantity = st.sidebar.slider("Quantity", 1, 10, 1)
    customer_age = st.sidebar.slider("Customer Age", 18, 100, 35)
    account_age_days = st.sidebar.slider("Account Age Days", 1, 365, 180)
    transaction_date = st.sidebar.date_input("Transaction Date")
    device_used = st.sidebar.selectbox("Device Used", ['mobile', 'desktop', 'tablet'])
    payment_method = st.sidebar.selectbox("Payment Method", ['credit card', 'debit card', 'bank transfer', 'PayPal'])
    product_category = st.sidebar.selectbox("Product Category", ['electronics', 'clothing', 'home & garden', 'books', 'beauty'])
    customer_location = st.sidebar.text_input("Customer Location", "Unknown")
    shipping_address = st.sidebar.text_input("Shipping Address", "Address1")
    billing_address = st.sidebar.text_input("Billing Address", "Address1")
    ip_address = st.sidebar.text_input("IP Address", "0.0.0.0")

    data = {
        'Transaction Amount': transaction_amount,
        'Quantity': quantity,
        'Customer Age': customer_age,
        'Account Age Days': account_age_days,
        'Transaction Date': pd.to_datetime(transaction_date),
        'Device Used': device_used,
        'Payment Method': payment_method,
        'Product Category': product_category,
        'Customer Location': customer_location,
        'Shipping Address': shipping_address,
        'Billing Address': billing_address,
        'IP Address': ip_address
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ------------------------
# Preprocessing untuk prediksi
# ------------------------
def preprocess_input(input_df, scaler, label_encoders, selected_features):
    df = input_df.copy()

    # Feature engineering: hanya buat kolom yang ada di selected_features
    if 'Transaction_Day' in selected_features:
        df['Transaction_Day'] = 15
    if 'Transaction_DayOfWeek' in selected_features:
        df['Transaction_DayOfWeek'] = 3
    if 'Transaction_IsNight' in selected_features:
        df['Transaction_IsNight'] = ((df['Transaction Hour'] >= 0) & (df['Transaction Hour'] <= 6)).astype(int)
    if 'Address_Mismatch' in selected_features:
        df['Address_Mismatch'] = 0
    if 'IP_FirstOctet' in selected_features:
        df['IP_FirstOctet'] = 0
    if 'IP_SecondOctet' in selected_features:
        df['IP_SecondOctet'] = 0
    if 'Amount_per_Item' in selected_features:
        df['Amount_per_Item'] = df['Transaction Amount'] / (df['Quantity'] + 1e-6)
    if 'Large_Transaction' in selected_features:
        df['Large_Transaction'] = (df['Transaction Amount'] > 500).astype(int)
    if 'Transaction_Amount_Log' in selected_features:
        df['Transaction_Amount_Log'] = np.log1p(df['Transaction Amount'])
    if 'Avg_Amount_Customer' in selected_features:
        df['Avg_Amount_Customer'] = 0
    if 'Deviation_Amount' in selected_features:
        df['Deviation_Amount'] = 0
    if 'New_Customer' in selected_features:
        df['New_Customer'] = (df['Account Age Days'] < 30).astype(int)

    # Encode categorical
    for col in ['Payment Method', 'Product Category', 'Customer Location', 'Device Used']:
        if col in selected_features:
            le = label_encoders[col]
            val = df[col].iloc[0]
            df[col] = le.transform([val])[0] if val in le.classes_ else -1

    # Scale numerik
    num_cols = [col for col in scaler.feature_names_in_ if col in df.columns and col in selected_features]
    if num_cols:
        df[num_cols] = scaler.transform(df[num_cols])

    # Tambahkan kolom yang hilang di selected_features
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0

    # Urutkan sesuai selected_features
    return df[selected_features]


# ------------------------
# Main panel
# ------------------------
st.subheader("Data Transaksi")
st.write(input_df)

if st.button("Predict Fraud Risk"):
    try:
        processed_input = preprocess_input(input_df, scaler, label_encoders, selected_features)

        st.write("✅ Final Processed Features:", processed_input.columns.tolist())
        st.write("Processed shape:", processed_input.shape)

        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)

        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        col1.metric("Prediction", "🚨 FRAUD" if prediction[0]==1 else "✅ LEGITIMATE")
        col2.metric("Fraud Probability", f"{prediction_proba[0][1]*100:.2f}%")

        st.progress(float(prediction_proba[0][1]))

    except Exception as e:
        st.error(f"❌ Error saat preprocessing/predict: {e}")
