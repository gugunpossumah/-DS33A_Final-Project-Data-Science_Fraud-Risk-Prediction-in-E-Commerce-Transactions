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
def preprocess_input(df, scaler, label_encoders, selected_features):
    df_processed = df.copy()

    # --- Feature engineering ---
    df_processed['Transaction_Day'] = df_processed['Transaction Date'].dt.day
    df_processed['Transaction_Month'] = df_processed['Transaction Date'].dt.month
    df_processed['Transaction_DayOfWeek'] = df_processed['Transaction Date'].dt.dayofweek
    df_processed['Transaction_IsWeekend'] = df_processed['Transaction_DayOfWeek'].isin([5,6]).astype(int)
    df_processed['Transaction Hour'] = df_processed['Transaction Date'].dt.hour
    df_processed['Transaction_IsNight'] = ((df_processed['Transaction Hour'] >=0) & (df_processed['Transaction Hour'] <=6)).astype(int)
    df_processed['Address_Mismatch'] = (df_processed['Shipping Address'] != df_processed['Billing Address']).astype(int)

    # IP Address
    def extract_ip(ip):
        try:
            parts = str(ip).split('.')
            if len(parts)==4:
                return int(parts[0]), int(parts[1])
        except:
            pass
        return 0,0
    ip_feats = df_processed['IP Address'].apply(extract_ip)
    df_processed['IP_FirstOctet'] = ip_feats.apply(lambda x:x[0])
    df_processed['IP_SecondOctet'] = ip_feats.apply(lambda x:x[1])

    # Transaction based
    df_processed['Amount_per_Item'] = df_processed['Transaction Amount'] / (df_processed['Quantity'] + 1e-6)
    df_processed['Large_Transaction'] = (df_processed['Transaction Amount'] > 500).astype(int)
    df_processed['Transaction_Amount_Log'] = np.log1p(df_processed['Transaction Amount'])

    # Behavioral features: hanya dummy yang ada di selected_features
    for col in ['Transaction_Frequency', 'Avg_Amount_Customer', 'Deviation_Amount', 'Device_Change', 'New_Customer']:
        if col in selected_features:
            if col == 'New_Customer':
                df_processed[col] = (df_processed['Account Age Days'] < 30).astype(int)
            else:
                df_processed[col] = 0

    # Encode categorical hanya yang ada di selected_features
    categorical_cols = ['Payment Method', 'Product Category', 'Device Used', 'Customer Location']
    for col in categorical_cols:
        if col in selected_features and col in df_processed.columns and col in label_encoders:
            le = label_encoders[col]
            val = df_processed[col].iloc[0]
            df_processed[col] = le.transform([val])[0] if val in le.classes_ else -1

    # Drop kolom yang tidak dipakai
    drop_cols = ['Transaction Date', 'Shipping Address', 'Billing Address', 'IP Address', 'Transaction Hour']
    df_processed = df_processed.drop(columns=[col for col in drop_cols if col in df_processed.columns], errors='ignore')

    # Tambahkan kolom hilang yang ada di selected_features
    for col in selected_features:
        if col not in df_processed.columns:
            df_processed[col] = 0

    # Scale numerik
    num_cols = [col for col in scaler.feature_names_in_ if col in df_processed.columns]
    if num_cols:
        df_processed[num_cols] = scaler.transform(df_processed[num_cols])

    # Urutkan sesuai selected_features
    df_final = df_processed[selected_features]

    return df_final

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
