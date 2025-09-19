import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

#kita set page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="⚠️",
    layout="wide"
)

#kita load model dan preprocessing objects
@st.cache_resource
def load_model():
    model = joblib.load('fraud_detection_model.pkl')
    preprocessing = joblib.load('preprocessing_objects.joblib')
    return model, preprocessing

try:
    model, preprocessing = load_model()
    st.write("Isi preprocessing:", preprocessing)  # Debug
    scaler = preprocessing['scaler']
    label_encoders = preprocessing['label_encoders']
    selected_features = preprocessing.get('selected_features', None)
except:
    st.error("Model tidak ditemukan!. Pastikan file model & preprocessing ada di direktori yang sama.")
    st.stop()

#buat judul
st.title("⚠️ Fraud Detection System - E-Commerce Transactions")
st.markdown("""
Sistem ini memprediksi kemungkinan transaksi e-commerce merupakan fraud berdasarkan karakteristik transaksi.
""")

#buat Sidebar untuk input data
st.sidebar.header("Input Data Transaksi")

#buat Fungsi untuk input data
def user_input_features():
    transaction_amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
    quantity = st.sidebar.slider("Quantity", 1, 5, 2)
    customer_age = st.sidebar.slider("Customer Age", 18, 100, 35)
    account_age_days = st.sidebar.slider("Account Age Days", 1, 365, 180)
    transaction_hour = st.sidebar.slider("Transaction Hour", 0, 23, 12)
    
    payment_method = st.sidebar.selectbox("Payment Method", ['credit card', 'debit card', 'bank transfer', 'PayPal'])
    product_category = st.sidebar.selectbox("Product Category", ['electronics', 'clothing', 'home & garden', 'books', 'beauty'])
    device_used = st.sidebar.selectbox("Device Used", ['mobile', 'desktop', 'tablet'])
    
    data = {
        'Transaction Amount': transaction_amount,
        'Quantity': quantity,
        'Customer Age': customer_age,
        'Account Age Days': account_age_days,
        'Transaction Hour': transaction_hour,
        'Payment Method': payment_method,
        'Product Category': product_category,
        'Device Used': device_used
    }
    
    return pd.DataFrame(data, index=[0])
    
#Get user input   
input_df = user_input_features()

#buat fungsi preprocessing
def preprocess_input(input_df, scaler, label_encoders=None):
    data = input_df.copy()

    #kita buat Feature Engineering yang dipakai saat training
    if "Transaction Date" in data.columns:
        data["Transaction_Day"] = pd.to_datetime(data["Transaction Date"]).dt.day
        data["Transaction_Month"] = pd.to_datetime(data["Transaction Date"]).dt.month
        data["Transaction_DayOfWeek"] = pd.to_datetime(data["Transaction Date"]).dt.dayofweek
        data["Transaction_IsWeekend"] = data["Transaction_DayOfWeek"].isin([5,6]).astype(int)
        data["Transaction_IsNight"] = data["Transaction Hour"].between(0, 6).astype(int)

    if "Billing Address" in data.columns and "Shipping Address" in data.columns:
        data["Address_Mismatch"] = (data["Billing Address"] != data["Shipping Address"]).astype(int)

    if "IP Address" in data.columns:
        data["IP_FirstOctet"] = data["IP Address"].str.split(".").str[0].astype(int)
        data["IP_SecondOctet"] = data["IP Address"].str.split(".").str[1].astype(int)

    if "Transaction Amount" in data.columns and "Quantity" in data.columns:
        data["Amount_per_Item"] = data["Transaction Amount"] / data["Quantity"]
        data["Large_Transaction"] = (data["Transaction Amount"] > 500).astype(int)
        data["Transaction_Amount_Log"] = np.log1p(data["Transaction Amount"])

    # Dummy contoh agregasi customer
    if "Customer ID" in data.columns and "Transaction Amount" in data.columns:
        data["Transaction_Frequency"] = 1  # default jika 1 transaksi
        data["Avg_Amount_Customer"] = data["Transaction Amount"]
        data["Deviation_Amount"] = 0

    if "Device Used" in data.columns:
        data["Device_Change"] = 0
    data["New_Customer"] = 0

    # Scale numerical features sesuai training ---
    numerical_cols = scaler.feature_names_in_
    data[numerical_cols] = scaler.transform(data[numerical_cols])

    return data[scaler.feature_names_in_]


#buat Main panel
st.subheader("Data Transaksi yang Dimasukkan")
st.write(input_df)

#buat tombol Prediksi ketika ditekan
if st.button("Predict Fraud Risk"):
    # Preprocess input
    processed_input = preprocess_input(input_df, scaler, label_encoders, selected_features)
    
    # Predict
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)
    
    # Display results
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Prediction", "🚨 FRAUD" if prediction[0] == 1 else "✅ LEGITIMATE")
    
    with col2:
        fraud_prob = prediction_proba[0][1] * 100
        st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
    
    # Progress bar untuk probability
    st.progress(float(prediction_proba[0][1]))
    
    # Interpretasi hasil
    if prediction[0] == 1:
        st.error("🚨 WARNING: Transaksi ini terdeteksi sebagai potensial FRAUD!")
        st.info("Rekomendasi: Lakukan verifikasi tambahan pada transaksi ini.")
    else:
        st.success("✅ Transaksi ini terdeteksi sebagai LEGITIMATE")
        st.info("Transaksi dapat diproses secara normal.")
    
    # Additional information
    st.subheader("Detail Probabilitas")
    prob_df = pd.DataFrame({
        'Class': ['Legitimate', 'Fraud'],
        'Probability': [prediction_proba[0][0] * 100, prediction_proba[0][1] * 100]
    })
    st.bar_chart(prob_df.set_index('Class'))

#kita tambahkan informasi tentang model
st.sidebar.markdown("---")
st.sidebar.subheader("About the Model")
st.sidebar.info("""
**Model Machine Learning yang digunakan:**
- **Algorithm**: XGBoost Classifier (Optimal)
- **Best Threshold**: 0.7726
- **AUC Score**: 0.7735
- **F1-Score**: 0.3604

**Performance Metrics (Optimal Threshold):**
- **Precision**: 30.42%
- **Recall**: 44.19%
- **Accuracy**: 92%

**Business Impact:**
- **Fraud Detection Rate**: 44%
- **False Positive Rate**: 30%
- **Estimated Savings**: Rp 2.1M per 1000 transactions
""")


#buat Footer untuk peringatan
st.markdown("---")
st.markdown("""
**Disclaimer**: Prediksi ini berdasarkan model machine learning dan harus digunakan sebagai alat bantu keputusan.
""")
