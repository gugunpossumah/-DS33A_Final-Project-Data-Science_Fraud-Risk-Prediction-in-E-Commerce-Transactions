import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

#kita set page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="⚠️",
    layout="wide"
)

#kita load model dan preprocessing objects
BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "fraud_detection_model.pkl"))
    preprocessing = joblib.load(os.path.join(BASE_DIR, "preprocessing_objects.joblib"))
    return model, preprocessing

try:
    model, preprocessing = load_model()

    scaler = preprocessing['scaler']
    label_encoders = preprocessing['label_encoders']

    # Override selected_features dengan fitur yang dipakai model
    if hasattr(model, "feature_names_in_"):
        selected_features = list(model.feature_names_in_)
    else:
        selected_features = preprocessing['selected_features']

    st.success("Model & preprocessing berhasil dimuat!")
    st.write("Jumlah fitur yang dipakai:", len(selected_features))
    st.write("Daftar fitur final:", selected_features)

except Exception as e:
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
def preprocess_input(input_df, scaler, label_encoders, selected_features):
    df_processed = input_df.copy()

    # Feature Engineering hanya yang dipakai
    if "Transaction_Day" in selected_features:
        df_processed["Transaction_Day"] = 15
    if "Transaction_DayOfWeek" in selected_features:
        df_processed["Transaction_DayOfWeek"] = 3
    if "Transaction_IsNight" in selected_features:
        df_processed["Transaction_IsNight"] = (
            df_processed["Transaction Hour"].between(0, 6).astype(int)
        )
    if "Transaction_IsWeekend" in selected_features:
        df_processed["Transaction_IsWeekend"] = 0
    if "Transaction_Month" in selected_features:
        df_processed["Transaction_Month"] = 1
    if "Address_Mismatch" in selected_features:
        df_processed["Address_Mismatch"] = 0
    if "IP_FirstOctet" in selected_features:
        df_processed["IP_FirstOctet"] = 0
    if "IP_SecondOctet" in selected_features:
        df_processed["IP_SecondOctet"] = 0
    if "Amount_per_Item" in selected_features:
        df_processed["Amount_per_Item"] = (
            df_processed["Transaction Amount"] / (df_processed["Quantity"] + 1e-6)
        )
    if "Large_Transaction" in selected_features:
        df_processed["Large_Transaction"] = (df_processed["Transaction Amount"] > 500).astype(int)
    if "Transaction_Amount_Log" in selected_features:
        df_processed["Transaction_Amount_Log"] = np.log1p(df_processed["Transaction Amount"])

    # Encode categorical
    categorical_cols = ["Payment Method", "Product Category", "Device Used", "Customer Location"]
    for col in categorical_cols:
        if col in selected_features:
            if col in df_processed.columns and col in label_encoders:
                le = label_encoders[col]
                val = df_processed[col].iloc[0]
                if val in le.classes_:
                    df_processed[col] = le.transform([val])[0]
                else:
                    df_processed[col] = -1
            else:
                df_processed[col] = -1

    # Lengkapi missing features
    for col in selected_features:
        if col not in df_processed.columns:
            df_processed[col] = 0

    # Scaling numeric sesuai training
    numerical_cols = [col for col in scaler.feature_names_in_ if col in selected_features]
    if numerical_cols:
        df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])

    # Pastikan urutan fitur sesuai training
    df_processed = df_processed[selected_features]

    return df_processed
    
#buat Main panel
st.subheader("Data Transaksi yang Dimasukkan")
st.write(input_df)

#buat tombol Prediksi ketika ditekan
if st.button("Predict Fraud Risk"):
    try:
        # Preprocess input
        processed_input = preprocess_input(
            input_df, scaler, label_encoders, selected_features, model
        )

        # Debug check
        st.write("✅ Final Processed Features:", processed_input.columns.tolist())
        st.write("Processed shape:", processed_input.shape)

        # Bandingkan dengan selected_features
        st.write("--DEBUG FEATURE CHECK--")
        st.write("Expected (selected_features):", selected_features)
        st.write("Got from preprocessing:", processed_input.columns.tolist())

        # Predict
        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)

    except Exception as e:
        st.error(f"❌ Error saat preprocessing/predict: {e}")
        st.stop()
        
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
