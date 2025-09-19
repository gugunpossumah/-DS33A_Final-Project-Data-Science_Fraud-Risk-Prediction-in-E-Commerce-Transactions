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
    selected_features = preprocessing['selected_features'] #20 fitur hasil SelectKBest

    st.success("Model & preprocessing berhasil dimuat!")
    st.write("Jumlah fitur yang dipakai:", len(selected_features))
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
    
    return pd.DataFrame({
        'Transaction Amount': [transaction_amount],
        'Quantity': [quantity],
        'Customer Age': [customer_age],
        'Account Age Days': [account_age_days],
        'Transaction Hour': [transaction_hour],
        'Payment Method': [payment_method],
        'Product Category': [product_category],
        'Device Used': [device_used]
    })
    
#Get user input   
input_df = user_input_features()

#buat fungsi preprocessing
def preprocess_input(input_df, scaler, label_encoders, selected_features):
    # Buat dataframe kosong sesuai selected_features
    df = pd.DataFrame(0, index=[0], columns=selected_features)

    # Pisahkan kolom kategorikal dan numerik
    numeric_cols = [c for c in selected_features if c not in label_encoders.keys()]

    # Masukkan nilai numerik dari input user
    for col in numeric_cols:
        if col in input_df.columns:
            df[col] = input_df[col]
        else:
            df[col] = 0.0  # default

    # Encode kolom kategorikal
    for col in label_encoders.keys():
        le = label_encoders[col]
        if col in input_df.columns:
            val = input_df[col].iloc[0]
            df[col] = le.transform([val])[0] if val in le.classes_ else -1
        else:
            df[col] = -1  # default jika user tidak input

    # Scale numerik
    if numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Pastikan urutan kolom sesuai training
    df_final = df[selected_features]

    return df_final

#buat Main panel
st.subheader("Data Transaksi yang Dimasukkan")
st.write(input_df)

#buat tombol Prediksi ketika ditekan
if st.button("Predict Fraud Risk"):
    try:
        processed_input = preprocess_input(input_df, scaler, label_encoders, selected_features)
        st.write("✅ Final Processed Features:", processed_input.columns.tolist())
        st.write("Processed shape:", processed_input.shape)

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
    except Exception as e:
        st.error(f"❌ Error saat preprocessing/predict: {e}")
        
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
