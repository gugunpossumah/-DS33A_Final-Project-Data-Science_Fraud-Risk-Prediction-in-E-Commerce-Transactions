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
    scaler = preprocessing['scaler']
    label_encoders = preprocessing['label_encoders']
except:
    st.error("Model tidak ditemukan. Pastikan file model ada di direktori yang sama.")
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
    
    features = pd.DataFrame(data, index=[0])
    return features

#Get user input
input_df = user_input_features()

#buat fungsi preprocessing
def preprocess_input(data, scaler, label_encoders):
    #Copy data
    data_processed = data.copy()
    
    #Pastikan urutan kolom sesuai dengan yang diharapkan scaler
    numerical_cols = ['Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days', 'Transaction Hour'] #
    categorical_cols = ['Payment Method', 'Product Category', 'Device Used']
    
    # Label encoding untuk categorical features
    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen labels
            if data_processed[col].iloc[0] in le.classes_:
                data_processed[col] = le.transform([data_processed[col].iloc[0]])[0]
            else:
                # Jika label tidak dikenal, gunakan nilai default (0)
                data_processed[col] = 0
        else:
            data_processed[col] = 0
    
    # Pastikan hanya kolom numerik yang di-scale
    numerical_data = data_processed[numerical_cols]
    
    # Scale numerical features
    scaled_numerical = scaler.transform(numerical_data)
    
    # Gabungkan kembali dengan categorical features
    data_processed[numerical_cols] = scaled_numerical
    
    # Urutkan kolom sesuai dengan yang diharapkan model
    expected_columns = numerical_cols + categorical_cols
    data_processed = data_processed[expected_columns]
    
    return data_processed


#buat Main panel
st.subheader("Data Transaksi yang Dimasukkan")
st.write(input_df)

#buat tombol Prediksi ketika ditekan
if st.button("Predict Fraud Risk"):
    # Preprocess input
    processed_input = preprocess_input(input_df, scaler, label_encoders)
    
    # Predict
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)
    
    # Display results
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Prediction", "FRAUD" if prediction[0] == 1 else "LEGITIMATE")
    
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
- **False Positive Rate**: 3.1%
- **Estimated Savings**: Rp 2.1M per 1000 transactions
""")


#buat Footer untuk peringatan
st.markdown("---")
st.markdown("""
**Disclaimer**: Prediksi ini berdasarkan model machine learning dan harus digunakan sebagai alat bantu keputusan, bukan sebagai satu-satunya sumber kebenaran.
""")
