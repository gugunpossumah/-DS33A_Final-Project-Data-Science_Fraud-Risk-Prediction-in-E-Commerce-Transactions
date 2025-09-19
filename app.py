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
    # Input untuk fitur-fitur yang disebutkan dalam error
    input_data = {}
    
    # Fitur numerik
    input_data['Transaction Amount'] = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
    input_data['Quantity'] = st.sidebar.slider("Quantity", 1, 5, 2)
    input_data['Customer Age'] = st.sidebar.slider("Customer Age", 18, 100, 35)
    input_data['Account Age Days'] = st.sidebar.slider("Account Age Days", 1, 365, 180)
    input_data['Transaction Hour'] = st.sidebar.slider("Transaction Hour", 0, 23, 12)
    
    # Fitur kategorikal dari error message
    input_data['Payment Method'] = st.sidebar.selectbox("Payment Method", ['credit card', 'debit card', 'bank transfer', 'PayPal'])
    input_data['Product Category'] = st.sidebar.selectbox("Product Category", ['electronics', 'clothing', 'home & garden', 'books', 'beauty'])
    input_data['Device Used'] = st.sidebar.selectbox("Device Used", ['mobile', 'desktop', 'tablet'])
    
    # Tambahkan fitur-fitur yang missing dari error
    # Anda perlu menyesuaikan nilai default ini berdasarkan domain knowledge
    input_data['Customer Location'] = st.sidebar.selectbox("Customer Location", 
                                                          ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Other'])
    input_data['Device_Change'] = st.sidebar.selectbox("Device Change", 
                                                      ['no', 'yes'])
    
    # Tambahkan fitur-fitur numerik lainnya dengan nilai default
    # Sesuaikan dengan fitur-fitur lain yang mungkin ada di selected_features
    for feature in selected_features:
        if feature not in input_data and feature not in ['Payment Method', 'Product Category', 
                                                        'Device Used', 'Customer Location', 'Device_Change']:
            if 'Amount' in feature or 'age' in feature.lower() or 'hour' in feature.lower():
                input_data[feature] = st.sidebar.number_input(feature, value=0.0)
            else:
                input_data[feature] = st.sidebar.number_input(feature, value=0)
    
    return pd.DataFrame([input_data])
    
#Get user input   
input_df = user_input_features()

#buat fungsi preprocessing
def preprocess_input(input_df, scaler, label_encoders, selected_features):
    try:
        # Buat dataframe dengan semua fitur yang diperlukan
        df_processed = pd.DataFrame(0, index=input_df.index, columns=selected_features)
        
        # Salin nilai dari input ke dataframe processed
        for col in selected_features:
            if col in input_df.columns:
                df_processed[col] = input_df[col].values
            else:
                # Jika fitur tidak ada di input, beri nilai default
                df_processed[col] = 0
        
        # Encode kolom kategorikal
        categorical_cols = [col for col in selected_features if col in label_encoders]
        
        for col in categorical_cols:
            if col in input_df.columns:
                le = label_encoders[col]
                # Handle unseen categories
                if input_df[col].iloc[0] in le.classes_:
                    df_processed[col] = le.transform(input_df[col])
                else:
                    df_processed[col] = -1  # nilai untuk kategori tidak dikenal
        
        # Scale kolom numerik
        numerical_cols = [col for col in selected_features if col not in categorical_cols]
        
        if numerical_cols and scaler is not None:
            df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])
        
        return df_processed[selected_features]
        
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        raise e

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
