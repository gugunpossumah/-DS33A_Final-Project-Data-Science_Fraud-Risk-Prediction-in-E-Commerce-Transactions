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
    st.write("Daftar fitur lengkap:", selected_features.tolist())
    
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error("Model tidak ditemukan!. Pastikan file model & preprocessing ada di direktori yang sama.")
    st.stop()

#buat judul
st.title("⚠️ Fraud Detection System - E-Commerce Transactions")
st.markdown("""
Sistem ini memprediksi kemungkinan transaksi e-commerce merupakan fraud berdasarkan karakteristik transaksi.
""")

# Tampilkan semua fitur yang diperlukan
st.sidebar.header("Daftar Fitur yang Diperlukan")
st.sidebar.write("Total fitur:", len(selected_features))
st.sidebar.write("Fitur kategorikal:", [col for col in selected_features if col in label_encoders])
st.sidebar.write("Fitur numerik:", [col for col in selected_features if col not in label_encoders])

#buat Sidebar untuk input data
st.sidebar.header("Input Data Transaksi")

# Identifikasi semua fitur yang diperlukan
categorical_features = [col for col in selected_features if col in label_encoders]
numerical_features = [col for col in selected_features if col not in label_encoders]

#buat Fungsi untuk input data yang lengkap
def user_input_features():
    input_data = {}
    
    # Buat input untuk setiap fitur kategorikal
    for feature in categorical_features:
        if feature in label_encoders:
            classes = label_encoders[feature].classes_
            input_data[feature] = st.sidebar.selectbox(f"{feature}", classes, index=0)
    
    # Buat input untuk setiap fitur numerik
    for feature in numerical_features:
        # Tentukan range yang sesuai berdasarkan jenis fitur
        if 'Amount' in feature:
            input_data[feature] = st.sidebar.number_input(f"{feature}", min_value=0.0, value=100.0)
        elif 'Age' in feature or 'Days' in feature:
            input_data[feature] = st.sidebar.number_input(f"{feature}", min_value=0, value=30)
        elif 'Hour' in feature:
            input_data[feature] = st.sidebar.slider(f"{feature}", 0, 23, 12)
        elif 'Quantity' in feature:
            input_data[feature] = st.sidebar.slider(f"{feature}", 1, 10, 1)
        else:
            # Default untuk fitur numerik lainnya
            input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0)
    
    return pd.DataFrame([input_data])

#Get user input   
input_df = user_input_features()

#buat fungsi preprocessing yang lebih robust
def preprocess_input(input_df, scaler, label_encoders, selected_features):
    try:
        # Buat dataframe dengan semua fitur yang diperlukan
        df_processed = pd.DataFrame(index=input_df.index, columns=selected_features)
        
        # Isi nilai untuk setiap fitur
        for col in selected_features:
            if col in input_df.columns:
                df_processed[col] = input_df[col].values
            else:
                # Beri nilai default berdasarkan tipe fitur
                if col in label_encoders:
                    df_processed[col] = label_encoders[col].classes_[0]  # kategori pertama
                else:
                    df_processed[col] = 0.0  # default numerik
        
        # Encode kolom kategorikal
        for col in selected_features:
            if col in label_encoders:
                le = label_encoders[col]
                # Pastikan nilai ada dalam classes
                if df_processed[col].iloc[0] in le.classes_:
                    df_processed[col] = le.transform(df_processed[col])
                else:
                    # Jika nilai tidak dikenal, gunakan nilai default (kelas pertama)
                    df_processed[col] = le.transform([le.classes_[0]])[0]
        
        # Scale kolom numerik
        numerical_cols = [col for col in selected_features if col not in label_encoders]
        if numerical_cols and scaler is not None:
            df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])
        
        return df_processed[selected_features]
        
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        raise e

#buat Main panel
st.subheader("Data Transaksi yang Dimasukkan")
st.write(input_df)

# Tampilkan informasi tentang fitur
st.subheader("Informasi Fitur")
col1, col2 = st.columns(2)

with col1:
    st.write("**Fitur Kategorikal:**")
    for feat in categorical_features:
        st.write(f"- {feat}: {label_encoders[feat].classes_}")

with col2:
    st.write("**Fitur Numerik:**")
    st.write(numerical_features)

#buat tombol Prediksi ketika ditekan
if st.button("Predict Fraud Risk"):
    try:
        processed_input = preprocess_input(input_df, scaler, label_encoders, selected_features)
        
        st.subheader("Data Setelah Preprocessing")
        st.dataframe(processed_input)
        st.write("Shape:", processed_input.shape)
        st.write("Data types:", processed_input.dtypes)
        
        # Pastikan urutan kolom benar
        processed_input = processed_input[selected_features]
        
        # Debug: Tampilkan beberapa nilai
        st.write("Sample values:", processed_input.iloc[0, :5].to_dict())
        
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
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

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
