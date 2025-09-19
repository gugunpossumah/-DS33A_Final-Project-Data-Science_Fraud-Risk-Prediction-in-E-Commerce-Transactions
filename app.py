import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# ---------------------------
# Set page Streamlit biar rapi
# ---------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="⚠️",
    layout="wide"
)

BASE_DIR = os.path.dirname(__file__)

# ---------------------------
# Load model dan preprocessing objects
# Kita cache biar ga load berulang-ulang
# ---------------------------
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "fraud_detection_model.pkl"))
    preprocessing = joblib.load(os.path.join(BASE_DIR, "preprocessing_objects.joblib"))
    return model, preprocessing

try:
    model, preprocessing = load_model()
    scaler = preprocessing['scaler']  # StandardScaler dari training
    label_encoders = preprocessing['label_encoders']  # LabelEncoder untuk kolom kategorikal
    selected_features = preprocessing['selected_features']  # urutan kolom pas training
    st.success("✅ Model & preprocessing berhasil dimuat!")
except Exception as e:
    st.error("❌ Model tidak ditemukan! Pastikan file model & preprocessing ada di direktori yang sama.")
    st.stop()

# ---------------------------
# Judul & deskripsi
# ---------------------------
st.title("⚠️ Fraud Detection System E-Commerce Transactions")
st.markdown("""
Sistem ini memprediksi kemungkinan transaksi e-commerce merupakan fraud berdasarkan karakteristik transaksi.
""")

# ---------------------------
# Sidebar: Input Data
# ---------------------------
st.sidebar.header("Input Data Transaksi")

def user_input_features():
    # kita ambil input user dari sidebar
    transaction_amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
    quantity = st.sidebar.slider("Quantity", 1, 5, 2)
    customer_age = st.sidebar.slider("Customer Age", 18, 100, 35)
    account_age_days = st.sidebar.slider("Account Age Days", 1, 365, 180)
    transaction_hour = st.sidebar.slider("Transaction Hour", 0, 23, 12)
    
    payment_method = st.sidebar.selectbox("Payment Method", ['credit card', 'debit card', 'bank transfer', 'PayPal'])
    product_category = st.sidebar.selectbox("Product Category", ['electronics', 'clothing', 'home & garden', 'books', 'beauty'])
    device_used = st.sidebar.selectbox("Device Used", ['mobile', 'desktop', 'tablet'])
    
    # bikin DataFrame dari input user
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

input_df = user_input_features()  # ambil input user

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_input(input_df, scaler, label_encoders, selected_features):
    # bikin DataFrame kosong dulu sesuai selected_features biar aman
    df = pd.DataFrame(0, index=[0], columns=selected_features)
    
    # pisahin kolom numerik vs kategorikal
    numeric_cols = [c for c in selected_features if c not in label_encoders.keys()]
    
    # isi numerik dari input user, kalau nggak ada isi 0.0
    for col in numeric_cols:
        if col in input_df.columns:
            df[col] = input_df[col]
        else:
            df[col] = 0.0  # default aman
    
    # encode kategorikal sesuai LabelEncoder yang dipakai waktu training
    for col in label_encoders.keys():
        le = label_encoders[col]
        if col in input_df.columns:
            val = input_df[col].iloc[0]
            df[col] = le.transform([val])[0] if val in le.classes_ else -1  # -1 kalo value baru
        else:
            df[col] = -1  # default aman
    
    # scale numerik biar sama dengan training
    if numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # pastiin kolom tetap urut sama kaya pas training
    df_final = df[selected_features]
    return df_final

# ---------------------------
# Main panel: tampil input user
# ---------------------------
st.subheader("Data Transaksi yang Dimasukkan")
st.write(input_df)

if st.button("Predict Fraud Risk"):
    try:
        # preprocessing input user
        processed_input = preprocess_input(input_df, scaler, label_encoders, selected_features)
        st.write("✅ Final Processed Features:", processed_input.columns.tolist())
        st.write("Processed shape:", processed_input.shape)
        
        # prediksi
        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)
        
        # tampil hasil
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", "🚨 FRAUD" if prediction[0]==1 else "✅ LEGITIMATE")
        with col2:
            fraud_prob = prediction_proba[0][1]*100
            st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
        
        # progress bar simple biar visual
        st.progress(float(prediction_proba[0][1]))
        
        # interpretasi hasil
        if prediction[0]==1:
            st.error("🚨 Transaksi ini terdeteksi sebagai potensial FRAUD!")
            st.info("Lakukan verifikasi tambahan.")
        else:
            st.success("✅ Transaksi ini terdeteksi sebagai LEGITIMATE")
            st.info("Transaksi dapat diproses normal.")
        
        # chart probabilitas
        st.subheader("Detail Probabilitas")
        prob_df = pd.DataFrame({
            'Class': ['Legitimate','Fraud'],
            'Probability':[prediction_proba[0][0]*100, prediction_proba[0][1]*100]
        })
        st.bar_chart(prob_df.set_index('Class'))
        
    except Exception as e:
        st.error(f"❌ Error saat preprocessing/predict: {e}")
        st.stop()

# ---------------------------
# Sidebar: info model
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("About the Model")
st.sidebar.info("""
**Algorithm**: XGBoost Classifier (Optimal)
**Best Threshold**: 0.7726
**AUC Score**: 0.7735
**F1-Score**: 0.3604
**Precision**: 30.42%
**Recall**: 44.19%
**Accuracy**: 92%
**Business Impact**: Estimated savings Rp 2.1M per 1000 transactions
""")

# ---------------------------
# Footer: disclaimer
# ---------------------------
st.markdown("---")
st.markdown("**Disclaimer**: Prediksi ini hanya alat bantu keputusan. Gunakan sebagai referensi, bukan keputusan final.")
