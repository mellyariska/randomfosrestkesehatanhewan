import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ===============================
# CEK FILE MODEL
# ===============================
MODEL_PATH = "model_rf.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ File model_rf.pkl tidak ditemukan. Pastikan file ada di folder yang sama dengan app.py")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("❌ File scaler.pkl tidak ditemukan. Pastikan file ada di folder yang sama dengan app.py")
    st.stop()

# ===============================
# LOAD MODEL & SCALER
# ===============================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===============================
# SETUP DASHBOARD
# ===============================
st.set_page_config(
    page_title="Dashboard Prediksi Kesehatan Hewan",
    layout="wide"
)

st.title("🐁 Dashboard Prediksi Kesehatan Hewan Percobaan")
st.markdown(
    """
    Dashboard ini mengimplementasikan **Machine Learning Random Forest**
    untuk memprediksi kondisi kesehatan hewan percobaan berbasis data lingkungan
    dan fisiologis.
    """
)

# ===============================
# UPLOAD DATA
# ===============================
uploaded_file = st.file_uploader(
    "📤 Upload Data Excel",
    type=["xlsx"]
)

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    st.subheader("📊 Data Input")
    st.dataframe(data.head())

    # ===============================
    # PREPROCESSING
    # ===============================
    X = data.drop(columns=["Label Kesehatan"], errors="ignore")
    X_scaled = scaler.transform(X)

    # ===============================
    # PREDIKSI
    # ===============================
    data["Prediksi Kesehatan"] = model.predict(X_scaled)

    st.subheader("🧠 Hasil Prediksi")
    st.dataframe(data)

    # ===============================
    # VISUALISASI
    # ===============================
    st.subheader("📈 Distribusi Hasil Prediksi")
    counts = data["Prediksi Kesehatan"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Kategori Kesehatan")
    ax.set_ylabel("Jumlah")
    ax.set_title("Distribusi Prediksi Kesehatan Hewan")
    st.pyplot(fig)

    # ===============================
    # DOWNLOAD
    # ===============================
    st.download_button(
        label="⬇️ Download Hasil Prediksi",
        data=data.to_csv(index=False),
        file_name="hasil_prediksi_kesehatan.csv",
        mime="text/csv"
    )
else:
    st.info("📌 Silakan upload file Excel untuk memulai prediksi.")
