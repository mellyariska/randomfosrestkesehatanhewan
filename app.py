import joblib

joblib.dump(model, "model_rf.pkl")
joblib.dump(scaler, "scaler.pkl")

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# Load Model & Scaler
# ===============================
model = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# UI
# ===============================
st.set_page_config(page_title="Dashboard Prediksi Kesehatan Hewan", layout="wide")

st.title("🐁 Dashboard Prediksi Kesehatan Hewan Percobaan")
st.markdown("""
Dashboard ini menggunakan **Machine Learning (Random Forest)**  
untuk memprediksi **kondisi kesehatan hewan percobaan** berdasarkan  
data fisiologis dan lingkungan.
""")

# ===============================
# Upload Data
# ===============================
uploaded_file = st.file_uploader(
    "📤 Upload data hewan (Excel)",
    type=["xlsx"]
)

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    st.subheader("📊 Data Input")
    st.dataframe(data.head())

    # ===============================
    # Preprocessing
    # ===============================
    X = data.drop(columns=["Label Kesehatan"], errors="ignore")
    X_scaled = scaler.transform(X)

    # ===============================
    # Prediksi
    # ===============================
    predictions = model.predict(X_scaled)
    data["Prediksi Kesehatan"] = predictions

    st.subheader("🧠 Hasil Prediksi")
    st.dataframe(data)

    # ===============================
    # Visualisasi
    # ===============================
    st.subheader("📈 Distribusi Prediksi Kesehatan")

    pred_counts = data["Prediksi Kesehatan"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(pred_counts.index, pred_counts.values)
    ax.set_ylabel("Jumlah Sampel")
    ax.set_xlabel("Kategori Kesehatan")
    ax.set_title("Hasil Prediksi Random Forest")

    st.pyplot(fig)

    # ===============================
    # Download
    # ===============================
    st.subheader("⬇️ Unduh Hasil Prediksi")
    st.download_button(
        label="Download hasil prediksi (CSV)",
        data=data.to_csv(index=False),
        file_name="hasil_prediksi_kesehatan_hewan.csv",
        mime="text/csv"
    )

else:
    st.info("Silakan upload file data untuk memulai prediksi.")
