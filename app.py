import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# Load model & scaler
# ===============================
model = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# UI
# ===============================
st.set_page_config(
    page_title="Dashboard Prediksi Kesehatan Hewan",
    layout="wide"
)

st.title("🐁 Dashboard Prediksi Kesehatan Hewan Percobaan")
st.markdown(
    "Implementasi **Machine Learning Random Forest** untuk prediksi kesehatan hewan percobaan."
)

# ===============================
# Upload Data
# ===============================
uploaded_file = st.file_uploader(
    "📤 Upload data (Excel)",
    type=["xlsx"]
)

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    st.subheader("📊 Data Input")
    st.dataframe(data.head())

    # Preprocessing
    X = data.drop(columns=["Label Kesehatan"], errors="ignore")
    X_scaled = scaler.transform(X)

    # Prediction
    data["Prediksi Kesehatan"] = model.predict(X_scaled)

    st.subheader("🧠 Hasil Prediksi")
    st.dataframe(data)

    # Visualization
    st.subheader("📈 Distribusi Prediksi")
    counts = data["Prediksi Kesehatan"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Jumlah")
    ax.set_xlabel("Kategori")
    ax.set_title("Prediksi Kesehatan Hewan")
    st.pyplot(fig)

    # Download
    st.download_button(
        "⬇️ Download hasil prediksi",
        data=data.to_csv(index=False),
        file_name="hasil_prediksi_kesehatan.csv",
        mime="text/csv"
    )
else:
    st.info("Silakan upload file Excel untuk memulai prediksi.")
