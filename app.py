import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# JUDUL DASHBOARD
# ===============================
st.set_page_config(page_title="Dashboard Kesehatan Hewan - Random Forest", layout="wide")
st.title("🐭 Dashboard Prediksi Kesehatan Hewan (Random Forest)")
st.write("Implementasi Machine Learning untuk Prediksi Kesehatan Hewan Percobaan")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    return pd.read_excel("data_mencit_tendik_alfinsuhanda.xlsx")

data = load_data()

st.subheader("📊 Data Asli")
st.dataframe(data.head())

# ===============================
# FITUR & LABEL
# ===============================
X = data.drop(columns=["Label Kesehatan"])
y = data["Label Kesehatan"]

# ===============================
# SPLIT & SCALING
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# TRAIN MODEL
# ===============================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ===============================
# EVALUASI MODEL
# ===============================
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

st.subheader("✅ Evaluasi Model")
st.metric("Akurasi Model", f"{acc:.2%}")

with st.expander("Lihat Classification Report"):
    st.text(classification_report(y_test, y_pred))

# ===============================
# PREDIKSI DATA BARU
# ===============================
st.subheader("🔮 Prediksi Kesehatan Hewan Baru")

st.write("Masukkan nilai parameter fisiologis hewan:")

input_data = {}

for col in X.columns:
    input_data[col] = st.number_input(
        f"{col}",
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)

# ===============================
# HASIL PREDIKSI
# ===============================
st.subheader("📌 Hasil Prediksi")

if prediction == "Sehat":
    st.success(f"✅ Status Kesehatan: **{prediction}**")
elif prediction == "Risiko Rendah":
    st.warning(f"⚠️ Status Kesehatan: **{prediction}**")
else:
    st.error(f"❌ Status Kesehatan: **{prediction}**")

st.write("Probabilitas Prediksi:")
proba_df = pd.DataFrame(
    prediction_proba,
    columns=model.classes_
)
st.dataframe(proba_df)
