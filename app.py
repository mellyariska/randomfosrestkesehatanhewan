import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    return pd.read_excel("data_mencit_tendik_alfinsuhanda.xlsx")

data = load_data()

# ===============================
# PREPARE DATA
# ===============================
X = data.drop(columns=["ID_Tikus", "Label Kesehatan"])
y = data["Label Kesehatan"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# TRAIN MODEL (DI CLOUD)
# ===============================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ===============================
# EVALUATION
# ===============================
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

# ===============================
# STREAMLIT UI
# ===============================
st.title("Dashboard Prediksi Kesehatan Hewan Percobaan")
st.subheader("Model: Random Forest")

st.success(f"Akurasi Model: {acc:.2%}")

st.markdown("### Input Data Baru")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, float(X[col].min()), float(X[col].max()))

if st.button("Prediksi Kesehatan"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.subheader("Hasil Prediksi")
    st.info(f"Status Kesehatan: **{prediction}**")
