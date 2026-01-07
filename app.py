import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(page_title="Dashboard Kesehatan Hewan", layout="wide")
st.title("🐭 Dashboard Prediksi Kesehatan Hewan (Random Forest)")
st.caption("Implementasi Machine Learning untuk Prediksi Kesehatan Hewan Percobaan")

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
y = data["Label Kesehatan"]
X = data.drop(columns=["Label Kesehatan"])

# ===============================
# ENCODING KATEGORIKAL
# ===============================
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

encoder = LabelEncoder()
for col in cat_cols:
    X[col] = encoder.fit_transform(X[col])

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# SCALING (NUMERIK SAJA)
# ===============================
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ===============================
# TRAIN RANDOM FOREST
# ===============================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# ===============================
# EVALUASI MODEL
# ===============================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("✅ Evaluasi Model")
st.metric("Akurasi", f"{acc:.2%}")

with st.expander("📄 Classification Report"):
    st.text(classification_report(y_test, y_pred))

# ===============================
# CONFUSION MATRIX (MATPLOTLIB)
# ===============================
st.subheader("🧩 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

fig, ax = plt.subplots()
im = ax.imshow(cm)

ax.set_xticks(range(len(model.classes_)))
ax.set_yticks(range(len(model.classes_)))
ax.set_xticklabels(model.classes_)
ax.set_yticklabels(model.classes_)

ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")
ax.set_title("Confusion Matrix")

for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)

# ===============================
# FEATURE IMPORTANCE
# ===============================
st.subheader("📈 Feature Importance")

fi_df = pd.DataFrame({
    "Fitur": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(fi_df)

fig, ax = plt.subplots()
ax.barh(fi_df["Fitur"], fi_df["Importance"])
ax.set_xlabel("Importance")
ax.set_title("Feature Importance Random Forest")
ax.invert_yaxis()
st.pyplot(fig)

# ===============================
# PREDIKSI DATA BARU (SATUAN)
# ===============================
st.subheader("🔮 Prediksi Data Baru")

input_data = {}
for col in X.columns:
    if col in num_cols:
        input_data[col] = st.number_input(
            f"{col}",
            value=float(X[col].mean())
        )
    else:
        input_data[col] = st.number_input(
            f"{col}",
            value=int(X[col].mode()[0])
        )

input_df = pd.DataFrame([input_data])
input_df[num_cols] = scaler.transform(input_df[num_cols])

pred = model.predict(input_df)[0]
proba = model.predict_proba(input_df)

st.subheader("📌 Hasil Prediksi")
st.success(f"Hasil Prediksi Kesehatan: **{pred}**")
st.dataframe(pd.DataFrame(proba, columns=model.classes_))

# ===============================
# PREDIKSI MASSAL (UPLOAD CSV) - FIX
# ===============================
st.subheader("📂 Prediksi Massal (Upload CSV)")

uploaded_file = st.file_uploader(
    "Upload file CSV (struktur sama dengan data training)",
    type=["csv"]
)

if uploaded_file:
    df_new = pd.read_csv(uploaded_file)
    st.write("📄 Data yang diunggah:")
    st.dataframe(df_new.head())

    # 🔹 Ambil kolom training (X.columns)
    expected_cols = X.columns.tolist()

    # 🔹 Cek kolom hilang
    missing_cols = set(expected_cols) - set(df_new.columns)
    if missing_cols:
        st.error(f"❌ Kolom berikut HILANG di CSV: {missing_cols}")
        st.stop()

    # 🔹 Buang kolom berlebih
    df_new = df_new[expected_cols]

    # 🔹 Encoding kategorikal
    for col in cat_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype(str)
            df_new[col] = LabelEncoder().fit_transform(df_new[col])

    # 🔹 Scaling numerik
    df_new[num_cols] = scaler.transform(df_new[num_cols])

    # 🔹 Prediksi
    df_new["Prediksi_Kesehatan"] = model.predict(df_new)

    st.success("✅ Prediksi massal berhasil")
    st.dataframe(df_new)

    # 🔹 Download hasil
    csv = df_new.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Hasil Prediksi",
        data=csv,
        file_name="hasil_prediksi_kesehatan.csv",
        mime="text/csv"
    )
