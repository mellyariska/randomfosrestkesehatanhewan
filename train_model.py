import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_excel("data_mencit_tendik_alfinsuhanda.xlsx")

X = data.drop(columns=["Label Kesehatan"])
y = data["Label Kesehatan"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Save model & scaler
joblib.dump(model, "model_rf.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model dan scaler berhasil disimpan")
