import os
import zipfile
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="📈 Crypto Price Prediction", layout="wide")

# -----------------------------
# Local paths
# -----------------------------
dataset_zip = "assets/preprocessed_datasets.zip"
models_zip = "assets/trained_models.zip"
dataset_folder = "preprocessed_datasets"
models_folder = "trained_models"

# -----------------------------
# Function to unzip if missing
# -----------------------------
def unzip_if_missing(zip_name, folder_name):
    if not os.path.exists(folder_name):
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(folder_name)
        st.write(f"📦 Unzipped {zip_name} into {folder_name}")

unzip_if_missing(dataset_zip, dataset_folder)
unzip_if_missing(models_zip, models_folder)

# -----------------------------
# Load datasets
# -----------------------------
data_files = [f for f in os.listdir(dataset_folder) if f.endswith(".csv")]
coins = [os.path.basename(f).replace("_scaled.csv","") for f in data_files]

# -----------------------------
# Load models
# -----------------------------
model_files = [f for f in os.listdir(models_folder) if f.endswith(".h5")]
models = {}
for f in model_files:
    coin_name = f.split("_")[0]
    models[coin_name] = load_model(os.path.join(models_folder, f))

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📈 Crypto Price Prediction Dashboard")

selected_coin = st.selectbox("Select a coin:", coins)

# Load dataset
df = pd.read_csv(os.path.join(dataset_folder, f"{selected_coin}_scaled.csv"))
df = df.sort_values('timestamp')
prices = df['close'].values.reshape(-1,1)

# Scale prices for prediction
scaler = MinMaxScaler(feature_range=(0,1))
prices_scaled = scaler.fit_transform(prices)

# -----------------------------
# Prepare data for model
# -----------------------------
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i,0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(prices_scaled, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# -----------------------------
# Make predictions
# -----------------------------
model = models[selected_coin]  # choose best model
y_pred_scaled = model.predict(X)
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1))
y_true = scaler.inverse_transform(y.reshape(-1,1))

# -----------------------------
# Plot actual vs predicted
# -----------------------------
st.subheader(f"Actual vs Predicted Prices for {selected_coin.capitalize()}")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(y_true, label="Actual")
ax.plot(y_pred, label="Predicted")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# -----------------------------
# Predict next 15 days
# -----------------------------
last_seq = prices_scaled[-seq_length:]
future_preds = []

current_seq = last_seq.copy()
for _ in range(15):
    pred = model.predict(current_seq.reshape(1,seq_length,1))
    future_preds.append(pred[0,0])
    current_seq = np.append(current_seq[1:], pred[0,0])

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))

st.subheader(f"Next 15 Days Predicted Prices for {selected_coin.capitalize()}")
fig2, ax2 = plt.subplots(figsize=(12,5))
ax2.plot(range(1,16), future_preds, marker='o', color='orange')
ax2.set_xlabel("Days")
ax2.set_ylabel("Predicted Price")
st.pyplot(fig2)
