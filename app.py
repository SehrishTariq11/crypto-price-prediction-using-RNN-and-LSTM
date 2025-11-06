# app.py
import streamlit as st
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf

# -----------------------------
# Add src folder to Python path
# -----------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from train_rnn import create_rnn_model
from train_lstm import create_lstm_model

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Crypto Price Prediction", layout="wide")
st.title("📈 Crypto Price Prediction Dashboard")

# -----------------------------
# Paths for ZIP files
# -----------------------------
PREPROCESSED_ZIP = "/content/preprocessed_datasets.zip"
TRAINED_MODELS_ZIP = "/content/trained_models.zip"

# -----------------------------
# Unzip preprocessed datasets if folder does not exist
# -----------------------------
if not os.path.exists("preprocessed_datasets"):
    if os.path.exists(PREPROCESSED_ZIP):
        with zipfile.ZipFile(PREPROCESSED_ZIP, 'r') as zip_ref:
            zip_ref.extractall("preprocessed_datasets")
        st.success("✅ Preprocessed datasets unzipped.")
    else:
        st.error(f"{PREPROCESSED_ZIP} not found!")
        st.stop()

# -----------------------------
# Unzip trained models if folder does not exist
# -----------------------------
if not os.path.exists("trained_models"):
    if os.path.exists(TRAINED_MODELS_ZIP):
        with zipfile.ZipFile(TRAINED_MODELS_ZIP, 'r') as zip_ref:
            zip_ref.extractall("trained_models")
        st.success("✅ Trained models unzipped.")
    else:
        st.error(f"{TRAINED_MODELS_ZIP} not found!")
        st.stop()

# -----------------------------
# Detect all coins
# -----------------------------
DATA_DIR = "preprocessed_datasets"
coin_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_scaled.csv")]
if not coin_files:
    st.error("No CSV files found in preprocessed_datasets!")
    st.stop()

coins = [f.replace("_scaled.csv", "") for f in coin_files]
selected_coin = st.selectbox("Select a Coin", coins)

# -----------------------------
# Load selected coin CSV
# -----------------------------
df_scaled = pd.read_csv(os.path.join(DATA_DIR, f"{selected_coin}_scaled.csv"))

# -----------------------------
# Prepare sequences for prediction
# -----------------------------
TIME_STEPS = 60
data_scaled = df_scaled[['close']].values

def create_sequences(data, time_steps=TIME_STEPS):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i,0])
        y.append(data[i,0])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled)
X = X.reshape(-1, TIME_STEPS, 1)
y = np.array(y)

# -----------------------------
# Load RNN & LSTM models
# -----------------------------
rnn_path = f"trained_models/{selected_coin}_RNN.h5"
lstm_path = f"trained_models/{selected_coin}_LSTM.h5"

if not os.path.exists(rnn_path) or not os.path.exists(lstm_path):
    st.error(f"Trained models for {selected_coin} not found!")
    st.stop()

rnn_model = tf.keras.models.load_model(rnn_path)
lstm_model = tf.keras.models.load_model(lstm_path)

# -----------------------------
# Choose best model based on loss
# -----------------------------
rnn_loss = rnn_model.evaluate(X, y, verbose=0)
lstm_loss = lstm_model.evaluate(X, y, verbose=0)

if lstm_loss < rnn_loss:
    best_model = lstm_model
    model_name = "LSTM"
else:
    best_model = rnn_model
    model_name = "RNN"

st.success(f"✅ Best model for {selected_coin}: {model_name} (Loss: {min(rnn_loss,lstm_loss):.4f})")

# -----------------------------
# Predict historical prices
# -----------------------------
predicted = best_model.predict(X)

st.subheader("📊 Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(y, label="Actual")
ax.plot(predicted, label="Predicted")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# -----------------------------
# 15-Day Forecast
# -----------------------------
st.subheader("🔮 15-Day Forecast")
future_input = data_scaled[-TIME_STEPS:].reshape(1, TIME_STEPS, 1)
future_prices = []

for _ in range(15):
    pred = best_model.predict(future_input)
    future_prices.append(pred[0,0])
    future_input = np.append(future_input[:,1:,:], [[pred]], axis=1)

fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.plot(future_prices, marker='o', linestyle='-', color='orange')
ax2.set_xlabel("Day")
ax2.set_ylabel("Predicted Price")
ax2.set_title(f"{selected_coin} 15-Day Forecast using {model_name}")
st.pyplot(fig2)

