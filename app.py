import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from train_rnn import create_rnn_model
from train_lstm import create_lstm_model

st.set_page_config(page_title="Crypto Price Prediction", layout="wide")
st.title("📈 Crypto Price Prediction Dashboard")

# -----------------------------
# Path to Colab preprocessed datasets
# -----------------------------
DATA_DIR = "/content/preprocessed_datasets"

if not os.path.exists(DATA_DIR):
    st.error(f"Folder '{DATA_DIR}' not found. Make sure CSVs exist in Colab.")
    st.stop()

# Detect all CSV files in folder
coin_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_scaled.csv")]
if not coin_files:
    st.error("No preprocessed CSV files found in the folder.")
    st.stop()

coins = [f.replace("_scaled.csv", "") for f in coin_files]
selected_coin = st.selectbox("Select a Coin", coins)

# Load the selected coin CSV
df_scaled = pd.read_csv(os.path.join(DATA_DIR, f"{selected_coin}_scaled.csv"))

# -----------------------------
# Prepare data for model
# -----------------------------
TIME_STEPS = 60
adj_min = df_scaled['adjclose'].min()
adj_max = df_scaled['adjclose'].max()
data_scaled = df_scaled[['adjclose']].values

X, y = [], []
for i in range(TIME_STEPS, len(data_scaled)):
    X.append(data_scaled[i-TIME_STEPS:i,0])
    y.append(data_scaled[i,0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1],1))

# -----------------------------
# Train models dynamically (short demo for Streamlit)
# -----------------------------
st.text("Training RNN model...")
rnn_model = create_rnn_model((X.shape[1], 1))
rnn_model.fit(X, y, epochs=3, batch_size=32, verbose=0)
rnn_loss = rnn_model.evaluate(X, y, verbose=0)

st.text("Training LSTM model...")
lstm_model = create_lstm_model((X.shape[1], 1))
lstm_model.fit(X, y, epochs=3, batch_size=32, verbose=0)
lstm_loss = lstm_model.evaluate(X, y, verbose=0)

# Pick best model
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
predicted_prices = predicted * (adj_max - adj_min) + adj_min
actual_prices = y * (adj_max - adj_min) + adj_min

st.subheader("📊 Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(actual_prices, label="Actual")
ax.plot(predicted_prices, label="Predicted")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# -----------------------------
# 15-Day Forecast
# -----------------------------
st.subheader("🔮 15-Day Forecast")
future_input = data_scaled[-TIME_STEPS:].reshape(1, TIME_STEPS,1)
future_prices = []

for _ in range(15):
    pred = best_model.predict(future_input)
    future_prices.append(pred[0,0])
    future_input = np.append(future_input[:,1:,:], [[pred]], axis=1)

future_prices = np.array(future_prices) * (adj_max - adj_min) + adj_min

fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.plot(future_prices, marker='o', linestyle='-', color='orange')
ax2.set_xlabel("Day")
ax2.set_ylabel("Predicted Price")
ax2.set_title(f"{selected_coin} 15-Day Forecast using {model_name}")
st.pyplot(fig2)
