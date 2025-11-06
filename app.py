import streamlit as st
import os
import zipfile
import gdown
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -----------------------------
# CONFIG
# -----------------------------
FILE_ID = "1Hz4UFpIbNdwhSJZlLUIxCIP4OMxnzKWJ"  # 👈 put your Google Drive File ID here
ZIP_PATH = "models_output.zip"
MODELS_DIR = "models_output"

# -----------------------------
# STEP 1: SAFE SETUP FUNCTION
# -----------------------------
def setup_models():
    try:
        if not os.path.exists(MODELS_DIR):
            st.info("⬇️ Downloading trained models from Google Drive... (~67MB)")
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, ZIP_PATH, quiet=False)

            with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(".")
            st.success("✅ Models extracted successfully!")

        # Handle if folder name inside zip isn't exact
        if not os.path.exists(MODELS_DIR):
            for f in os.listdir("."):
                if os.path.isdir(f) and "model" in f.lower():
                    os.rename(f, MODELS_DIR)
                    break

        if not os.path.exists(MODELS_DIR):
            st.error("❌ 'models_output' folder not found after extraction. Please check your Google Drive link.")
            return False

        return True

    except Exception as e:
        st.error(f"⚠️ Error setting up models: {e}")
        return False

# -----------------------------
# STEP 2: LOAD MODELS & PREDICTIONS
# -----------------------------
def load_coin_data(coin_folder):
    """Load scaler, model, and prediction data for a specific coin."""
    try:
        scaler = joblib.load(os.path.join(coin_folder, f"{coin_name}_scaler.pkl"))
        lstm_model = load_model(os.path.join(coin_folder, f"{coin_name}_lstm_best.h5"))
        preds = pd.read_csv(os.path.join(coin_folder, f"{coin_name}_predictions.csv"))
        return scaler, lstm_model, preds
    except Exception as e:
        st.error(f"Error loading {coin_folder}: {e}")
        return None, None, None

# -----------------------------
# STEP 3: FORECAST FUNCTION
# -----------------------------
def forecast_next_days(model, scaler, series, window_size=60, days=15):
    data_scaled = scaler.transform(series)
    seq = data_scaled[-window_size:].reshape(1, window_size, 1)
    preds = []
    for _ in range(days):
        pred = model.predict(seq, verbose=0)
        preds.append(pred[0][0])
        seq = np.append(seq[:, 1:, :], [[pred]], axis=1)
    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds_inv

# -----------------------------
# STEP 4: STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Crypto Price Prediction", page_icon="💰", layout="wide")
st.title("📈 Crypto Price Prediction using RNN & LSTM")

ok = setup_models()

if not ok:
    st.stop()

# Get all coin folders
coin_folders = [f for f in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, f))]

if not coin_folders:
    st.error("No coin model folders found in models_output.")
    st.stop()

# Dropdown for coin selection
coin_name = st.selectbox("Select a cryptocurrency:", sorted(coin_folders))

coin_folder = os.path.join(MODELS_DIR, coin_name)
scaler, lstm_model, preds_df = load_coin_data(coin_folder)

if scaler is None:
    st.stop()

# -----------------------------
# STEP 5: SHOW ACTUAL VS PREDICTED
# -----------------------------
st.subheader(f"📊 {coin_name} - Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(preds_df['actual'], label='Actual', linewidth=2)
ax.plot(preds_df['lstm_pred'], label='Predicted (LSTM)', linestyle='dashed')
ax.set_title(f"{coin_name} - Actual vs LSTM Predicted")
ax.legend()
st.pyplot(fig)

# -----------------------------
# STEP 6: 15-DAY FORECAST
# -----------------------------
st.subheader("🔮 15-Day Future Price Forecast")

# Load recent close prices for forecasting
try:
    df_pred_source = pd.read_csv(os.path.join(coin_folder, f"{coin_name}_predictions.csv"))
    series = df_pred_source['actual'].values.reshape(-1, 1)
    preds_future = forecast_next_days(lstm_model, scaler, series)
    st.line_chart(preds_future)
    st.success("✅ Forecast generated successfully!")
except Exception as e:
    st.error(f"Forecast failed: {e}")

st.caption("Made with ❤️ using RNN & LSTM on cryptocurrency datasets.")
