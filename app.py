import os
import gdown
import zipfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ------------------------------------------------
# 🗂️ 1. Download + unzip models if missing
# ------------------------------------------------
def setup_models():
    models_dir = "models_output"
    zip_file = "models_output.zip"

    # Already exists
    if os.path.exists(models_dir):
        return True

    FILE_ID = "1Hz4UFpIbNdwhSJZlLUIxCIP4OMxnzKWJ"  # 👈 change if you reupload
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    st.info("⬇️ Downloading trained models from Google Drive... (~67MB)")
    try:
        gdown.download(url, zip_file, quiet=False)
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False

    # Extract
    if os.path.exists(zip_file):
        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(".")

            # Try to detect folder automatically
            extracted_items = [f for f in os.listdir(".") if os.path.isdir(f)]
            for folder in extracted_items:
                if folder.lower().startswith("models_output"):
                    if folder != models_dir:
                        os.rename(folder, models_dir)
                    break

            if os.path.exists(models_dir):
                st.success("✅ Models extracted successfully and ready!")
                return True
            else:
                st.error("❌ Could not find 'models_output' folder after extraction. Please verify ZIP structure.")
                return False

        except zipfile.BadZipFile:
            st.error("❌ Bad ZIP file — please make sure your Google Drive file is shared as 'Anyone with link'.")
            return False
    else:
        st.error("❌ Could not find downloaded zip file.")
        return False



# ------------------------------------------------
# ⚙️ 2. Helper functions
# ------------------------------------------------
def load_coin_models(coin_name):
    base_path = os.path.join("models_output", coin_name)
    if not os.path.exists(base_path):
        return None, None, None

    lstm_path = os.path.join(base_path, f"{coin_name}_lstm_best.h5")
    rnn_path = os.path.join(base_path, f"{coin_name}_rnn_best.h5")
    scaler_path = os.path.join(base_path, f"{coin_name}_scaler.pkl")

    lstm_model = load_model(lstm_path) if os.path.exists(lstm_path) else None
    rnn_model = load_model(rnn_path) if os.path.exists(rnn_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    return lstm_model, rnn_model, scaler


def plot_predictions(df, coin_name):
    plt.figure(figsize=(10, 5))
    plt.plot(df["actual"], label="Actual", color="black")
    plt.plot(df["rnn_pred"], label="RNN Prediction", color="orange")
    plt.plot(df["lstm_pred"], label="LSTM Prediction", color="blue")
    plt.title(f"{coin_name} - Actual vs Predicted")
    plt.legend()
    st.pyplot(plt)


def predict_next_days(model, data, scaler, window_size=60, days=15):
    data_scaled = scaler.transform(np.array(data).reshape(-1, 1)).flatten()
    seq = data_scaled[-window_size:]
    preds = []

    for _ in range(days):
        X = np.array(seq[-window_size:]).reshape(1, window_size, 1)
        pred_scaled = model.predict(X, verbose=0)[0][0]
        preds.append(pred_scaled)
        seq = np.append(seq, pred_scaled)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds


# ------------------------------------------------
# 🖥️ 3. Streamlit UI
# ------------------------------------------------
st.set_page_config(page_title="Crypto Prediction App", page_icon="📈", layout="centered")
st.title("📈 Crypto Price Prediction using RNN & LSTM")

# Check folder exists before listing
if not os.path.exists("models_output"):
    st.error("❌ 'models_output' folder not found after extraction. Please check your Google Drive link.")
    st.stop()

# List all coin folders
coin_folders = [f for f in os.listdir("models_output") if os.path.isdir(os.path.join("models_output", f))]
if not coin_folders:
    st.error("No coin models found inside 'models_output'. Make sure your ZIP has correct structure.")
    st.stop()

coin_folders.sort()
selected_coin = st.selectbox("Select a coin:", coin_folders)

if selected_coin:
    st.subheader(f"🔍 Results for {selected_coin}")

    pred_path = os.path.join("models_output", selected_coin, f"{selected_coin}_predictions.csv")
    if not os.path.exists(pred_path):
        st.error(f"Prediction file not found for {selected_coin}")
    else:
        df_preds = pd.read_csv(pred_path)
        plot_predictions(df_preds, selected_coin)

        lstm_model, rnn_model, scaler = load_coin_models(selected_coin)
        if lstm_model and scaler:
            last_close = df_preds["actual"].values
            future_preds = predict_next_days(lstm_model, last_close, scaler, window_size=60, days=15)

            st.subheader("📅 15-Day Future Price Prediction (LSTM)")
            st.line_chart(pd.DataFrame(future_preds, columns=["Predicted Price"]))
        else:
            st.warning("LSTM model or scaler not available for this coin.")

