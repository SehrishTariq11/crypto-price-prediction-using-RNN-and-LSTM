import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Step 1: Get all cleaned CSV files
# ------------------------------------------------
clean_files = glob.glob("cleaned_datasets/*_clean.csv")
print(f"Found {len(clean_files)} cleaned coin files.")

# Create folder for scaled outputs
os.makedirs("preprocessed_datasets", exist_ok=True)

# Dictionaries to store data
scaled_data = {}   # scaled prices
scalers = {}       # scalers for each coin
raw_data = {}      # original unscaled data
# Step 2: Process each coin
# ------------------------------------------------
for file in clean_files:
    coin_name = os.path.basename(file).replace("_clean.csv", "")
    print(f"🔹 Preprocessing: {coin_name}")

    # Load data
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort just to be sure
    df = df.sort_values('timestamp').reset_index(drop=True)

# Step 3: Scale the numeric columns (0–1)

from sklearn.preprocessing import MinMaxScaler

# Folder for scaled outputs
os.makedirs("preprocessed_datasets", exist_ok=True)

# Example: if you have multiple coin DataFrames in a dictionary
# Example structure: coin_dfs = {'BTC-USD': df_btc, 'ETH-USD': df_eth, ...}

scaled_data = {}
raw_data = {}
scalers = {}

# List of numeric columns to scale
numeric_cols = ['adjclose', 'open', 'high', 'low', 'close', 'volume']

# Load cleaned data into coin_dfs
coin_dfs = {}
clean_files = glob.glob("cleaned_datasets/*_clean.csv")
for file in clean_files:
    coin_name = os.path.basename(file).replace("_clean.csv", "")
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    coin_dfs[coin_name] = df


# ✅ Loop through all coins
for coin_name, df in coin_dfs.items():
    print(f"\n🚀 Processing {coin_name}...")

    # Step 1: Copy and scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    existing_cols = [c for c in numeric_cols if c in df.columns]
    df_scaled = df.copy()
    df_scaled[existing_cols] = scaler.fit_transform(df[existing_cols])

    # Step 2: Save results in dictionaries
    scaled_data[coin_name] = df_scaled
    raw_data[coin_name] = df
    scalers[coin_name] = scaler

    # Step 3: Save scaled CSV
    output_path = f"preprocessed_datasets/{coin_name}_scaled.csv"
    df_scaled.to_csv(output_path, index=False)

    print(f"✅ Scaled data saved to: {output_path}")

print("\n🎯 Preprocessing complete for ALL coins!")