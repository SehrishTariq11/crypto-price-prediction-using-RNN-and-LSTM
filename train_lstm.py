import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Function to create sequences
def create_sequences(data, time_step):
    X, y = [], []
    if len(data) <= time_step:
        return np.array([]), np.array([])
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# -----------------------------
# Train Models for Each Coin
# -----------------------------
time_step = 60

for file in data_files:
    coin_name = os.path.basename(file).replace("_scaled.csv", "")
    print(f"\n🚀 Training models for: {coin_name}")

    # Load dataset
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Use "close" price for training
    data = df[['close']].values

    # Create sequences
    X, y = create_sequences(data, time_step)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Skip if not enough data
    if X.size == 0:
        print(f"⚠️ Not enough data for {coin_name}. Skipping...")
        continue

    # Reshape for RNN/LSTM input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Train-test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # -----------------------------
    # LSTM Model
    # -----------------------------
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    print("🧠 Training LSTM...")
    lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                   epochs=20, batch_size=32, verbose=1, callbacks=[es])

    lstm_path = f"trained_models/{coin_name}_LSTM.h5"
    lstm_model.save(lstm_path)
    print(f"✅ LSTM model saved: {lstm_path}")
