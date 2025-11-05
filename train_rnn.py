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

    # SimpleRNN Model
    # -----------------------------
    rnn_model = Sequential([
        SimpleRNN(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        SimpleRNN(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    rnn_model.compile(optimizer='adam', loss='mean_squared_error')

    print("🧠 Training RNN...")
    rnn_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=20, batch_size=32, verbose=1, callbacks=[es])

    rnn_path = f"trained_models/{coin_name}_RNN.h5"
    rnn_model.save(rnn_path)
    print(f"✅ RNN model saved: {rnn_path}")