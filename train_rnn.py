import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

# Function to create sequences
def create_sequences(data, time_step):
    X, y = [], []
    if len(data) <= time_step:
        return np.array([]), np.array([])
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Function to create RNN model
def create_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        SimpleRNN(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train RNN on a single coin dataset
def train_rnn_on_coin(file_path, time_step=60, epochs=20, batch_size=32, verbose=1):
    coin_name = os.path.basename(file_path).replace("_scaled.csv", "")
    print(f"\n🚀 Training RNN for: {coin_name}")

    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    data = df[['close']].values

    X, y = create_sequences(data, time_step)
    if X.size == 0:
        print(f"⚠️ Not enough data for {coin_name}. Skipping...")
        return None

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = create_rnn_model((X_train.shape[1], 1))
    print("🧠 Training RNN...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size, verbose=verbose)

    rnn_path = f"trained_models/{coin_name}_RNN.h5"
    os.makedirs("trained_models", exist_ok=True)
    model.save(rnn_path)
    print(f"✅ RNN model saved: {rnn_path}")
    return model
