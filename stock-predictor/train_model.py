import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_model(ticker='AAPL', start_date='2015-01-01', end_date='2024-12-31'):
    # 1. Download Data
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise Exception("No data found. Check your ticker or date range.")

    close_prices = df['Close'].values.reshape(-1, 1)

    # 2. Normalize
    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(close_prices)

    # 3. Create sequences
    X, y = create_sequences(close_scaled)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 4. Split into train and test
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]

    # 5. Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # 6. Save model and scaler
    model.save(f'models/{ticker}_model.h5')
    joblib.dump(scaler, f'models/{ticker}_scaler.pkl')

    print(f"[✔] Model and scaler saved for {ticker}")

# Run only if script is executed directly
if __name__ == "__main__":
    train_model(ticker='AAPL', start_date='2015-01-01', end_date='2024-12-31')
