from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Function to create time series sequences
def create_sequences(data, window_size=60):
    X = []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
    return np.array(X)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_plot = None
    error_message = None

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        start_date = request.form['start']
        end_date = request.form['end']

        try:
            # Download stock data
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                raise ValueError("No data found for the given ticker and date range.")

            close_prices = df['Close'].values.reshape(-1, 1)

            # Ensure enough data points
            if len(close_prices) < 60:
                raise ValueError(f"Only {len(close_prices)} trading days found. Please select a longer date range (at least 60 trading days).")

            # Paths to model and scaler
            scaler_path = f'models/{ticker}_scaler.pkl'
            model_path = f'models/{ticker}_model.h5'

            if not os.path.exists(scaler_path) or not os.path.exists(model_path):
                raise FileNotFoundError("Model not trained. Please train it first.")

            # Load scaler and model
            scaler = joblib.load(scaler_path)
            model = load_model(model_path, compile=False)

            # Scale and sequence the data
            scaled_data = scaler.transform(close_prices)
            X = create_sequences(scaled_data)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Predict and inverse transform
            preds = model.predict(X)
            preds_rescaled = scaler.inverse_transform(preds)

            # Plot results
            plt.figure(figsize=(10, 5))
            plt.plot(df['Close'].values[60:], label='Actual')
            plt.plot(preds_rescaled, label='Predicted')
            plt.title(f'{ticker} Stock Price Prediction')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plt.savefig('static/prediction.png')
            plt.close()

            prediction_plot = 'static/prediction.png'

        except Exception as e:
            error_message = str(e)

    return render_template('index.html', prediction_plot=prediction_plot, error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
