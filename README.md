# 📈 Stock Price Prediction with LSTM (Deployed using Flask)

This project is a web application for predicting stock prices using LSTM (Long Short-Term Memory) neural networks. It uses historical stock price data, trains a deep learning model, and displays both actual and predicted values on a graph.

---

## Features

- Predict stock prices based on historical data
- Uses LSTM model for time series forecasting
- Built with Flask web framework
- User inputs:
  - Stock ticker (e.g., AAPL, TSLA)
  - Start and end date for historical data
- Visualizes actual vs predicted stock prices

---

## Tech Stack

- **Frontend**: HTML, CSS (Jinja2 templating)
- **Backend**: Flask (Python)
- **Data**: yFinance for stock data
- **ML Model**: Keras (LSTM), scikit-learn (scaler)
- **Plotting**: Matplotlib
