import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

st.title("Stock Price Predictor App")

# Input for stock ticker symbol
stock = st.text_input("Enter The Stock ID", "INFY.NS")

# Set date range for historical data download
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)
infy_data = yf.download(stock, start, end)

# Load pre-trained LSTM model
lstm_model = load_model("stock_price_lstm_model.keras")

st.subheader("Stock Data")
st.write(infy_data)

# Define function to plot moving average graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange', label="Moving Average")
    plt.plot(full_data.Close, 'b', label="Close Price")
    if extra_data:
        plt.plot(extra_dataset, 'g', label="Extra Data")
    plt.legend()
    return fig

# Display 250-day Moving Average
st.subheader('Original Close Price and MA for 250 days')
infy_data['250 Day MA'] = infy_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15, 6), infy_data['250 Day MA'], infy_data, 0))

# Display 50-day Moving Average
st.subheader('Original Close Price and MA for 50 days')
infy_data['50 Day MA'] = infy_data.Close.rolling(50).mean()
st.pyplot(plot_graph((15, 6), infy_data['50 Day MA'], infy_data, 0))

# Set up data for LSTM prediction
splitting_len = int(len(infy_data) * 0.7)
x_test = pd.DataFrame(infy_data.Close[splitting_len:])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

# Prepare test data for prediction
x_infy_data = []
y_infy_data = []
for i in range(100, len(scaled_data)):
    x_infy_data.append(scaled_data[i-100:i])
    y_infy_data.append(scaled_data[i])

x_infy_data, y_infy_data = np.array(x_infy_data), np.array(y_infy_data)

# Generate predictions
predictions = lstm_model.predict(x_infy_data)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_infy_test = scaler.inverse_transform(y_infy_data)

# Display predictions vs original values
ploting_data = pd.DataFrame(
    {
        'Original Test Data': inv_y_infy_test.reshape(-1),
        'Predictions': inv_predictions.reshape(-1)
    },
    index=infy_data.index[splitting_len + 100:]
)

st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Future prediction input
future_days = st.number_input("Enter number of days to predict into the future", min_value=1, max_value=365, value=30)

# Predict future prices
last_100_days = scaled_data[-100:]  # Starting point for future predictions
future_predictions = []

for _ in range(future_days):
    next_pred = lstm_model.predict(last_100_days.reshape(1, -1, 1))
    future_predictions.append(next_pred[0, 0])
    last_100_days = np.append(last_100_days[1:], next_pred[0, 0])  # Shift and add next prediction

# Convert predictions back to original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate future dates for predictions
future_dates = pd.date_range(end + timedelta(days=1), periods=future_days, freq='D')
future_data = pd.DataFrame(future_predictions, index=future_dates, columns=['Future Predictions'])

st.subheader('Future Stock Price Predictions')
st.write(future_data)

# Plot historical data, test data predictions, and future predictions
fig = plt.figure(figsize=(15, 6))
plt.plot(infy_data.Close, label="Historical Data")
plt.plot(pd.concat([ploting_data['Predictions'], future_data['Future Predictions']], axis=0), label="Predictions")
plt.legend(["Historical Data", "Test Data Predictions", "Future Predictions"])
st.pyplot(fig)
