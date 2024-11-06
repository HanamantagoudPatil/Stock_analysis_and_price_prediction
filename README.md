**Stock Price Prediction App**

This repository contains a Streamlit application for predicting stock prices using historical data. The project utilizes the yfinance library to gather data and implements multiple forecasting models.

**Project Overview:**

Data Collection: Historical stock price data was collected using the yfinance library

Exploratory Data Analysis (EDA): Conducted detailed EDA on the collected data over the past 5 years, including visualizations and statistical analysis to identify trends and seasonality.

**Model Building:**

ARIMA, SARIMAX, and Linear Regression: Implemented these traditional time series models using the 5-year dataset to establish baseline predictions and analyze performance.

LSTM Model: Utilized a Long Short-Term Memory (LSTM) model trained on the 20-year dataset for enhanced performance in capturing complex patterns, building on insights gained from the initial models.

Deployment: The final prediction model is deployed in Streamlit app, allowing users to input parameters and obtain stock price forecasts.

Try the App
You can try out the Stock Price Prediction App here.  <https://stock-price-prediction-hnp.streamlit.app/>
