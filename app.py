import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

#PAGE TITLE

st.set_page_config(
    page_title="Stock Price Prediction",
    layout="wide"
)

st.title("📈 Stock Price Prediction using Machine Learning")

st.write(
    "Predict future stock prices using Machine Learning and historical stock market data."
)


# SIDEBAR

st.sidebar.header("Select Stock")

stock = st.sidebar.selectbox(
    "Choose Stock",
    (
        "AAPL",
        "MSFT",
        "GOOGL",
        "TSLA",
        "AMZN",
        "META",
        "RELIANCE.NS",
        "TCS.NS"
    )
)

# DOWNLOAD DATA

data = yf.download(
    stock,
    start="2020-01-01",
    end="2025-01-01"
)

st.subheader("Stock Dataset")

st.write(data.tail())

# FEATURE ENGINEERING

data["MA10"] = data["Close"].rolling(window=10).mean()

data["MA50"] = data["Close"].rolling(window=50).mean()

data["Daily_Return"] = data["Close"].pct_change()

data["Target"] = data["Close"].shift(-1)

data.dropna(inplace=True)


# FEATURES


features = [
    "Open",
    "High",
    "Low",
    "Volume",
    "MA10",
    "MA50",
    "Daily_Return"
]

X = data[features]

y = data["Target"]


# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# MODEL TRAINING


model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# PREDICTIONS


predictions = model.predict(X_test)


# MODEL ACCURACY

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

st.subheader("Model Performance")

st.write("### Mean Absolute Error:", round(mae, 2))

st.write("### R2 Score:", round(r2, 2))

# ACTUAL VS PREDICTED GRAPH


st.subheader("Actual vs Predicted Prices")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    y_test.values,
    label="Actual Price"
)

ax.plot(
    predictions,
    label="Predicted Price"
)

ax.set_xlabel("Days")

ax.set_ylabel("Price")

ax.legend()

st.pyplot(fig)

# NEXT DAY PREDICTION


latest_data = X.tail(1)

next_day_prediction = model.predict(latest_data)

st.subheader("Next Day Prediction")

st.success(
    f"Predicted Next Day Closing Price: ${round(next_day_prediction[0], 2)}"
)

# STOCK TREND GRAPH

st.subheader("Stock Closing Price Trend")

fig2, ax2 = plt.subplots(figsize=(12, 6))

ax2.plot(data["Close"])

ax2.set_xlabel("Date")

ax2.set_ylabel("Closing Price")

st.pyplot(fig2)