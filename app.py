import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from datetime import date, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📈 Stock Price Prediction using Machine Learning")

st.write(
    "Predict the next trading day's closing stock price "
    "using historical stock market data and Random Forest Regression."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📊 Select Stock")

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


st.sidebar.info(
    "The model uses historical price, volume, "
    "moving averages, returns and volatility."
)


# ============================================================
# CURRENCY
# ============================================================

if stock.endswith(".NS"):
    currency = "₹"
else:
    currency = "$"


# ============================================================
# DOWNLOAD DATA
# ============================================================

@st.cache_data(ttl=3600)
def download_stock_data(stock):

    start_date = "2020-01-01"

    # Tomorrow is used because yfinance's end date is exclusive
    end_date = (
        date.today() + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    try:

        data = yf.download(
            stock,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )

    except Exception as e:

        st.error(
            f"Unable to download stock data: {e}"
        )

        return pd.DataFrame()

    return data


data = download_stock_data(stock)


# ============================================================
# CHECK DATA
# ============================================================

if data.empty:

    st.error(
        "No stock data was downloaded. "
        "Please check your internet connection or try another stock."
    )

    st.stop()


# ============================================================
# HANDLE YFINANCE MULTI-INDEX
# ============================================================

if isinstance(data.columns, pd.MultiIndex):

    data.columns = data.columns.get_level_values(0)


data = data.copy()


# ============================================================
# STOCK DATASET
# ============================================================

st.subheader("📋 Stock Dataset")

st.write(
    f"Showing the latest available data for **{stock}**"
)

st.dataframe(
    data.tail(10),
    use_container_width=True
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

data["MA10"] = (
    data["Close"]
    .rolling(window=10)
    .mean()
)

data["MA50"] = (
    data["Close"]
    .rolling(window=50)
    .mean()
)

data["Daily_Return"] = (
    data["Close"]
    .pct_change()
)

data["Previous_Close"] = (
    data["Close"]
    .shift(1)
)

data["Return_5D"] = (
    data["Close"]
    .pct_change(5)
)

data["Volatility_10D"] = (
    data["Daily_Return"]
    .rolling(window=10)
    .std()
)


# ============================================================
# TARGET VARIABLE
# ============================================================

# Today's features → Tomorrow's closing price

data["Target"] = (
    data["Close"]
    .shift(-1)
)


# ============================================================
# FEATURES
# ============================================================

features = [
    "Open",
    "High",
    "Low",
    "Volume",
    "MA10",
    "MA50",
    "Daily_Return",
    "Previous_Close",
    "Return_5D",
    "Volatility_10D"
]


# ============================================================
# MODEL DATA
# ============================================================

model_data = data.dropna(
    subset=features + ["Target"]
).copy()


X = model_data[features]

y = model_data["Target"]


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(
    len(model_data) * 0.80
)


X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]


y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


# ============================================================
# MODEL TRAINING
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# TEST SET PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)


r2 = r2_score(
    y_test,
    predictions
)


# ============================================================
# MAPE
# ============================================================

mape = (
    np.mean(
        np.abs(
            (y_test - predictions)
            / y_test
        )
    )
    * 100
)


# ============================================================
# BASELINE MODEL
# ============================================================

# Simple baseline:
# Tomorrow's price = today's closing price

baseline_predictions = (
    X_test["Previous_Close"]
)


baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)


baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)


# ============================================================
# MODEL PERFORMANCE DISPLAY
# ============================================================

st.subheader("📊 Model Performance")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Mean Absolute Error",
        f"{mae:.2f}"
    )


with col2:

    st.metric(
        "R² Score",
        f"{r2:.4f}"
    )


with col3:

    st.metric(
        "MAPE",
        f"{mape:.2f}%"
    )


# ============================================================
# EXPLANATION
# ============================================================

st.info(
    f"""
    **Model interpretation**

    • MAE: The model's predictions are off by approximately
    {currency}{mae:.2f} on average.

    • R² Score: {r2:.4f}. A value closer to 1 indicates
    that the model explains the variation in the test data well.

    • MAPE: {mape:.2f}%. This represents the average absolute
    percentage error of the predictions.

    **Important:** R² is not the same as prediction accuracy.
    """
)


# ============================================================
# TRAINING / TESTING PERIOD
# ============================================================

st.subheader("📅 Training and Testing Period")


period_col1, period_col2 = st.columns(2)


with period_col1:

    st.write("### Training Period")

    st.write(
        f"**{X_train.index.min().date()}** "
        f"→ "
        f"**{X_train.index.max().date()}**"
    )

    st.write(
        f"Training observations: "
        f"**{len(X_train)}**"
    )


with period_col2:

    st.write("### Testing Period")

    st.write(
        f"**{X_test.index.min().date()}** "
        f"→ "
        f"**{X_test.index.max().date()}**"
    )

    st.write(
        f"Testing observations: "
        f"**{len(X_test)}**"
    )


# ============================================================
# BASELINE COMPARISON
# ============================================================

st.subheader("📌 Random Forest vs Simple Baseline")


baseline_col1, baseline_col2 = st.columns(2)


with baseline_col1:

    st.write("### Random Forest")

    st.write(
        f"MAE: **{mae:.2f}**"
    )

    st.write(
        f"R²: **{r2:.4f}**"
    )


with baseline_col2:

    st.write("### Previous Close Baseline")

    st.write(
        f"MAE: **{baseline_mae:.2f}**"
    )

    st.write(
        f"R²: **{baseline_r2:.4f}**"
    )


if mae < baseline_mae:

    st.success(
        "The Random Forest model has lower MAE "
        "than the simple previous-close baseline."
    )

else:

    st.warning(
        "The Random Forest model does not outperform "
        "the simple previous-close baseline in MAE."
    )


# ============================================================
# ACTUAL VS PREDICTED GRAPH
# ============================================================

st.subheader("📈 Actual vs Predicted Prices")


fig, ax = plt.subplots(
    figsize=(14, 6)
)


ax.plot(
    y_test.index,
    y_test.values,
    label="Actual Price",
    color="blue",
    linewidth=2
)


ax.plot(
    y_test.index,
    predictions,
    label="Predicted Price",
    color="red",
    linewidth=2
)


ax.set_title(
    f"{stock} - Actual vs Predicted Closing Prices"
)


ax.set_xlabel("Date")

ax.set_ylabel(
    f"Price ({currency})"
)


ax.legend()

ax.grid(
    alpha=0.3
)


plt.xticks(
    rotation=45
)


plt.tight_layout()


st.pyplot(fig)


# ============================================================
# NEXT TRADING DAY PREDICTION
# ============================================================

st.subheader(
    "🔮 Next Trading Day Prediction"
)


# Get latest available feature data
# This does NOT require a known Target

latest_features = (
    data
    .dropna(subset=features)
    .iloc[-1:]
)


latest_X = latest_features[
    features
]


# ============================================================
# FINAL MODEL
# ============================================================

# Train a final model using all available
# historical observations with known targets

final_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)


final_model.fit(
    X,
    y
)


# ============================================================
# NEXT DAY PREDICTION
# ============================================================

next_day_prediction = (
    final_model
    .predict(latest_X)[0]
)


latest_close = (
    latest_features["Close"].iloc[0]
)


# Expected percentage change

predicted_change = (
    (next_day_prediction / latest_close) - 1
) * 100


# ============================================================
# DISPLAY NEXT DAY PREDICTION
# ============================================================

prediction_col1, prediction_col2 = st.columns(2)


with prediction_col1:

    st.metric(
        "Predicted Next Trading Day Close",
        f"{currency}{next_day_prediction:.2f}"
    )


with prediction_col2:

    st.metric(
        "Predicted Change",
        f"{predicted_change:+.2f}%"
    )


st.write(
    f"""
    Latest available closing price:
    **{currency}{latest_close:.2f}**

    Latest trading date:
    **{latest_features.index[-1].date()}**
    """
)


# ============================================================
# PRICE DIRECTION
# ============================================================

if predicted_change > 0:

    st.success(
        f"📈 Model predicts an increase of approximately "
        f"{predicted_change:.2f}%."
    )

elif predicted_change < 0:

    st.warning(
        f"📉 Model predicts a decrease of approximately "
        f"{abs(predicted_change):.2f}%."
    )

else:

    st.info(
        "Model predicts very little change."
    )


# ============================================================
# STOCK CLOSING PRICE TREND
# ============================================================

st.subheader(
    "📉 Stock Closing Price Trend"
)


fig2, ax2 = plt.subplots(
    figsize=(14, 6)
)


ax2.plot(
    data.index,
    data["Close"],
    color="blue",
    linewidth=1.5
)


ax2.set_title(
    f"{stock} Closing Price Trend"
)


ax2.set_xlabel("Date")


ax2.set_ylabel(
    f"Closing Price ({currency})"
)


ax2.grid(
    alpha=0.3
)


plt.xticks(
    rotation=45
)


plt.tight_layout()


st.pyplot(fig2)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "🔍 Feature Importance"
)


importance = pd.DataFrame(
    {
        "Feature": features,
        "Importance": final_model.feature_importances_
    }
)


importance = importance.sort_values(
    "Importance",
    ascending=False
)


fig3, ax3 = plt.subplots(
    figsize=(10, 6)
)


ax3.barh(
    importance["Feature"],
    importance["Importance"]
)


ax3.set_xlabel(
    "Importance"
)


ax3.set_ylabel(
    "Feature"
)


ax3.set_title(
    "Random Forest Feature Importance"
)


ax3.invert_yaxis()


plt.tight_layout()


st.pyplot(fig3)


# ============================================================
# PROJECT SUMMARY
# ============================================================

st.subheader(
    "📝 Project Summary"
)


st.write(
    f"""
    **Stock:** {stock}

    **Machine Learning Algorithm:** Random Forest Regression

    **Training Observations:** {len(X_train)}

    **Testing Observations:** {len(X_test)}

    **Mean Absolute Error:** {currency}{mae:.2f}

    **R² Score:** {r2:.4f}

    **MAPE:** {mape:.2f}%

    The model uses historical stock prices, trading volume,
    moving averages, returns and volatility to predict the
    next trading day's closing price.

    The model is evaluated using a chronological train-test
    split so that earlier observations are used for training
    and later observations are used for testing.
    """
)


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    """
    ⚠️ Disclaimer: This application is an educational machine
    learning project. Stock-price predictions are estimates and
    are not guaranteed. Actual prices can be affected by news,
    market sentiment, economic conditions and unexpected events.
    This application should not be considered financial advice.
    """
)
