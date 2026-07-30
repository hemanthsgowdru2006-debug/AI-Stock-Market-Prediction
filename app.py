from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Stock Market Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

SUPPORTED_STOCKS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA"
]

selected_stock = st.sidebar.selectbox(
    "Select Stock",
    SUPPORTED_STOCKS
)

st.sidebar.success(
    f"Current Stock : {selected_stock}"
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------

stock = pd.read_csv(
    DATA_DIR /
    selected_stock /
    "processed" /
    "processed_stock_data.csv"
)

forecast = pd.read_csv(
    REPORTS_DIR /
    selected_stock /
    "prophet_forecast.csv"
)

sentiment = pd.read_csv(
    REPORTS_DIR /
    selected_stock /
    "news_sentiment.csv"
)

signal = pd.read_csv(
    RESULTS_DIR /
    f"{selected_stock}_recommendation.csv"
)

metrics_file = REPORTS_DIR / "lstm_metrics.csv"

if metrics_file.exists():

    metrics = pd.read_csv(metrics_file)

else:

    metrics = pd.DataFrame()

# --------------------------------------------------
# Data Preparation
# --------------------------------------------------

stock["Date"] = pd.to_datetime(stock["Date"])

if "Date" in forecast.columns:

    forecast["Date"] = pd.to_datetime(
        forecast["Date"]
    )

elif "ds" in forecast.columns:

    forecast["ds"] = pd.to_datetime(
        forecast["ds"]
    )

if "Date" in sentiment.columns:

    sentiment["Date"] = pd.to_datetime(
        sentiment["Date"],
        errors="coerce"
    )


# --------------------------------------------------
# Dashboard Title
# --------------------------------------------------

st.title(
    f"📈 {selected_stock} AI Stock Market Prediction Dashboard"
)

st.caption(
    "AI-powered Stock Forecasting • Sentiment Analysis • Trading Signal Generation"
)

st.markdown("---")


# --------------------------------------------------
# KPI Cards
# --------------------------------------------------

st.subheader("📊 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Current Price",
        f"${signal['Current Price'][0]:.2f}"
    )

with col2:
    st.metric(
        "Predicted Price",
        f"${signal['Predicted Price'][0]:.2f}"
    )

with col3:

    recommendation = signal["Recommendation"][0]

    if recommendation == "BUY":
        st.success("🟢 BUY")

    elif recommendation == "SELL":
        st.error("🔴 SELL")

    else:
        st.warning("🟡 HOLD")


col4, col5, col6 = st.columns(3)

latest = stock.iloc[-1]

trend = "Bullish" if latest["SMA20"] > latest["SMA50"] else "Bearish"

with col4:
    st.metric(
        "Market Trend",
        trend
    )

with col5:
    st.metric(
        "Confidence",
        f"{signal['Confidence (%)'][0]:.1f}%"
    )

with col6:
    st.metric(
        "RSI",
        f"{latest['RSI']:.2f}"
    )

st.markdown("---")


# --------------------------------------------------
# Historical Stock Price
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(14,6))

ax.plot(
    stock["Date"],
    stock["Close"],
    linewidth=2
)

ax.set_title(
    f"{selected_stock} Historical Closing Price"
)

ax.set_xlabel("Date")

ax.set_ylabel("Close Price")

ax.grid(True)

ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=3)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

plt.xticks(rotation=45)

plt.tight_layout()

st.pyplot(fig)


# --------------------------------------------------
# Forecast
# --------------------------------------------------

st.subheader("🔮 30-Day Price Forecast")

fig, ax = plt.subplots(figsize=(15,6))

if "Date" in forecast.columns:
    x = pd.to_datetime(forecast["Date"])
else:
    x = pd.to_datetime(forecast["ds"])

if "Predicted_Close" in forecast.columns:
    y = forecast["Predicted_Close"]
else:
    y = forecast["yhat"]

ax.plot(
    x,
    y,
    marker="o",
    markersize=4,
    linewidth=2,
    color="dodgerblue",
    label="Predicted Price"
)

ax.set_title(f"{selected_stock} 30-Day Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Price")

ax.grid(alpha=0.3)

# Show only weekly dates
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))

plt.xticks(rotation=45)

plt.legend()

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# Market Sentiment
# --------------------------------------------------

st.subheader("📰 Market Sentiment")

summary = sentiment["Sentiment"].value_counts()

fig, ax = plt.subplots(figsize=(5,3.5))

summary.plot(
    kind="bar",
    edgecolor="black",
    ax=ax
)

ax.set_title("Sentiment Distribution")

ax.set_xlabel("Sentiment")

ax.set_ylabel("Count")

plt.tight_layout()

st.pyplot(fig)


# --------------------------------------------------
# Trading Recommendation
# --------------------------------------------------

st.markdown("---")

st.subheader("💹 Trading Recommendation Details")

st.dataframe(
    signal,
    use_container_width=True
)

st.markdown("---")

st.subheader("📊 Technical Indicators")

latest = stock.iloc[-1]

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "RSI",
    f"{latest['RSI']:.2f}"
)

c2.metric(
    "MACD",
    f"{latest['MACD']:.2f}"
)

c3.metric(
    "SMA20",
    f"{latest['SMA20']:.2f}"
)

c4.metric(
    "EMA20",
    f"{latest['EMA20']:.2f}"
)

# --------------------------------------------------
# Model Performance
# --------------------------------------------------

st.subheader("🤖 LSTM Model Performance")

if metrics_file.exists():

    metrics = pd.read_csv(metrics_file)

    if len(metrics.columns) >= 2:

        col1,col2,col3,col4 = st.columns(4)

        col1.metric(
            metrics.iloc[0,0],
            metrics.iloc[0,1]
        )

        col2.metric(
            metrics.iloc[1,0],
            metrics.iloc[1,1]
        )

        col3.metric(
            metrics.iloc[2,0],
            metrics.iloc[2,1]
        )

        col4.metric(
            metrics.iloc[3,0],
            metrics.iloc[3,1]
        )

    else:

        st.dataframe(metrics,use_container_width=True)

else:

    st.warning("LSTM metrics file not found.")
# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")
st.success(
    f"Currently Viewing : {selected_stock}"
)

st.info(
    f"""
📈 AI Stock Market Prediction Dashboard

Selected Stock : {selected_stock}

Models Used

• Prophet Forecasting

• LSTM Deep Learning

• NLP Sentiment Analysis

• Technical Indicators

• Trading Signal Generation
"""
)