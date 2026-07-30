from pathlib import Path
from prophet import Prophet
import yfinance as yf
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

@st.cache_data(ttl=3600)
def generate_prophet_forecast(ticker):

    # Download latest 2 years of data
    df = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    df = df[["Date", "Close"]].dropna()

    df.columns = ["ds", "y"]

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    model.fit(df)

    future = model.make_future_dataframe(
        periods=30,
        freq="D"      # Calendar days
    )

    forecast = model.predict(future)

    future_forecast = forecast[
        forecast["ds"] > df["ds"].max()
    ][["ds", "yhat"]]

    return future_forecast

# --------------------------------------------------
# Load Data
# --------------------------------------------------

stock = pd.read_csv(
    DATA_DIR /
    selected_stock /
    "processed" /
    "processed_stock_data.csv"
)

forecast = generate_prophet_forecast(selected_stock)
# Keep only future forecast
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


st.subheader("🔮 Next 30-Day AI Forecast")

forecast_plot = forecast.copy()

fig, ax = plt.subplots(figsize=(15,6))

ax.plot(
    forecast_plot["ds"],
    forecast_plot["yhat"],
    marker="o",
    linewidth=2,
    label="Forecast"
)

ax.set_title(f"{selected_stock} Next 30 Business-Day Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Closing Price")

ax.grid(alpha=0.3)

locator = mdates.AutoDateLocator()
formatter = mdates.DateFormatter("%d-%b")

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

plt.xticks(rotation=45)

plt.legend()

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

        metric_dict = dict(zip(metrics["Metric"], metrics["Value"]))

        col1.metric(
           "MAE",
           f"{metric_dict['MAE']:.2f}"
        )

        col2.metric(
          "RMSE",
          f"{metric_dict['RMSE']:.2f}"
        )

        col3.metric(
            "R² Score",
            f"{metric_dict['R2']:.3f}"
        )

        col4.metric(
         "MAPE",
         f"{metric_dict['MAPE']:.2f}%"
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