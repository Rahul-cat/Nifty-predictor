import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# Load Models
model = joblib.load("rf_model.pkl")
hold_model = joblib.load("hold_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔮 Nifty Option Signal Predictor")

# Get Latest NIFTY Data
df = yf.download("^NSEI", interval="5m", period="1d")
df.dropna(inplace=True)

# Add Technical Indicators
df['EMA_10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
macd = MACD(close=df['Close'])
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()

# Prepare features
df.dropna(inplace=True)
latest = df.iloc[-1:]
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_10', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
X = scaler.transform(latest[features])

# Predict
signal = model.predict(X)[0]
hold_minutes = int(hold_model.predict(X)[0])

# Output
st.write("## 📊 Prediction Result")
st.write(f"**Trade Type:** {'📈 CALL' if signal == 1 else '📉 PUT'}")
st.write(f"**Suggested Holding Time:** {hold_minutes} minutes")
