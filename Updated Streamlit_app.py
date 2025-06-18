import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Nifty Next Candle Predictor", layout="centered")
st.title("📈 Nifty Next 5-Min Candle Predictor")

if st.button("🔮 Predict Next Candle"):
    with st.spinner("Fetching data and predicting..."):

        # Step 1: Fetch 5-min interval data
        data = yf.download("^NSEI", period="5d", interval="5m")

        # Step 2: Basic sanity checks
        if data.empty or 'Close' not in data.columns:
            st.error("Failed to fetch data or 'Close' column missing.")
            st.stop()

        # Step 3: Clean Close prices
        close_prices = data['Close'].copy()
        close_prices = close_prices.fillna(method='ffill').fillna(method='bfill')
        if close_prices.ndim != 1:
            close_prices = close_prices.squeeze()

        # Step 4: Compute indicators safely
        try:
            data['RSI'] = ta.momentum.RSIIndicator(close=close_prices).rsi()
            data['EMA_10'] = ta.trend.EMAIndicator(close=close_prices, window=10).ema_indicator()
            data['EMA_20'] = ta.trend.EMAIndicator(close=close_prices, window=20).ema_indicator()
            data['MACD'] = ta.trend.MACD(close=close_prices).macd()
            data['Returns'] = close_prices.pct_change()
        except Exception as e:
            st.error(f"Error computing indicators: {e}")
            st.stop()

        # Step 5: Drop missing values
        data.dropna(inplace=True)

        # Step 6: Define prediction target
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

        # Step 7: Feature selection
        features = ['RSI', 'EMA_10', 'EMA_20', 'MACD', 'Returns']
        X = data[features]
        y = data['Target']

        # Step 8: Train model (leave last row out for prediction)
        model = RandomForestClassifier()
        model.fit(X[:-1], y[:-1])

        # Step 9: Make prediction
        last_row = X.iloc[[-1]]
        prediction = model.predict(last_row)[0]

        # Step 10: Show result
        if prediction == 1:
            st.success("✅ Next Candle May Go UP 📈")
        else:
            st.error("❌ Next Candle May Go DOWN 📉")
