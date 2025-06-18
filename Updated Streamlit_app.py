import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Nifty Next Candle Predictor", layout="centered")
st.title("📈 Nifty Next 5-Min Candle Predictor")

if st.button("🔮 Predict Next Candle"):

    with st.spinner("Fetching data and predicting..."):

        data = yf.download("^NSEI", period="5d", interval="5m")

        # Ensure 'Close' has no NaN and is 1D
        close = data['Close'].dropna()

        # Only calculate if there's enough data
        if len(close) > 20:
            data['RSI'] = ta.momentum.RSIIndicator(close=close, window=14).rsi().values.ravel()
            data['EMA_10'] = ta.trend.EMAIndicator(close=close, window=10).ema_indicator()
            data['EMA_20'] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
            data['MACD'] = ta.trend.MACD(close=close).macd()
            data['Returns'] = close.pct_change()

            data.dropna(inplace=True)

            data['Target'] = data['Close'].shift(-1) > data['Close']
            data['Target'] = data['Target'].astype(int)

            features = ['RSI', 'EMA_10', 'EMA_20', 'MACD', 'Returns']
            X = data[features]
            y = data['Target']

            model = RandomForestClassifier()
            model.fit(X[:-1], y[:-1])  # leave last row

            last_row = X.iloc[-1:]
            prediction = model.predict(last_row)[0]

            if prediction == 1:
                st.success("✅ Next Candle May Go UP 📈")
            else:
                st.error("❌ Next Candle May Go DOWN 📉")
        else:
            st.error("Not enough data to make a prediction. Try again later.")
