import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Nifty 4H Candle Predictor", layout="centered")
st.title("📊 Nifty 4-Hour Candle Direction Predictor")

if st.button("🔮 Predict Next 4H Candle"):

    with st.spinner("Analyzing Nifty 4H trend..."):

        # Download Nifty 4-hour data
        df = yf.download("^NSEI", period="60d", interval="4h")

        if df.empty:
            st.error("❌ Unable to fetch data. Please try again.")
        else:
            # Technical Indicators
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['EMA_10'] = ta.trend.EMAIndicator(df['Close'], window=10).ema_indicator()
            df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['Returns'] = df['Close'].pct_change()

            df.dropna(inplace=True)

            # Create binary target: Will next candle close higher?
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

            # Prepare features
            features = ['RSI', 'EMA_10', 'EMA_20', 'MACD', 'Returns']
            X = df[features]
            y = df['Target']

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X[:-1], y[:-1])

            last_input = X.iloc[-1:]
            prediction = model.predict(last_input)[0]
            confidence = model.predict_proba(last_input)[0][prediction]

            # Output result
            if prediction == 1:
                st.success(f"📈 Prediction: Next 4-Hour Candle may go **UP** (Confidence: {confidence:.1%})")
            else:
                st.error(f"📉 Prediction: Next 4-Hour Candle may go **DOWN** (Confidence: {confidence:.1%})")
