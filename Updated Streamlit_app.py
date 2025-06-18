import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from sklearn.ensemble import RandomForestClassifier

# Streamlit App Title
st.set_page_config(page_title="Nifty 4H Candle Predictor", layout="centered")
st.title("📊 Nifty 4-Hour Candle Direction Predictor")

# When user clicks the button
if st.button("🔮 Predict Next 4H Candle"):

    with st.spinner("Fetching and analyzing data..."):

        try:
            # Download 4-hour candles for the last 60 days
            df = yf.download("^NSEI", period="60d", interval="4h")

            # Apply Technical Indicators
            df['RSI'] = RSIIndicator(close=df['Close']).rsi()
            df['EMA_10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
            df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
            df['MACD'] = MACD(close=df['Close']).macd()
            df['Returns'] = df['Close'].pct_change()

            # Drop rows with missing values
            df.dropna(inplace=True)

            # Target: 1 if next candle is higher, else 0
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

            # Select features and target
            features = ['RSI', 'EMA_10', 'EMA_20', 'MACD', 'Returns']
            X = df[features]
            y = df['Target']

            # Train Random Forest model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X[:-1], y[:-1])  # Exclude last row from training

            # Predict using the latest row
            latest_input = X.iloc[-1:]
            prediction = model.predict(latest_input)[0]
            confidence = model.predict_proba(latest_input)[0][prediction]

            # Show prediction
            if prediction == 1:
                st.success(f"📈 The next 4-hour Nifty candle is likely to go **UP** (Confidence: {confidence:.1%})")
            else:
                st.error(f"📉 The next 4-hour Nifty candle is likely to go **DOWN** (Confidence: {confidence:.1%})")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
