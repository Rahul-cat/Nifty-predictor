import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from sklearn.ensemble import RandomForestClassifier

# Set page config and title
st.set_page_config(page_title="Nifty 4H Predictor", layout="centered")
st.title("📊 Nifty 4-Hour Candle Direction Predictor")

# When button is clicked
if st.button("🔮 Predict Next 4H Candle"):

    with st.spinner("Fetching and analyzing data..."):

        try:
            # Download Nifty 50 4-hour data for 60 days
            df = yf.download("^NSEI", period="60d", interval="4h")

            # Calculate indicators (1D Series, not 2D DataFrames)
            df['RSI'] = RSIIndicator(close=df['Close']).rsi()
            df['EMA_10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
            df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
            df['MACD'] = MACD(close=df['Close']).macd()
            df['Returns'] = df['Close'].pct_change()

            # Clean data
            df.dropna(inplace=True)

            # Define target: 1 if next close > current close, else 0
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

            # Define features and target
            features = ['RSI', 'EMA_10', 'EMA_20', 'MACD', 'Returns']
            X = df[features]
            y = df['Target']

            # Train the model (skip last row)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X.iloc[:-1], y.iloc[:-1])

            # Predict using the latest feature row
            last_row = X.iloc[-1].values.reshape(1, -1)  # Make it 2D
            prediction = model.predict(last_row)[0]
            confidence = model.predict_proba(last_row)[0][prediction]

            # Show the result
            if prediction == 1:
                st.success(f"📈 Next 4H Candle may go **UP** (Confidence: {confidence:.1%})")
            else:
                st.error(f"📉 Next 4H Candle may go **DOWN** (Confidence: {confidence:.1%})")

        except Exception as e:
            st.error(f"❌ Error: {e}")
