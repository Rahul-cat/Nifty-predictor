import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔮 Nifty 5-Min Candle Predictor & Option Helper")
st.markdown("**Predicts CALL/PUT + Suggests Entry, Strike, Exit, SL, Target, Holding Time**")

# User inputs
ema_20 = st.number_input("📊 EMA 20", value=18000)
rsi = st.number_input("📈 RSI", value=55)
atr = st.number_input("🌊 ATR", value=15)

if st.button("🚀 Predict Now"):
    # Create DataFrame
    input_data = pd.DataFrame([[ema_20, rsi, atr]], columns=["EMA_20", "RSI", "ATR"])
    X_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    confidence = max(prob)

    # Suggested Option Type
    option_type = "CALL (BUY)" if prediction == 1 else "PUT (BUY)"
    st.markdown(f"### 🎯 Prediction: **{option_type}**")
    st.markdown(f"🧠 Model Confidence: **{confidence * 100:.2f}%**")

    # Nearest strike price logic
    nearest_strike = int(round(ema_20 / 50) * 50)
    st.markdown(f"🏹 Suggested Strike Price: **{nearest_strike} CE/PE**")

    # Entry time = now
    now = datetime.now()
    entry_time = now.strftime("%I:%M %p")

    # Exit time based on confidence
    if confidence > 0.8:
        hold_min = 20
    elif confidence > 0.6:
        hold_min = 15
    else:
        hold_min = 10
    exit_time = (now + timedelta(minutes=hold_min)).strftime("%I:%M %p")

    st.markdown(f"📌 Suggested Entry Time: **{entry_time}**")
    st.markdown(f"📤 Suggested Exit Time: **{exit_time}**")
    st.markdown(f"⏱️ Suggested Holding Time: **{hold_min} minutes**")

    # Premium Target & Stop-Loss (assume sample premium)
    current_premium = 100  # you can change this
    target = current_premium + atr * 2
    stop_loss = current_premium - atr * 1.5

    st.markdown(f"🎯 Target Premium: **₹{target:.2f}**")
    st.markdown(f"🛑 Stop Loss: **₹{stop_loss:.2f}**")

    # Profit simulation
    st.markdown("---")
    st.markdown("### 💰 Example Profit Simulation:")
    qty = 50  # 1 lot
    profit = (target - current_premium) * qty
    loss = (current_premium - stop_loss) * qty
    st.markdown(f"✅ Potential Profit: **₹{profit:.2f}**")
    st.markdown(f"❌ Potential Loss: **₹{loss:.2f}**")
