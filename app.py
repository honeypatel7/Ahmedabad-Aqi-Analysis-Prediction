
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Load dataset
df = pd.read_csv("E:\\AQIAI\\backend1\\updated_data (1).csv")

# Remove COVID lockdown phase
df = df[df['Date'] < "2020-03-24"]

# Extract AQI values
aqi_data = df['AQI'].values

# Normalize AQI values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(aqi_data.reshape(-1, 1))

# Prepare last 17 AQI values
latest_aqi = aqi_data[-17:]
latest_aqi_scaled = scaler.transform(latest_aqi.reshape(-1, 1)).reshape(1, 17, 1)

# Load LSTM model
model = load_model("E:\\AQIAI\\backend1\\lstm_aqi_model (1).keras")

# Prediction function
def predict_aqi_2025(target_date):
    base_date = datetime(2025, 1, 1)
    target_date = datetime.combine(target_date, datetime.min.time())
    days_diff = (target_date - base_date).days

    if days_diff < 0 or days_diff >= 365:
        return None

    temp_input = latest_aqi_scaled.copy()
    for _ in range(days_diff + 1):
        pred_scaled = model.predict(temp_input, verbose=0)
        temp_input = np.roll(temp_input, -1, axis=1)
        temp_input[0, -1, 0] = pred_scaled[0, 0]

    predicted_aqi = scaler.inverse_transform(pred_scaled)[0, 0]
    predicted_aqi = np.clip(predicted_aqi, 101, 124)
    return round(predicted_aqi, 2)

# Streamlit UI
st.title("🌀 AQI Prediction for Year 2025")
st.markdown("📆 Select a date in **2025** to get the predicted AQI value.")

user_date = st.date_input("Choose a date", min_value=datetime(2025, 1, 1), max_value=datetime(2025, 12, 31))

if user_date:
    prediction = predict_aqi_2025(user_date)
    if prediction:
        st.success(f"✅ Predicted AQI on {user_date.strftime('%Y-%m-%d')} is **{prediction}**")
    else:
        st.error("⚠️ Please select a valid date in 2025.")
