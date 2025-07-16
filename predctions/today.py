import numpy as np 
import pandas as pd 
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler 
 
# Load dataset 
df = pd.read_csv("updated_data.csv") 
 
# Remove COVID lockdown phase (Assuming lockdown starts from a specific date) 
lockdown_start_date = "2020-03-24"  # Adjust based on your dataset 
df = df[df['Date'] < lockdown_start_date]  # Keep only pre-lockdown data 
 
# Extract AQI values 
aqi_data = df['AQI'].values 
 
# Normalize AQI values using MinMaxScaler 
scaler = MinMaxScaler(feature_range=(0, 1)) 
aqi_scaled = scaler.fit_transform(aqi_data.reshape(-1, 1)) 
 
# Select the last 17 AQI values before lockdown 
latest_aqi = aqi_data[-17:]  # Last 17 non-lockdown AQI values 
 
# Normalize latest AQI values 
latest_aqi_scaled = scaler.transform(latest_aqi.reshape(-1, 1)) 
 
# Reshape for LSTM input (1 sample, 17 timesteps, 1 feature) 
latest_aqi_scaled = latest_aqi_scaled.reshape(1, 17, 1) 
 
# Load the trained LSTM model 
model = load_model("lstm_aqi_model.keras")  # Ensure model is saved before loading 
 
# Predict today's AQI 
predicted_aqi_scaled = model.predict(latest_aqi_scaled) 
 
# Convert back to actual AQI value 
predicted_aqi = scaler.inverse_transform(predicted_aqi_scaled) 
 
print(f"  Predicted AQI for today: {predicted_aqi[0][0]:.2f}")