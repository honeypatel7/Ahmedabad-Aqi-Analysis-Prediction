#2030 
import numpy as np 
import pandas as pd 
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler 
from datetime import datetime, timedelta 
import random 
 
# Load dataset 
df = pd.read_csv("updated_data.csv") 
 
# Remove COVID lockdown phase 
lockdown_start_date = "2020-03-24" 
df = df[df['Date'] < lockdown_start_date] 
 
# Extract AQI values 
aqi_data = df['AQI'].values 
 
# Normalize AQI values 
scaler = MinMaxScaler(feature_range=(0, 1)) 
aqi_scaled = scaler.fit_transform(aqi_data.reshape(-1, 1)) 
 
# Select last 17 AQI values before lockdown 
latest_aqi = aqi_data[-17:] 
 
# Normalize latest AQI values 
latest_aqi_scaled = scaler.transform(latest_aqi.reshape(-1, 1)) 
latest_aqi_scaled = latest_aqi_scaled.reshape(1, 17, 1) 
 
# Load trained LSTM model 
model = load_model("lstm_aqi_model.keras") 
 
# Define target dates 
dates_2025 = [datetime(2025, 3, i) for i in range(15, 20)] 
dates_2030 = [datetime(2030, random.randint(1, 12), random.randint(1, 28)) for _ in range(5)] 
all_dates = dates_2025 + dates_2030 
 
# Function to predict AQI step-by-step 
def predict_aqi_future(days_to_predict): 
    temp_input = latest_aqi_scaled.copy() 
    predictions = [] 
 
    for _ in range(days_to_predict): 
        predicted_aqi_scaled = model.predict(temp_input, verbose=0)  # Fast execution 
        predicted_aqi = scaler.inverse_transform(predicted_aqi_scaled)[0, 0] 

 
predictions.append(predicted_aqi) 
# Shift input window and append new prediction 
temp_input = np.roll(temp_input, -1, axis=1) 
temp_input[0, -1, 0] = predicted_aqi_scaled[0, 0]  # Maintain shape 
    return predictions 
# Predict AQI for selected dates 
aqi_predictions = predict_aqi_future(len(all_dates)) 
# Print results 
print("\n  Predicted AQI for Selected Dates:") 
for date, aqi in zip(all_dates, aqi_predictions): 
print(f"{date.strftime('%Y-%m-%d')}: {aqi:.2f}")