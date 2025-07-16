import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense 
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error 
import numpy as np 
# Load dataset 
df = pd.read_csv("updated_data.csv") 
 
# Extract AQI values 
if 'AQI' not in df.columns: 
    raise KeyError("Column 'AQI' not found in the dataset. Check the CSV file.") 
 
aqi_data = df['AQI'].values 
 
# Normalize AQI values 
scaler = MinMaxScaler(feature_range=(0, 1)) 
aqi_scaled = scaler.fit_transform(aqi_data.reshape(-1, 1)) 
 
# Function to create sequences of 17 past AQI values as input and the next AQI value as output 
def create_sequences(data, seq_length=17): 
    X, y = [], [] 
    for i in range(len(data) - seq_length): 
        X.append(data[i:i+seq_length]) 
        y.append(data[i+seq_length]) 
    return np.array(X), np.array(y) 
 
# Create input-output sequences 
X, y = create_sequences(aqi_scaled, seq_length=17) 
 
# Split into training (80%) and testing (20%) sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False) 
 
# Reshape input for LSTM (samples, time steps, features) 
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)) 
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)) 
 
# Define LSTM Model 
model = Sequential([ 
    LSTM(50, return_sequences=True, input_shape=(17, 1)), 
    LSTM(50), 
 
 
    Dense(1) 
]) 
 
# Compile the model 
model.compile(optimizer='adam', loss='mse', metrics=['mae']) 
 
# Implement EarlyStopping to monitor validation loss and stop training when it stops improving 
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) 
 
# Train the model 
history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                    validation_data=(X_test, y_test), callbacks=[early_stop]) 
 
# Evaluate the model 
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2) 
print(f"LSTM Model - MSE: {test_loss:.2f}, MAE: {test_mae:.2f}") 
 
# Predict on test data 
y_pred = model.predict(X_test) 
 
# Reverse normalization 
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)) 
y_pred_actual = scaler.inverse_transform(y_pred) 
 
 
# Calculate R2 Score 
r2 = r2_score(y_test_actual, y_pred_actual) 
 
# Calculate RMSE (Root Mean Squared Error) 
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual)) 
 
print(f"R2 Score: {r2:.4f}") 
print(f"RMSE: {rmse:.4f}") 
 
# Plot actual vs predicted AQI values 
plt.figure(figsize=(10,5)) 
plt.plot(y_test_actual, label="Actual AQI") 
plt.plot(y_pred_actual, label="Predicted AQI", linestyle='dashed') 
plt.xlabel("Time Steps") 
plt.ylabel("AQI") 
plt.legend() 
plt.title("Actual vs Predicted AQI using LSTM") 
plt.show()