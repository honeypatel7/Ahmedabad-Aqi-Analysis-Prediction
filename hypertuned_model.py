
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error 
 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense 
from tensorflow.keras.callbacks import EarlyStopping 
 
from kerastuner.tuners import Hyperband 
from kerastuner.engine.hyperparameters import HyperParameters 
 
# Load dataset 
df = pd.read_csv("updated_data.csv") 
 
if 'AQI' not in df.columns: 
    raise KeyError("Column 'AQI' not found in the dataset.") 
 
aqi_data = df['AQI'].values 
 
# Normalize 
scaler = MinMaxScaler(feature_range=(0, 1)) 
aqi_scaled = scaler.fit_transform(aqi_data.reshape(-1, 1)) 
 
# Create sequences 
def create_sequences(data, seq_length=17): 
    X, y = [], [] 
    for i in range(len(data) - seq_length): 
        X.append(data[i:i+seq_length]) 
        y.append(data[i+seq_length]) 
    return np.array(X), np.array(y) 
 
X, y = create_sequences(aqi_scaled, seq_length=17) 
 
# Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) 
 
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)) 
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)) 
 
# Define the model builder 
def build_model(hp): 
    model = Sequential() 
    model.add(LSTM(units=hp.Int('units1', min_value=32, max_value=128, step=32), 
                   return_sequences=True, input_shape=(17, 1))) 

     
    if hp.Boolean('add_second_lstm'): 
        model.add(LSTM(units=hp.Int('units2', min_value=32, max_value=128, step=32))) 
    else: 
        model.add(LSTM(units=hp.Int('units1_last', min_value=32, max_value=128, step=32))) 
     
    model.add(Dense(1)) 
     
    model.compile(optimizer=tf.keras.optimizers.Adam( 
                      hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG')), 
                  loss='mse', 
                  metrics=['mae']) 
     
    return model 
 
# Set up the tuner 
tuner = Hyperband( 
    build_model, 
    objective='val_loss', 
    max_epochs=30, 
    factor=3, 
    directory='lstm_tuning', 
    project_name='AQI_LSTM', 
    overwrite=True 
) 
 
# Early stopping 
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) 
 
# Search best model 
tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stop]) 
 
# Retrieve the best model 
best_model = tuner.get_best_models(num_models=1)[0] 
 
# Evaluate 
test_loss, test_mae = best_model.evaluate(X_test, y_test) 
print(f"Tuned LSTM - MSE: {test_loss:.4f}, MAE: {test_mae:.4f}") 
 
# Predict 
y_pred = best_model.predict(X_test) 
 
# Reverse normalization 
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)) 
y_pred_actual = scaler.inverse_transform(y_pred) 
 
# Metrics 
r2 = r2_score(y_test_actual, y_pred_actual) 
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual)) 
 
print(f"R2 Score: {r2:.4f}") 
print(f"RMSE: {rmse:.4f}") 
 
# Plot 
plt.figure(figsize=(10, 5)) 
plt.plot(y_test_actual, label="Actual AQI") 
plt.plot(y_pred_actual, label="Predicted AQI", linestyle='dashed') 
plt.xlabel("Time Steps") 

plt.ylabel("AQI") 
plt.legend() 
plt.title("Tuned LSTM: Actual vs Predicted AQI") 
plt.show() 