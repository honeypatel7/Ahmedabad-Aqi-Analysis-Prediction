import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from datetime import datetime 
# Load the dataset 
df = pd.read_excel("city_day_ahmd_final1.xlsx") 
# Display basic information 
print(df.info()) 
print(df.head()) 
# Convert 'Date' column to datetime format 
df['Date'] = pd.to_datetime(df['Date']) 

 
# Handle missing values by filling with median 
df.fillna(df.median(numeric_only=True), inplace=True) 
 
# Visualizing trends of major pollutants 
plt.figure(figsize=(10, 5)) 
for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3']: 
    plt.plot(df['Date'], df[col], label=col) 
plt.xlabel('Year') 
plt.ylabel('Pollutant Levels') 
plt.legend() 
plt.title('Pollutant Trends Over Time') 
plt.show() 
 
# Correlation heatmap 
plt.figure(figsize=(10, 6)) 
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f") 
plt.title("Correlation Matrix") 
plt.show() 
 
# Encode AQI_Bucket to numeric values 
# aqi_mapping = {'Good': 1, 'Satisfactory': 2, 'Moderate': 3, 'Poor': 4, 'Very Poor': 5, 'Severe': 6} 
# df['AQI_Bucket'] = df['AQI_Bucket'].map(aqi_mapping) 
 
# Selecting features and target 
X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'O3']] 
y = df['AQI'] 
 
# Splitting data (manually, as train_test_split is not used) 
split_index = int(0.8 * len(df)) 
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:] 
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:] 
 
# Implementing Linear Regression manually 
X_train_mean = X_train.mean() 
X_train_std = X_train.std() 
X_train_scaled = (X_train - X_train_mean) / X_train_std 
X_test_scaled = (X_test - X_train_mean) / X_train_std 
 
# Adding bias term 
X_train_scaled = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled] 
X_test_scaled = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled] 
 
# Compute weights using Normal Equation 
theta = np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ y_train 
 
# Predictions 
y_pred = X_test_scaled @ theta 
 
# Mean Absolute Error (Manual Calculation) 
mae = np.mean(np.abs(y_test - y_pred)) 
print(f"Mean Absolute Error: {mae:.2f}") 
 
#data preprocessing 
 
# Check for missing values 
missing_values = df.isnull().sum() 
 
# Display columns with missing values 
print("Missing values in each column:\n", missing_values[missing_values > 0]) 
 
# Optionally, show percentage of missing values 
missing_percentage = (df.isnull().sum() / len(df)) * 100 
print("\nPercentage of missing values:\n", missing_percentage[missing_percentage > 0]) 
 

 
#removing NH3 column because it is 100% empty 
df.drop(columns=['NH3'], inplace=True) 
df.dropna(inplace=True) 
df.to_csv("updated_data.csv", index=False) 
 
#calculating missing aqi values 
# AQI Breakpoints for Different Pollutants (CPCB Standard) 
aqi_breakpoints = { 
    "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), 
              (251, 500, 401, 500)], 
    "PM10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), 
             (431, 500, 401, 500)], 
    "NO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), 
            (401, 500, 401, 500)], 
    "NO": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), 
           (401, 500, 401, 500)],  # Approximate 
    "NOx": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), 
            (401, 500, 401, 500)],  # Approximate 
    "CO": [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300), (17.1, 34, 301, 400), 
           (34.1, 50, 401, 500)], 
    "SO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), 
            (1601, 2000, 401, 500)], 
    "O3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), 
           (749, 1000, 401, 500)], 
    "Benzene": [(0, 3, 0, 50), (3.1, 9, 51, 100), (9.1, 15, 101, 200), (15.1, 21, 201, 300), (21.1, 30, 301, 400), 
                (30.1, 50, 401, 500)], 
    "Toluene": [(0, 10, 0, 50), (10.1, 100, 51, 100), (100.1, 200, 101, 200), (200.1, 300, 201, 300), 
                (300.1, 400, 301, 400), (400.1, 500, 401, 500)], 
    "Xylene": [(0, 10, 0, 50), (10.1, 100, 51, 100), (100.1, 200, 101, 200), (200.1, 300, 201, 300), 
               (300.1, 400, 301, 400), (400.1, 500, 401, 500)] 
} 
 
 
def calculate_aqi(concentration, breakpoints): 
    """ 
    Compute AQI for a given pollutant concentration. 
    :param concentration: Pollutant concentration (µg/m³) 
    :param breakpoints: List of tuples [(C_low, C_high, I_low, I_high), ...] 
    :return: AQI value or None if out of range 
    """ 
    for C_low, C_high, I_low, I_high in breakpoints: 
        if C_low <= concentration <= C_high: 
            return ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low 
    return None  # If concentration is out of range 
 
 
def compute_final_aqi(row): 
    """ 
    Compute AQI for all pollutants in a row and return the maximum AQI. 
    """ 
    aqi_values = [] 
 
    for pollutant, breakpoints in aqi_breakpoints.items(): 
        if pollutant in row and not pd.isnull(row[pollutant]):  # Check if column exists and is not NaN 
            aqi = calculate_aqi(row[pollutant], breakpoints) 
            if aqi is not None: 
                aqi_values.append(aqi) 
 
    return max(aqi_values) if aqi_values else None  # Return max AQI if available 
 
 
# Apply the function to fill missing AQI values 
df["AQI"] = df.apply(compute_final_aqi, axis=1) 
 

# Save the updated dataset 
df.to_csv("updated_data.csv", index=False) 
# Show first few rows 
print(df.head()) 
# Correlation heatmap 
plt.figure(figsize=(10, 6)) 
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f") 
plt.title("Correlation Matrix") 
plt.show() 