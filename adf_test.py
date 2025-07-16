import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Load the dataset
df = pd.read_csv("updated_data.csv")

# Extract the AQI column
aqi_series = df['AQI']

# Perform Augmented Dickey-Fuller Test
adf_result = adfuller(aqi_series)

# Extract values
adf_stat = adf_result[0]
p_value = adf_result[1]
crit_values = adf_result[4]

# Print results
print("ADF Test Results:")
print(f"1] ADF Statistic: {adf_stat:.4f}")
print(f"2] p-value: {p_value:.4f}")
print("3] Critical Values:")
for key, value in crit_values.items():
    print(f"   {key}%: {value:.4f}")
