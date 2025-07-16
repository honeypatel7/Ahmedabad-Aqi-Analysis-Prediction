
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from datetime import datetime 
 
# Load the dataset 
df = pd.read_csv("updated_data.csv") 
 
# Display basic information 
print(df.info()) 
print(df.head()) 
 
# Convert 'Date' column to datetime format 
df['Date'] = pd.to_datetime(df['Date']) 
 
# Handle missing values by filling with median 
df.fillna(df.median(numeric_only=True), inplace=True) 
# Plot 3: Identify AQI Trends & Seasonal Patterns 
df['Month'] = pd.to_datetime(df['Date']).dt.month 
monthly_avg = df.groupby('Month')['AQI'].mean() 
plt.figure(figsize=(10,5)) 
plt.plot(monthly_avg, marker='o', linestyle='-', color='r') 
plt.xlabel("Month") 
plt.ylabel("Average AQI") 
plt.title("Seasonal Variation in AQI - Ahmedabad") 
plt.grid() 
plt.show()