
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
# Plot 4: AQI over time 
plt.figure(figsize=(12, 6)) 
plt.plot(df['Date'], df['AQI'], color='blue', marker='o', markersize=3, linestyle='-', label='AQI') 
 
plt.title('AQI Over Time in Ahmedabad') 
plt.xlabel('Date') 
plt.ylabel('AQI') 
plt.legend() 
plt.grid(True) 
plt.tight_layout() 
plt.show()