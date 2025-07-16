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

# Plot 2: Plot the histogram 
plt.figure(figsize=(10, 6)) 
sns.histplot(df["AQI"], bins=30, kde=True, color="blue")  # kde=True adds a smooth density curve 
# Customize the plot 
plt.xlabel("AQI Value", fontsize=12) 
plt.ylabel("Frequency", fontsize=12) 
plt.title("AQI Distribution", fontsize=14) 
plt.grid(True) 
# Show the plot 
plt.show() 