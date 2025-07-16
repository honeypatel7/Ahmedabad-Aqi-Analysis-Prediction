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
 
# Plot 1: Correlation heatmap 
plt.figure(figsize=(10, 6)) 
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f") 
plt.title("Correlation Matrix") 
plt.show()