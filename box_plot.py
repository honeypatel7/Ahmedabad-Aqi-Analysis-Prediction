import seaborn as sns
import matplotlib.pyplot as plt

# Plot a box plot for AQI values
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['AQI'])
plt.title("Box Plot of AQI Values")
plt.show()

Q1 = df['AQI'].quantile(0.25)  # 25th percentile
Q3 = df['AQI'].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['AQI'] < lower_bound) | (df['AQI'] > upper_bound)]
print(f"Number of Outliers: {len(outliers)}")
print(outliers)  # Print the outlier rows

from scipy.stats import zscore

df['Z_Score'] = zscore(df['AQI'])
outliers = df[(df['Z_Score'] > 3) | (df['Z_Score'] < -3)]

print(f"Number of Outliers: {len(outliers)}")
print(outliers[['AQI', 'Z_Score']])
