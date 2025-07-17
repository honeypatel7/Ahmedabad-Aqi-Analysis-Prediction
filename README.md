# Ahmedabad-Aqi-Analysis-Prediction
Developed an LSTM-based model to analyze historical air pollution data and predict future AQI levels. The project includes data preprocessing, exploratory analysis, time-series training, and accurate forecasting of air quality trends to support proactive decision-making.


The main objective of this project is to analyze and predict the Air Quality Index (AQI) of Ahmedabad using 
AI Model. By studying historical AQI data and pollutant levels, we aim to identify patterns and build a 
reliable prediction model— using LSTM (Long Short-Term Memory)—to forecast future AQI values. These 
predictions can help in taking timely actions to reduce air pollution, protect public health, and support data
driven decision-making by government and environmental agencies.

Dataset Loading :
The dataset used for analyzing the Air Quality Index (AQI) of Ahmedabad and forecasting future trends was obtained 
from Kaggle, a popular platform for sharing datasets and machine learning resources. It encompasses AQI-related data 
recorded on a daily basis from January 2015 to July 2020, providing a comprehensive view of air quality patterns over 
a span of five and a half years. 

Data Pre-Processing:
Handling missing values, Outlier processing, ADF test for stationarity check:n time-series data analysis, checking for stationarity is an important step in time series analysis. For 
this we have implemented Augmented Dickey-Fuller(ADF) test.


Exploratory Data Analysis (EDA):

Time Series Analysis:

Model Implementation 
A. Long Short-Term Memory(LSTM) Model 


1)Data Preprocessing and Normalization 
The dataset was loaded from "updated_data.csv", and the AQI column was extracted for analysis. Since deep learning 
models perform better with scaled input values, the AQI data was normalized using Min-Max Scaling, mapping values 
to a 0 to 1 range: 
Xscaled = X – Xmin / Xmax - Xmin 
To structure the data for time-series forecasting, a sliding window approach was used. Sequences of 17 past AQI values 
were taken as input, with the next AQI value as the target. This method helps the LSTM model learn temporal patterns 
in the data. 
2) Model Architecture 
The LSTM model was designed for AQI prediction using sequential learning: 
• LSTM Layer 1: Configurable units (32–128) selected via hyperparameter tuning, with return_sequences=True 
for sequential feature extraction. 
• LSTM Layer 2: Configurable units (32–128) for deeper temporal feature learning. 
• Dense Layer: A single neuron for AQI prediction. 
• Activation & Compilation: The model was compiled using the Adam optimizer and Mean Squared Error 
(MSE) loss function, suitable for regression tasks. 
3) Training and Hyperparameter Optimization 
To improve the model accuracy we have perform hypertuning. The dataset was split into 80% training and 20% 
testing, maintaining sequence integrity. The AQI values were normalized using MinMax scaling to improve model 
convergence. 
Hyperparameter Tuning with Random Search 
To enhance model performance, a Random Search-based hyperparameter tuning approach was applied using Keras 
Tuner, optimizing: 
• LSTM units: Between 32 and 128 (step size 32). 
• Batch size: 32 
• Epochs: 50 (with Early Stopping to prevent overfitting). 
• Loss Function: MSE 
• Optimizer: Adam 

 
Metric Value 
MSE 0.0296 
MAE 0.1290 
R2 0.952 
RMSE 75.4692


