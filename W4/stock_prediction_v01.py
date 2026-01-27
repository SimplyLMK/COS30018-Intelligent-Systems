# File: stock_prediction_v01.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

#------------------------------------------------------------------------------
# Load Data
#------------------------------------------------------------------------------
COMPANY = 'CBA.AX'

TRAIN_START = '2020-01-01'
TRAIN_END = '2023-08-01'

data = yf.download(COMPANY, TRAIN_START, TRAIN_END)

#------------------------------------------------------------------------------
# Prepare Data
#------------------------------------------------------------------------------
PRICE_VALUE = "Close"

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1))

PREDICTION_DAYS = 60

x_train = []
y_train = []

scaled_data = scaled_data[:, 0]

for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x-PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#------------------------------------------------------------------------------
# Build the Model
#------------------------------------------------------------------------------
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'

test_data = yf.download(COMPANY, TEST_START, TEST_END)

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#------------------------------------------------------------------------------
# Calculate Performance Metrics
#------------------------------------------------------------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

actual_flat = actual_prices.flatten()
predicted_flat = predicted_prices.flatten()

mae = mean_absolute_error(actual_flat, predicted_flat)
mse = mean_squared_error(actual_flat, predicted_flat)
rmse = np.sqrt(mse)
r2 = r2_score(actual_flat, predicted_flat)
mape = np.mean(np.abs((actual_flat - predicted_flat) / actual_flat)) * 100

print("\n" + "="*60)
print("PERFORMANCE METRICS - v0.1")
print("="*60)
print(f"Mean Absolute Error (MAE):      ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Squared Error (MSE):       {mse:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")
print(f"R² Score:                       {r2:.4f}")
print("="*60 + "\n")

#------------------------------------------------------------------------------
# Plot the test predictions
#------------------------------------------------------------------------------
plt.figure(figsize=(14, 7))
plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price", linewidth=2)
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price", linewidth=2)
plt.title(f"{COMPANY} Stock Price Prediction (v0.1)\nMAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}")
plt.xlabel("Time (Trading Days)")
plt.ylabel(f"{COMPANY} Share Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('v01_results.png')
print("Graph saved as 'v01_results.png'")
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------
real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"\nNext Day Prediction: ${prediction[0][0]:.2f}")
