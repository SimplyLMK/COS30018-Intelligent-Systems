from stock_prediction_P1 import create_model, load_data
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parameters_P1 import *

# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")

print("="*60)
print("Loading data...")
print("="*60)

# load the data
data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, 
                shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, 
                feature_columns=FEATURE_COLUMNS)

# save the dataframe
data["df"].to_csv(ticker_data_filename)

print(f"Training samples: {len(data['X_train'])}")
print(f"Testing samples: {len(data['X_test'])}")

print("\n" + "="*60)
print("Building model...")
print("="*60)

# construct the model
model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

model.summary()

print("\n" + "="*60)
print("Training model...")
print("="*60)

# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".weights.h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

# train the model and save the weights whenever we see 
# a new optimal model using ModelCheckpoint
history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)

print("\n" + "="*60)
print("Making predictions...")
print("="*60)

# Make predictions (these are in scaled/normalized form)
y_pred_scaled = model.predict(data["X_test"])

# CRITICAL: Inverse transform predictions back to actual dollar values
if SCALE:
    # Get the scaler for 'adjclose' column
    adjclose_scaler = data["column_scaler"]["adjclose"]
    # Inverse transform both predictions and actual values
    y_pred = adjclose_scaler.inverse_transform(y_pred_scaled)
    y_test_actual = adjclose_scaler.inverse_transform(data["y_test"].reshape(-1, 1))
else:
    y_pred = y_pred_scaled
    y_test_actual = data["y_test"].reshape(-1, 1)

# Flatten for metric calculation
y_pred_flat = y_pred.flatten()
y_test_flat = y_test_actual.flatten()

# Calculate metrics on ACTUAL dollar values
mae = mean_absolute_error(y_test_flat, y_pred_flat)
mse = mean_squared_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_flat, y_pred_flat)
mape = np.mean(np.abs((y_test_flat - y_pred_flat) / y_test_flat)) * 100

print("\n" + "="*60)
print("PERFORMANCE METRICS - P1 (Actual Dollar Values)")
print("="*60)
print(f"Mean Absolute Error (MAE):      ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Squared Error (MSE):       {mse:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")
print(f"R² Score:                       {r2:.4f}")
print("="*60 + "\n")

# Plot results with ACTUAL dollar values
plt.figure(figsize=(14, 7))
plt.plot(y_test_flat, color="black", label=f"Actual {ticker} Price", linewidth=2)
plt.plot(y_pred_flat, color="blue", label=f"Predicted {ticker} Price", linewidth=2)
plt.title(f"{ticker} Stock Price Prediction (P1)\nMAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}")
plt.xlabel("Time (Trading Days)")
plt.ylabel(f"{ticker} Share Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('P1_results.png')
print("Graph saved as 'P1_results.png'")
plt.show()

print("\nTraining complete!")
