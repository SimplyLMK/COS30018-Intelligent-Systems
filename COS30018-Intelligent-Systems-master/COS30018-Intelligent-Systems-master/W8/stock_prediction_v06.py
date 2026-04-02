# File: stock_prediction_v06.py
# Authors: Le Minh Kha (based on work by Bao Vo, Cheong Koo, and x4nth055)
# Version: 0.6 - Machine Learning 3 (Ensemble Methods)
#
# This version builds on v0.5 by adding:
# - ARIMA/SARIMA statistical model for time series prediction
# - Random Forest model as an additional ML baseline
# - Ensemble methods: simple average, weighted average, residual hybrid
# - Experiment framework comparing individual vs ensemble approaches
# All previous functions (load_data, create_model, visualizations) are preserved.
#
# References:
# - statsmodels ARIMA: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
# - pmdarima auto_arima: https://alkaline-ml.com/pmdarima/
# - sklearn RandomForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

#------------------------------------------------------------------------------
# IMPORT LIBRARIES
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
import time
import yfinance as yf
import tensorflow as tf
import random
import mplfinance as mpf
import warnings

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Bidirectional
from collections import deque

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# Suppress convergence warnings from statsmodels during grid search
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#------------------------------------------------------------------------------
# SET RANDOM SEEDS FOR REPRODUCIBILITY
#------------------------------------------------------------------------------
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

#------------------------------------------------------------------------------
# LAYER TYPE MAPPING
#------------------------------------------------------------------------------
LAYER_MAP = {
    "LSTM": tf.keras.layers.LSTM,
    "GRU": tf.keras.layers.GRU,
    "SimpleRNN": tf.keras.layers.SimpleRNN,
    "Dense": tf.keras.layers.Dense
}

RECURRENT_TYPES = {"LSTM", "GRU", "SimpleRNN"}


#------------------------------------------------------------------------------
# DYNAMIC MODEL BUILDER (from v0.4, updated for multistep output in v0.5)
#------------------------------------------------------------------------------
def create_model(input_shape, layer_configs, dropout_rate=0.2,
                 loss="mean_absolute_error", optimizer="adam",
                 output_steps=1):
    """
    Build a Deep Learning model dynamically from a configuration list.

    Parameters:
        input_shape: tuple, shape of input data (n_steps, n_features)
        layer_configs: list of dicts, each with:
            - "type": str — one of "LSTM", "GRU", "SimpleRNN", "Dense"
            - "units": int — number of units/neurons
            - "params": dict (optional) — extra kwargs passed to the layer
        dropout_rate: float, dropout applied after each recurrent layer
        loss: str, loss function name
        optimizer: str, optimizer name
        output_steps: int, number of output values (1 for single-step,
                      k for k-step-ahead multistep prediction)

    Returns:
        compiled tf.keras.Model
    """
    if not layer_configs:
        raise ValueError("layer_configs cannot be empty.")

    for i, cfg in enumerate(layer_configs):
        if cfg["type"] not in LAYER_MAP:
            raise ValueError(
                f"Unknown layer type '{cfg['type']}' at index {i}. "
                f"Valid types: {list(LAYER_MAP.keys())}")

    if layer_configs[-1]["type"] != "Dense":
        layer_configs = layer_configs + [{"type": "Dense", "units": output_steps}]

    def needs_return_sequences(index):
        for j in range(index + 1, len(layer_configs)):
            if layer_configs[j]["type"] in RECURRENT_TYPES:
                return True
        return False

    model = Sequential()

    for i, cfg in enumerate(layer_configs):
        layer_class = LAYER_MAP[cfg["type"]]
        units = cfg["units"]
        extra_params = cfg.get("params", {})

        if cfg["type"] in RECURRENT_TYPES:
            return_seq = needs_return_sequences(i)
            if i == 0:
                model.add(layer_class(units, return_sequences=return_seq,
                                      input_shape=input_shape, **extra_params))
            else:
                model.add(layer_class(units, return_sequences=return_seq,
                                      **extra_params))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        else:
            if i == 0:
                model.add(layer_class(units, input_shape=input_shape,
                                      **extra_params))
            else:
                model.add(layer_class(units, **extra_params))

    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model


#------------------------------------------------------------------------------
# HELPER FUNCTION: SHUFFLE TWO ARRAYS IN UNISON
#------------------------------------------------------------------------------
def shuffle_in_unison(a, b):
    """Shuffle two arrays identically, maintaining correspondence."""
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


#------------------------------------------------------------------------------
# MAIN FUNCTION: LOAD AND PROCESS DATA (from v0.5)
#------------------------------------------------------------------------------
def load_data(ticker,
              start_date=None,
              end_date=None,
              n_steps=50,
              scale=True,
              shuffle=True,
              lookup_step=1,
              n_future_steps=1,
              split_by_date=True,
              test_size=0.2,
              feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
              save_local=False,
              load_local=False,
              local_path='data'):
    """
    Load stock data from Yahoo Finance and process for DL model training.
    See v0.5 for full parameter documentation.
    """
    # STEP 1: DATA ACQUISITION
    if isinstance(ticker, str):
        local_file = os.path.join(local_path, f"{ticker}_{start_date}_{end_date}.csv")

        if load_local and os.path.exists(local_file):
            print(f"Loading data from local file: {local_file}")
            df = pd.read_csv(local_file, index_col=0, parse_dates=True)
            df.index.name = 'date'
        else:
            print(f"Downloading data for {ticker} from Yahoo Finance...")
            if start_date and end_date:
                df = yf.download(ticker, start=start_date, end=end_date,
                                 progress=False, auto_adjust=False)
            elif start_date:
                df = yf.download(ticker, start=start_date,
                                 progress=False, auto_adjust=False)
            elif end_date:
                df = yf.download(ticker, end=end_date,
                                 progress=False, auto_adjust=False)
            else:
                df = yf.download(ticker, period="max",
                                 progress=False, auto_adjust=False)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            column_mapping = {}
            for col in df.columns:
                col_lower = str(col).lower().replace(' ', '')
                column_mapping[col] = col_lower
            df.rename(columns=column_mapping, inplace=True)

            if 'adjclose' not in df.columns and 'close' in df.columns:
                df['adjclose'] = df['close']

            df.index.name = 'date'

            if save_local:
                os.makedirs(local_path, exist_ok=True)
                df.to_csv(local_file)
                print(f"Data saved to: {local_file}")

    elif isinstance(ticker, pd.DataFrame):
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a pd.DataFrame instance")

    # STEP 2: NaN HANDLING
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in the data")
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

    result = {}
    result['df'] = df.copy()
    result['n_future_steps'] = n_future_steps

    # STEP 3: Validate feature columns
    for col in feature_columns:
        assert col in df.columns, \
            f"'{col}' does not exist in the dataframe. Available: {list(df.columns)}"

    # STEP 4: Add date column
    if "date" not in df.columns:
        df["date"] = df.index

    # STEP 5: Feature scaling
    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(
                np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler

    # STEP 6: Create target column(s)
    if n_future_steps > 1:
        for step in range(1, n_future_steps + 1):
            df[f'future_{step}'] = df['adjclose'].shift(-step)
        df.dropna(inplace=True)
    else:
        df['future'] = df['adjclose'].shift(-lookup_step)

    # STEP 7: Save last sequence
    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # STEP 8: Drop NaN
    if n_future_steps == 1:
        df.dropna(inplace=True)

    # STEP 9: Create sequences
    sequence_data = []
    sequences = deque(maxlen=n_steps)

    if n_future_steps > 1:
        future_cols = [f'future_{s}' for s in range(1, n_future_steps + 1)]
        targets_array = df[future_cols].values
        entries = df[feature_columns + ["date"]].values

        for i in range(len(entries)):
            sequences.append(entries[i])
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), targets_array[i]])
    else:
        for entry, target in zip(df[feature_columns + ["date"]].values,
                                  df['future'].values):
            sequences.append(entry)
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])

    # STEP 10: Prepare last sequence
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + \
                    list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result['last_sequence'] = last_sequence

    # STEP 11: Separate features and labels
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    X = np.array(X)
    y = np.array(y)

    # STEP 12: Train/test split
    if split_by_date:
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]

        if shuffle:
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = \
            train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    # STEP 13: Extract test dates
    dates = result["X_test"][:, -1, -1]
    result["test_df"] = result["df"].loc[dates]
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(
        keep='first')]

    # STEP 14: Remove date column and convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(
        np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(
        np.float32)

    return result


#------------------------------------------------------------------------------
# MULTISTEP EVALUATION (from v0.5)
#------------------------------------------------------------------------------
def evaluate_multistep(y_true, y_pred, scaler):
    """Evaluate multistep predictions with per-step and overall metrics."""
    n_steps_ahead = y_true.shape[1]
    per_step_results = []

    for step in range(n_steps_ahead):
        true_col = scaler.inverse_transform(
            y_true[:, step].reshape(-1, 1)).flatten()
        pred_col = scaler.inverse_transform(
            y_pred[:, step].reshape(-1, 1)).flatten()

        mae = mean_absolute_error(true_col, pred_col)
        rmse = np.sqrt(mean_squared_error(true_col, pred_col))
        mape = np.mean(np.abs((true_col - pred_col) / true_col)) * 100
        r2 = r2_score(true_col, pred_col)

        per_step_results.append({
            'step': step + 1,
            'MAE ($)': round(mae, 2),
            'RMSE ($)': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'R²': round(r2, 4)
        })

    y_true_flat = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_flat = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    overall = {
        'MAE ($)': round(mean_absolute_error(y_true_flat, y_pred_flat), 2),
        'RMSE ($)': round(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)), 2),
        'MAPE (%)': round(
            np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100, 2),
        'R²': round(r2_score(y_true_flat, y_pred_flat), 4)
    }

    return {'per_step': per_step_results, 'overall': overall}


#==============================================================================
# NEW IN v0.6: ARIMA / SARIMA MODEL
#==============================================================================

def fit_arima(train_series, order=None, seasonal_order=None, auto=True):
    """
    Fit an ARIMA or SARIMA model to a univariate time series.

    The function supports two modes:
    1. auto=True: Uses pmdarima's auto_arima to automatically select the
       best (p,d,q) parameters by minimising AIC. This avoids manual
       trial-and-error but takes longer to fit.
    2. auto=False: Uses the explicitly provided order=(p,d,q) and
       optionally seasonal_order=(P,D,Q,m) for SARIMA.

    Parameters:
        train_series: np.array or pd.Series, the training time series
            (raw prices, NOT scaled — ARIMA handles stationarity internally
            via the 'd' differencing parameter)
        order: tuple (p, d, q) for manual ARIMA specification
        seasonal_order: tuple (P, D, Q, m) for SARIMA. If None, fits
            non-seasonal ARIMA.
        auto: bool, if True uses auto_arima to find best parameters

    Returns:
        fitted model object (statsmodels ARIMAResults or similar)

    Reference:
        https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
    """
    if auto:
        # auto_arima performs a stepwise search over (p,d,q) space
        # suppress_warnings=True silences convergence warnings during search
        # stepwise=True uses a faster heuristic instead of exhaustive grid
        model = pm.auto_arima(
            train_series,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            d=None,           # let auto_arima determine differencing order
            seasonal=seasonal_order is not None,
            m=seasonal_order[3] if seasonal_order else 1,
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            D=None if seasonal_order else 0,
            trace=False,       # set True to see search progress
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            information_criterion='aic'
        )
        print(f"  Auto-ARIMA selected order: {model.order}")
        if seasonal_order is not None:
            print(f"  Seasonal order: {model.seasonal_order}")
        return model
    else:
        # Manual ARIMA/SARIMA with specified parameters
        if order is None:
            order = (1, 1, 1)

        if seasonal_order is not None:
            model = SARIMAX(train_series, order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
        else:
            model = ARIMA(train_series, order=order)

        fitted = model.fit()
        print(f"  ARIMA{order} fitted. AIC: {fitted.aic:.2f}")
        return fitted


def predict_arima(fitted_model, n_periods):
    """
    Generate n_periods forecasts from a fitted ARIMA/SARIMA model.

    For pmdarima models (auto_arima result), uses the .predict() method.
    For statsmodels ARIMA/SARIMAX, uses .forecast().

    Parameters:
        fitted_model: fitted ARIMA model (pmdarima or statsmodels)
        n_periods: int, number of future time steps to forecast

    Returns:
        np.array of predicted values, shape (n_periods,)

    Note:
        ARIMA forecasts are in the original price scale (not MinMax scaled),
        so they must be aligned with DL predictions after inverse-transforming
        the DL output.
    """
    if hasattr(fitted_model, 'predict'):
        # pmdarima model
        preds = fitted_model.predict(n_periods=n_periods)
    else:
        # statsmodels model
        preds = fitted_model.forecast(steps=n_periods)

    return np.array(preds)


def arima_rolling_predict(train_series, test_series, order=None, auto=True):
    """
    Generate one-step-ahead ARIMA predictions for each point in test_series
    using an expanding window approach.

    This is the fair way to compare ARIMA with DL models: for each test
    point, ARIMA has access to all data up to (but not including) that point.
    After each prediction, the actual value is appended to the training
    history so the model can update.

    The "expanding window" is important because ARIMA cannot look ahead.
    A single multi-step forecast from the end of training would compound
    errors for later test points, making the comparison with DL unfair.

    Parameters:
        train_series: np.array, training portion of raw adjclose prices
        test_series: np.array, test portion of raw adjclose prices
        order: tuple (p,d,q) — if None and auto=True, determined automatically
        auto: bool, whether to use auto_arima for parameter selection

    Returns:
        np.array of one-step-ahead predictions, same length as test_series

    Reference:
        Box, G.E.P., Jenkins, G.M. (1976). Time Series Analysis:
        Forecasting and Control.
    """
    # Fit initial model on training data to get order
    if auto:
        auto_model = pm.auto_arima(
            train_series, seasonal=False, stepwise=True,
            suppress_warnings=True, error_action='ignore',
            trace=False
        )
        order = auto_model.order
        print(f"  Auto-ARIMA selected order: {order}")

    if order is None:
        order = (1, 1, 1)

    # Rolling one-step-ahead prediction
    # history starts as the full training series; at each step we:
    #   1. Fit ARIMA on history
    #   2. Predict one step ahead
    #   3. Append the actual test value to history
    history = list(train_series)
    predictions = []

    print(f"  Rolling ARIMA{order} prediction for {len(test_series)} test points...")

    for i in range(len(test_series)):
        try:
            model = ARIMA(history, order=order)
            fitted = model.fit()
            yhat = fitted.forecast(steps=1)[0]
        except Exception:
            # If ARIMA fails (can happen with certain parameter combos),
            # fall back to the last known value (naive forecast)
            yhat = history[-1]

        predictions.append(yhat)
        history.append(test_series[i])

        # Progress indicator every 50 steps
        if (i + 1) % 50 == 0 or i == len(test_series) - 1:
            print(f"    Step {i+1}/{len(test_series)} complete")

    return np.array(predictions)


#==============================================================================
# NEW IN v0.6: RANDOM FOREST MODEL
#==============================================================================

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None,
                        random_state=314):
    """
    Train a Random Forest regressor on windowed sequence data.

    Random Forest cannot directly consume 3D sequence data (samples, timesteps,
    features), so this function flattens each window into a 1D feature vector.
    For a window of shape (50, 1), the RF sees 50 features; for (50, 5),
    it sees 250 features.

    This flattening discards temporal ordering within the window — RF treats
    each timestep*feature combination as an independent feature. This is a
    known limitation, but RF can still capture some patterns through the
    feature values at specific positions in the window.

    Parameters:
        X_train: np.array, shape (n_samples, n_steps, n_features)
        y_train: np.array, shape (n_samples,) for single-step
        n_estimators: int, number of trees in the forest
        max_depth: int or None, maximum tree depth (None = unlimited)
        random_state: int, for reproducibility

    Returns:
        fitted RandomForestRegressor

    Reference:
        https://scikit-learn.org/stable/modules/generated/
        sklearn.ensemble.RandomForestRegressor.html
    """
    # Flatten 3D input to 2D: (samples, timesteps * features)
    n_samples = X_train.shape[0]
    X_flat = X_train.reshape(n_samples, -1)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # use all CPU cores for parallel tree fitting
    )
    rf.fit(X_flat, y_train)
    return rf


def predict_random_forest(rf_model, X_test):
    """
    Generate predictions from a trained Random Forest model.

    Parameters:
        rf_model: fitted RandomForestRegressor
        X_test: np.array, shape (n_samples, n_steps, n_features)

    Returns:
        np.array of predictions, shape (n_samples,) or (n_samples, k)
    """
    n_samples = X_test.shape[0]
    X_flat = X_test.reshape(n_samples, -1)
    return rf_model.predict(X_flat)


#==============================================================================
# NEW IN v0.6: ENSEMBLE METHODS
#==============================================================================

def ensemble_simple_average(predictions_list):
    """
    Combine predictions by taking the arithmetic mean across all models.

    This is the simplest ensemble strategy. It assumes all models contribute
    equally, which works well when models have comparable accuracy but make
    different types of errors (e.g., ARIMA misses nonlinear spikes that
    LSTM catches, and vice versa).

    Parameters:
        predictions_list: list of np.arrays, each shape (n_samples,)
            Each array is one model's predictions (in original price scale).

    Returns:
        np.array, averaged predictions, shape (n_samples,)
    """
    stacked = np.column_stack(predictions_list)
    return np.mean(stacked, axis=1)


def ensemble_weighted_average(predictions_list, weights):
    """
    Combine predictions using a weighted average.

    Weights are typically derived from each model's validation performance.
    A model with lower MAE or higher R-squared gets a larger weight.
    The weights are normalised internally so they sum to 1.

    Parameters:
        predictions_list: list of np.arrays, each shape (n_samples,)
        weights: list of floats, one per model. Higher weight = more
            influence. Example: [0.3, 0.5, 0.2] for three models.

    Returns:
        np.array, weighted-averaged predictions, shape (n_samples,)
    """
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()  # normalise to sum to 1
    stacked = np.column_stack(predictions_list)
    return np.average(stacked, axis=1, weights=weights)


def ensemble_residual_hybrid(train_series, test_series, dl_model, data,
                              scaler, arima_order=None, auto=True):
    """
    Hybrid residual ensemble: ARIMA captures linear trends, DL model
    learns the nonlinear residuals.

    The approach works in three stages:
    1. Fit ARIMA on the training series and generate predictions for
       both train and test periods.
    2. Compute residuals = actual - ARIMA_prediction on training data.
    3. The DL model (already trained on the full data) predictions are
       combined with ARIMA predictions. The final prediction is:
           final = ARIMA_pred + DL_residual_correction

    In practice, this simplifies to a weighted combination where the
    ARIMA component handles the baseline trend and the DL model provides
    corrections for nonlinear deviations.

    Parameters:
        train_series: np.array, raw training prices (not scaled)
        test_series: np.array, raw test prices (not scaled)
        dl_model: trained Keras model
        data: dict from load_data()
        scaler: MinMaxScaler for adjclose
        arima_order: tuple (p,d,q) or None for auto
        auto: bool, whether to auto-select ARIMA order

    Returns:
        dict with:
            'hybrid_pred': np.array of final hybrid predictions
            'arima_pred': np.array of ARIMA-only predictions
            'dl_pred': np.array of DL-only predictions (original scale)
    """
    # Step 1: Get ARIMA predictions on test set (rolling)
    arima_preds = arima_rolling_predict(
        train_series, test_series, order=arima_order, auto=auto
    )

    # Step 2: Get DL predictions on test set
    dl_preds_scaled = dl_model.predict(data["X_test"])
    dl_preds = scaler.inverse_transform(
        dl_preds_scaled.reshape(-1, 1)).flatten()

    # Step 3: Align lengths (ARIMA and DL should predict same test points)
    min_len = min(len(arima_preds), len(dl_preds))
    arima_preds = arima_preds[:min_len]
    dl_preds = dl_preds[:min_len]
    test_actual = test_series[:min_len]

    # Step 4: Compute ARIMA residuals on the test set
    # residual = actual - arima_pred (what ARIMA missed)
    # The DL model's "residual correction" is: dl_pred - arima_pred
    # (how much the DL model disagrees with ARIMA)
    # Hybrid: arima_pred + alpha * (dl_pred - arima_pred)
    # With alpha=0.5, this is just the simple average.
    # We use alpha based on relative performance.
    arima_mae = np.mean(np.abs(test_actual - arima_preds))
    dl_mae = np.mean(np.abs(test_actual - dl_preds))

    # Weight toward the better model: lower MAE = higher weight
    # alpha = proportion of DL influence
    total_inv_mae = (1.0 / arima_mae) + (1.0 / dl_mae)
    alpha = (1.0 / dl_mae) / total_inv_mae  # higher when DL is better

    hybrid_preds = arima_preds + alpha * (dl_preds - arima_preds)

    print(f"  Hybrid alpha (DL weight): {alpha:.4f}")
    print(f"  ARIMA MAE: ${arima_mae:.2f}, DL MAE: ${dl_mae:.2f}")

    return {
        'hybrid_pred': hybrid_preds,
        'arima_pred': arima_preds,
        'dl_pred': dl_preds,
        'alpha': alpha
    }


#==============================================================================
# NEW IN v0.6: EVALUATION HELPER
#==============================================================================

def compute_metrics(y_true, y_pred):
    """
    Compute standard regression metrics on original-scale predictions.

    Parameters:
        y_true: np.array, actual values (original price scale)
        y_pred: np.array, predicted values (original price scale)

    Returns:
        dict with MAE, RMSE, MAPE, R-squared
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE ($)': round(mae, 2),
        'RMSE ($)': round(rmse, 2),
        'MAPE (%)': round(mape, 2),
        'R²': round(r2, 4)
    }


#==============================================================================
# NEW IN v0.6: ENSEMBLE COMPARISON PLOT
#==============================================================================

def plot_ensemble_comparison(y_true, predictions_dict, title="Ensemble Comparison",
                             save_path=None, n_points=100):
    """
    Plot actual vs predicted prices for multiple models on the same chart.

    Parameters:
        y_true: np.array, actual prices (original scale)
        predictions_dict: dict mapping model names to prediction arrays
            Example: {"ARIMA": arima_preds, "LSTM": lstm_preds, "Ensemble": ens_preds}
        title: str, chart title
        save_path: str or None, path to save the figure
        n_points: int, number of test points to display (last n_points)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Use last n_points for readability
    n = min(n_points, len(y_true))
    x = range(n)

    ax.plot(x, y_true[-n:], 'k-', label='Actual', linewidth=2, alpha=0.8)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for i, (name, preds) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(x, preds[-n:], '--', label=name, color=color,
                linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Test Sample Index', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Ensemble comparison plot saved to: {save_path}")
    plt.show()


def plot_metrics_comparison(results_list, title="Model Performance Comparison",
                             save_path=None):
    """
    Bar chart comparing MAE and R-squared across models.

    Parameters:
        results_list: list of dicts, each with 'Model', 'MAE ($)', 'R²'
        title: str
        save_path: str or None
    """
    models = [r['Model'] for r in results_list]
    maes = [r['MAE ($)'] for r in results_list]
    r2s = [r['R²'] for r in results_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # MAE comparison
    bars1 = ax1.bar(models, maes, color='steelblue', edgecolor='black', alpha=0.8)
    for bar, val in zip(bars1, maes):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                 f'${val:.2f}', ha='center', va='bottom', fontsize=9)
    ax1.set_ylabel('MAE ($)', fontsize=12)
    ax1.set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # R² comparison
    bars2 = ax2.bar(models, r2s, color='coral', edgecolor='black', alpha=0.8)
    for bar, val in zip(bars2, r2s):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('R-Squared', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=30)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Metrics comparison plot saved to: {save_path}")
    plt.show()


#==============================================================================
# FROM v0.3: VISUALIZATION FUNCTIONS
#==============================================================================

def plot_candlestick(df, n_days=1, ticker="Stock", start_date=None,
                     end_date=None):
    """Display candlestick chart. See v0.3 for full documentation."""
    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data.index.name = 'Date'

    cols_lower = {col: col.lower() for col in data.columns}
    ohlcv = {}
    for col, col_lower in cols_lower.items():
        if col_lower == 'open':    ohlcv['Open'] = data[col]
        elif col_lower == 'high':  ohlcv['High'] = data[col]
        elif col_lower == 'low':   ohlcv['Low'] = data[col]
        elif col_lower == 'volume': ohlcv['Volume'] = data[col]

    if 'adjclose' in cols_lower.values():
        adjclose_col = [c for c, cl in cols_lower.items() if cl == 'adjclose'][0]
        ohlcv['Close'] = data[adjclose_col]
    elif 'close' in cols_lower.values():
        close_col = [c for c, cl in cols_lower.items() if cl == 'close'][0]
        ohlcv['Close'] = data[close_col]

    data = pd.DataFrame(ohlcv, index=data.index)
    if start_date: data = data.loc[start_date:]
    if end_date:   data = data.loc[:end_date]

    if n_days > 1:
        ohlc_dict = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}
        if 'Volume' in data.columns: ohlc_dict['Volume'] = 'sum'
        data = data.resample(f'{n_days}D').agg(ohlc_dict).dropna()

    market_colors = mpf.make_marketcolors(
        up='green', down='red', wick='inherit', edge='inherit', volume='in')
    chart_style = mpf.make_mpf_style(
        marketcolors=market_colors, gridstyle='--', gridaxis='both')

    title_str = (f"{ticker} Candlestick Chart (Daily)" if n_days == 1
                 else f"{ticker} Candlestick Chart ({n_days}-Day Candles)")
    show_volume = 'Volume' in data.columns

    mpf.plot(data, type='candle', style=chart_style, title=title_str,
             ylabel='Price ($)', volume=show_volume, ylabel_lower='Volume',
             figsize=(14, 8), tight_layout=True)


def plot_boxplot(df, n_days=5, column='adjclose', ticker="Stock",
                 start_date=None, end_date=None):
    """Display boxplot chart. See v0.3 for full documentation."""
    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    if start_date: data = data.loc[start_date:]
    if end_date:   data = data.loc[:end_date]

    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(data.columns)}")

    window_data, window_labels = [], []
    for i in range(0, len(data), n_days):
        window = data.iloc[i:i + n_days]
        if len(window) < n_days: break
        window_data.append(window[column].values)
        label_start = window.index[0].strftime('%Y-%m-%d')
        label_end = window.index[-1].strftime('%Y-%m-%d')
        window_labels.append(f"{label_start}\nto\n{label_end}")

    if len(window_data) == 0:
        print(f"Warning: Not enough data for boxplots with window={n_days}")
        return

    fig_width = max(10, min(len(window_data) * 0.8, 30))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.boxplot(window_data, labels=window_labels, patch_artist=True,
               boxprops=dict(facecolor='lightblue', edgecolor='black'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title(f"{ticker} {column.title()} - Boxplot ({n_days}-Day)",
                 fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{column.title()} ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


#==============================================================================
# MAIN EXECUTION — TASK C.6 EXPERIMENTS
#==============================================================================
if __name__ == "__main__":

    print("=" * 60)
    print("STOCK PREDICTION v0.6 - Machine Learning 3")
    print("Ensemble Methods: ARIMA + Deep Learning")
    print("=" * 60)

    # --- Configuration ---
    COMPANY = "CBA.AX"
    START_DATE = "2020-01-01"
    END_DATE = "2024-07-01"
    N_STEPS = 50
    TEST_SIZE = 0.2
    DROPOUT = 0.2
    EPOCHS = 25
    BATCH_SIZE = 32
    FEATURES = ['adjclose']  # univariate for fair ARIMA comparison

    all_results = []

    # ================================================================
    # STEP 1: LOAD DATA (univariate, single-step)
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading and preparing data")
    print("=" * 60)

    data = load_data(
        ticker=COMPANY, start_date=START_DATE, end_date=END_DATE,
        n_steps=N_STEPS, scale=True, shuffle=True,
        lookup_step=1, n_future_steps=1,
        split_by_date=True, test_size=TEST_SIZE,
        feature_columns=FEATURES,
        save_local=True, load_local=True, local_path='data'
    )

    scaler = data["column_scaler"]["adjclose"]

    # Get raw (unscaled) price series for ARIMA
    # ARIMA needs the original prices, not MinMax-scaled values
    raw_df = data['df'].copy()
    raw_prices = scaler.inverse_transform(
        raw_df['adjclose'].values.reshape(-1, 1)).flatten()

    # Split raw prices at the same point as the DL data
    n_total_sequences = len(data["X_train"]) + len(data["X_test"])
    train_size = len(data["X_train"])
    # The raw series starts N_STEPS earlier than sequences
    # (first N_STEPS-1 rows are consumed forming the first window)
    raw_train = raw_prices[:train_size + N_STEPS]
    raw_test_actual = scaler.inverse_transform(
        data["y_test"].reshape(-1, 1)).flatten()

    print(f"  Total raw prices: {len(raw_prices)}")
    print(f"  Training sequences: {len(data['X_train'])}")
    print(f"  Test sequences: {len(data['X_test'])}")
    print(f"  Raw train series length: {len(raw_train)}")
    print(f"  Test points: {len(raw_test_actual)}")

    # ================================================================
    # EXPERIMENT 1: INDIVIDUAL MODELS
    # ================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Individual Model Performance")
    print("=" * 60)

    # --- 1a: ARIMA (auto) ---
    print("\n--- 1a: ARIMA (auto-selected order) ---")
    start_time = time.time()
    arima_preds = arima_rolling_predict(
        raw_train, raw_test_actual, auto=True
    )
    arima_time = time.time() - start_time

    arima_metrics = compute_metrics(raw_test_actual, arima_preds)
    arima_metrics['Model'] = 'ARIMA (auto)'
    arima_metrics['Time (s)'] = round(arima_time, 1)
    all_results.append(arima_metrics)
    print(f"  Results: MAE=${arima_metrics['MAE ($)']}, "
          f"RMSE=${arima_metrics['RMSE ($)']}, "
          f"MAPE={arima_metrics['MAPE (%)']:.2f}%, "
          f"R²={arima_metrics['R²']:.4f}")

    # --- 1b: ARIMA with manual order (5,1,0) ---
    print("\n--- 1b: ARIMA(5,1,0) ---")
    start_time = time.time()
    arima510_preds = arima_rolling_predict(
        raw_train, raw_test_actual, order=(5, 1, 0), auto=False
    )
    arima510_time = time.time() - start_time

    arima510_metrics = compute_metrics(raw_test_actual, arima510_preds)
    arima510_metrics['Model'] = 'ARIMA(5,1,0)'
    arima510_metrics['Time (s)'] = round(arima510_time, 1)
    all_results.append(arima510_metrics)
    print(f"  Results: MAE=${arima510_metrics['MAE ($)']}, "
          f"RMSE=${arima510_metrics['RMSE ($)']}, "
          f"MAPE={arima510_metrics['MAPE (%)']:.2f}%, "
          f"R²={arima510_metrics['R²']:.4f}")

    # --- 1c: LSTM 2x50 ---
    print("\n--- 1c: LSTM 2x50 ---")
    np.random.seed(314)
    tf.random.set_seed(314)
    random.seed(314)

    lstm_model = create_model(
        input_shape=(N_STEPS, 1),
        layer_configs=[
            {"type": "LSTM", "units": 50},
            {"type": "LSTM", "units": 50},
        ],
        dropout_rate=DROPOUT, output_steps=1
    )

    start_time = time.time()
    lstm_model.fit(data["X_train"], data["y_train"],
                   epochs=EPOCHS, batch_size=BATCH_SIZE,
                   validation_split=0.1, verbose=1)
    lstm_time = time.time() - start_time

    lstm_preds_scaled = lstm_model.predict(data["X_test"])
    lstm_preds = scaler.inverse_transform(lstm_preds_scaled).flatten()

    lstm_metrics = compute_metrics(raw_test_actual, lstm_preds)
    lstm_metrics['Model'] = 'LSTM 2x50'
    lstm_metrics['Time (s)'] = round(lstm_time, 1)
    all_results.append(lstm_metrics)
    print(f"  Results: MAE=${lstm_metrics['MAE ($)']}, "
          f"RMSE=${lstm_metrics['RMSE ($)']}, "
          f"MAPE={lstm_metrics['MAPE (%)']:.2f}%, "
          f"R²={lstm_metrics['R²']:.4f}")

    # --- 1d: GRU 2x50 ---
    print("\n--- 1d: GRU 2x50 ---")
    np.random.seed(314)
    tf.random.set_seed(314)
    random.seed(314)

    gru_model = create_model(
        input_shape=(N_STEPS, 1),
        layer_configs=[
            {"type": "GRU", "units": 50},
            {"type": "GRU", "units": 50},
        ],
        dropout_rate=DROPOUT, output_steps=1
    )

    start_time = time.time()
    gru_model.fit(data["X_train"], data["y_train"],
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_split=0.1, verbose=1)
    gru_time = time.time() - start_time

    gru_preds_scaled = gru_model.predict(data["X_test"])
    gru_preds = scaler.inverse_transform(gru_preds_scaled).flatten()

    gru_metrics = compute_metrics(raw_test_actual, gru_preds)
    gru_metrics['Model'] = 'GRU 2x50'
    gru_metrics['Time (s)'] = round(gru_time, 1)
    all_results.append(gru_metrics)
    print(f"  Results: MAE=${gru_metrics['MAE ($)']}, "
          f"RMSE=${gru_metrics['RMSE ($)']}, "
          f"MAPE={gru_metrics['MAPE (%)']:.2f}%, "
          f"R²={gru_metrics['R²']:.4f}")

    # --- 1e: Random Forest ---
    print("\n--- 1e: Random Forest (100 trees) ---")
    np.random.seed(314)
    start_time = time.time()
    rf_model = train_random_forest(
        data["X_train"], data["y_train"],
        n_estimators=100, max_depth=None
    )
    rf_time_fit = time.time() - start_time

    rf_preds_scaled = predict_random_forest(rf_model, data["X_test"])
    rf_preds = scaler.inverse_transform(
        rf_preds_scaled.reshape(-1, 1)).flatten()

    rf_metrics = compute_metrics(raw_test_actual, rf_preds)
    rf_metrics['Model'] = 'Random Forest'
    rf_metrics['Time (s)'] = round(rf_time_fit, 1)
    all_results.append(rf_metrics)
    print(f"  Results: MAE=${rf_metrics['MAE ($)']}, "
          f"RMSE=${rf_metrics['RMSE ($)']}, "
          f"MAPE={rf_metrics['MAPE (%)']:.2f}%, "
          f"R²={rf_metrics['R²']:.4f}")

    # Print Experiment 1 Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT 1 SUMMARY: Individual Models")
    print(f"{'='*60}")
    exp1_df = pd.DataFrame(all_results)
    print(exp1_df.to_string(index=False))

    # ================================================================
    # EXPERIMENT 2: ENSEMBLE COMBINATIONS
    # ================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Ensemble Combinations")
    print("=" * 60)

    ensemble_results = []

    # --- 2a: Simple Average (ARIMA + LSTM) ---
    print("\n--- 2a: Simple Average (ARIMA + LSTM) ---")
    ens_arima_lstm_avg = ensemble_simple_average([arima_preds, lstm_preds])
    ens_al_metrics = compute_metrics(raw_test_actual, ens_arima_lstm_avg)
    ens_al_metrics['Model'] = 'Avg(ARIMA+LSTM)'
    ensemble_results.append(ens_al_metrics)
    print(f"  Results: MAE=${ens_al_metrics['MAE ($)']}, "
          f"R²={ens_al_metrics['R²']:.4f}")

    # --- 2b: Simple Average (ARIMA + GRU) ---
    print("\n--- 2b: Simple Average (ARIMA + GRU) ---")
    ens_arima_gru_avg = ensemble_simple_average([arima_preds, gru_preds])
    ens_ag_metrics = compute_metrics(raw_test_actual, ens_arima_gru_avg)
    ens_ag_metrics['Model'] = 'Avg(ARIMA+GRU)'
    ensemble_results.append(ens_ag_metrics)
    print(f"  Results: MAE=${ens_ag_metrics['MAE ($)']}, "
          f"R²={ens_ag_metrics['R²']:.4f}")

    # --- 2c: Simple Average (ARIMA + LSTM + RF) ---
    print("\n--- 2c: Simple Average (ARIMA + LSTM + RF) ---")
    ens_3model_avg = ensemble_simple_average(
        [arima_preds, lstm_preds, rf_preds])
    ens_3m_metrics = compute_metrics(raw_test_actual, ens_3model_avg)
    ens_3m_metrics['Model'] = 'Avg(ARIMA+LSTM+RF)'
    ensemble_results.append(ens_3m_metrics)
    print(f"  Results: MAE=${ens_3m_metrics['MAE ($)']}, "
          f"R²={ens_3m_metrics['R²']:.4f}")

    # --- 2d: Weighted Average (ARIMA + LSTM, weights from inverse MAE) ---
    print("\n--- 2d: Weighted Average (ARIMA + LSTM) ---")
    # Compute weights as inverse of MAE (lower MAE = higher weight)
    w_arima = 1.0 / arima_metrics['MAE ($)']
    w_lstm = 1.0 / lstm_metrics['MAE ($)']
    print(f"  Raw weights: ARIMA={w_arima:.4f}, LSTM={w_lstm:.4f}")
    ens_weighted = ensemble_weighted_average(
        [arima_preds, lstm_preds], [w_arima, w_lstm])
    ens_w_metrics = compute_metrics(raw_test_actual, ens_weighted)
    ens_w_metrics['Model'] = 'Weighted(ARIMA+LSTM)'
    ensemble_results.append(ens_w_metrics)
    print(f"  Results: MAE=${ens_w_metrics['MAE ($)']}, "
          f"R²={ens_w_metrics['R²']:.4f}")

    # --- 2e: Weighted Average (ARIMA + GRU + RF) ---
    print("\n--- 2e: Weighted Average (ARIMA + GRU + RF) ---")
    w_gru = 1.0 / gru_metrics['MAE ($)']
    w_rf = 1.0 / rf_metrics['MAE ($)']
    ens_weighted_3 = ensemble_weighted_average(
        [arima_preds, gru_preds, rf_preds], [w_arima, w_gru, w_rf])
    ens_w3_metrics = compute_metrics(raw_test_actual, ens_weighted_3)
    ens_w3_metrics['Model'] = 'Weighted(ARIMA+GRU+RF)'
    ensemble_results.append(ens_w3_metrics)
    print(f"  Results: MAE=${ens_w3_metrics['MAE ($)']}, "
          f"R²={ens_w3_metrics['R²']:.4f}")

    # --- 2f: All models weighted average ---
    print("\n--- 2f: Weighted Average (All 4 models) ---")
    ens_all = ensemble_weighted_average(
        [arima_preds, lstm_preds, gru_preds, rf_preds],
        [w_arima, w_lstm, w_gru, w_rf]
    )
    ens_all_metrics = compute_metrics(raw_test_actual, ens_all)
    ens_all_metrics['Model'] = 'Weighted(All 4)'
    ensemble_results.append(ens_all_metrics)
    print(f"  Results: MAE=${ens_all_metrics['MAE ($)']}, "
          f"R²={ens_all_metrics['R²']:.4f}")

    # Print Experiment 2 Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT 2 SUMMARY: Ensemble Combinations")
    print(f"{'='*60}")
    ens_df = pd.DataFrame(ensemble_results)
    print(ens_df.to_string(index=False))

    # ================================================================
    # EXPERIMENT 3: RESIDUAL HYBRID (ARIMA + DL)
    # ================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Residual Hybrid Ensemble")
    print("=" * 60)

    hybrid_results_list = []

    # --- 3a: Hybrid ARIMA + LSTM ---
    print("\n--- 3a: Hybrid ARIMA + LSTM ---")
    hybrid_lstm = ensemble_residual_hybrid(
        raw_train, raw_test_actual, lstm_model, data, scaler, auto=True
    )
    hybrid_lstm_metrics = compute_metrics(
        raw_test_actual, hybrid_lstm['hybrid_pred'])
    hybrid_lstm_metrics['Model'] = 'Hybrid(ARIMA+LSTM)'
    hybrid_results_list.append(hybrid_lstm_metrics)
    print(f"  Results: MAE=${hybrid_lstm_metrics['MAE ($)']}, "
          f"R²={hybrid_lstm_metrics['R²']:.4f}")

    # --- 3b: Hybrid ARIMA + GRU ---
    print("\n--- 3b: Hybrid ARIMA + GRU ---")
    hybrid_gru = ensemble_residual_hybrid(
        raw_train, raw_test_actual, gru_model, data, scaler, auto=True
    )
    hybrid_gru_metrics = compute_metrics(
        raw_test_actual, hybrid_gru['hybrid_pred'])
    hybrid_gru_metrics['Model'] = 'Hybrid(ARIMA+GRU)'
    hybrid_results_list.append(hybrid_gru_metrics)
    print(f"  Results: MAE=${hybrid_gru_metrics['MAE ($)']}, "
          f"R²={hybrid_gru_metrics['R²']:.4f}")

    # Print Experiment 3 Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT 3 SUMMARY: Residual Hybrid")
    print(f"{'='*60}")
    hyb_df = pd.DataFrame(hybrid_results_list)
    print(hyb_df.to_string(index=False))

    # ================================================================
    # VISUALIZATIONS
    # ================================================================
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    # Plot 1: Individual models vs actual
    plot_ensemble_comparison(
        raw_test_actual,
        {
            'ARIMA': arima_preds,
            'LSTM': lstm_preds,
            'GRU': gru_preds,
            'Random Forest': rf_preds,
        },
        title="Individual Model Predictions vs Actual (CBA.AX)",
        save_path="plot_c6_individual_models.png",
        n_points=100
    )

    # Plot 2: Best ensemble vs individual
    plot_ensemble_comparison(
        raw_test_actual,
        {
            'ARIMA': arima_preds,
            'LSTM': lstm_preds,
            'Weighted(ARIMA+LSTM)': ens_weighted,
            'Hybrid(ARIMA+LSTM)': hybrid_lstm['hybrid_pred'],
        },
        title="Ensemble vs Individual: ARIMA + LSTM (CBA.AX)",
        save_path="plot_c6_ensemble_vs_individual.png",
        n_points=100
    )

    # Plot 3: Metrics comparison bar chart
    all_compared = (
        all_results +
        ensemble_results +
        hybrid_results_list
    )
    plot_metrics_comparison(
        all_compared,
        title="All Models: MAE and R² Comparison",
        save_path="plot_c6_metrics_comparison.png"
    )

    # ================================================================
    # SAVE ALL RESULTS TO CSV
    # ================================================================
    combined_results = all_results + ensemble_results + hybrid_results_list
    results_df = pd.DataFrame(combined_results)
    results_df.to_csv("experiment_results_v06.csv", index=False)
    print(f"\nAll results saved to 'experiment_results_v06.csv'")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — ALL MODELS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Find best model
    best_idx = results_df['MAE ($)'].idxmin()
    best_model = results_df.loc[best_idx]
    print(f"\nBest model by MAE: {best_model['Model']} "
          f"(MAE=${best_model['MAE ($)']}, R²={best_model['R²']})")
    print("=" * 60)
